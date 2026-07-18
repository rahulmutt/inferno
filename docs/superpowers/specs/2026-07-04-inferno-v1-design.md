# Inferno v1 Design

**Date:** 2026-07-04
**Status:** Approved design, pre-implementation

## What Inferno Is

Inferno is a CPU-first LLM inference engine written in Rust. Instead of shipping
generic kernels that dispatch on model shape, quantization format, and CPU
features at runtime (llama.cpp's model), Inferno **compiles each model for the
exact machine it runs on**: at setup time it detects the hardware, specializes
the whole computation graph — shapes, quant formats, cache-tiling, memory
layout, thread partitioning all baked in as constants — and emits native code
via LLVM. The compiled artifact is cached on disk; subsequent runs just load it.

Architectural inspirations: IREE (compiler-centric design), MLX (layout and
unified-memory thinking), Burn (Rust ML idioms), vLLM/SGLang (runtime
discipline; their batching ideas are deferred to v2+).

## Scope Decisions (v1)

| Decision | Choice |
|---|---|
| Primary use case | Local, single-user, single-stream inference (server/batching is v2+) |
| Compilation strategy | Generated code (fusion, loops, layout) + hand-tuned microkernels for matmul inner tiles |
| First hardware target | x86-64 (AVX2, then AVX-512); ARM/Apple Silicon (NEON) is v2 |
| Model formats | GGUF **and** MLX (safetensors) both in v1 |
| Model architectures | Llama-family transformer skeleton: Llama 3.x, Qwen 2.5/3, Mistral (RoPE, GQA, RMSNorm, SwiGLU) |
| Quantization formats | Q4_K, Q8_0, F16, BF16, F32 |
| Compilation model | Setup-time JIT (LLVM ORC/LLJIT) + on-disk artifact cache; target descriptor is an explicit input so AOT cross-compile (`--target`) is a v2 drop-in |
| Compiler stack | Own graph + loop IR in Rust, lowered to LLVM IR via `inkwell`; **not** MLIR, **not** Cranelift |
| Surface | Rust library crate (`inferno-core`) + thin CLI (`inferno run/compile/bench/inspect`); no HTTP server in v1 |
| Success criterion | Beat llama.cpp tokens/sec (prefill **and** decode) on the same machine, model, and quant |

Explicitly out of scope for v1: GPU support (permanently out of scope for the
project), continuous batching, paged KV cache, HTTP server, iOS/Android
delivery (design-ready via target descriptors, not built), speculative
decoding, LoRA.

## Architecture

Cargo workspace:

```
inferno/
├── crates/
│   ├── inferno-formats/    # GGUF + MLX(safetensors) parsing → ModelDesc + tensor data
│   ├── inferno-graph/      # Graph IR + per-architecture builders + scalar reference interpreter
│   ├── inferno-target/     # TargetDesc + hardware detection + named target profiles (TOML)
│   ├── inferno-plan/       # Fusion, weight layout planning, memory plan (target-aware)
│   ├── inferno-codegen/    # Loop IR → LLVM IR (inkwell), LLJIT, artifact emission + cache
│   ├── inferno-kernels/    # Hand-tuned microkernels (quant matmul tiles), fixed C ABI
│   ├── inferno-runtime/    # KV cache, tokenizer, sampling, generation loop
│   └── inferno-core/       # Public embeddable API
└── cli/                    # `inferno` binary
```

Data flow:

```
model file ──inferno-formats──▶ ModelDesc + weights
                                     │
target (detected or profile) ───────▶│
                                     ▼
     inferno-graph ─▶ inferno-plan ─▶ inferno-codegen ─▶ CompiledModel artifact (disk cache)
                                                              │
     inferno-runtime ◀──────── dlopen + execute ─────────────┘
                                     │
     inferno-core / CLI ◀────────────┘   generate() → streaming tokens
```

### Boundary rules

- **`inferno-formats` is dumb.** It emits a format-agnostic `ModelDesc`
  (architecture id, hyperparams, named tensors with shape/dtype/quant) plus
  access to raw tensor bytes. Nothing downstream may know which file format a
  model came from. Having both GGUF and MLX in v1 stress-tests this boundary
  from day one.
- **`inferno-target` is pure data + probing.** `TargetDesc` is always an
  explicit input to planning and codegen; no downstream crate re-probes
  hardware. This is the entire cross-compilation interface.
- **`inferno-kernels` exposes a fixed C-callable ABI** — one symbol per
  (operation × quant format × ISA level) microkernel. Generated code calls
  kernels by symbol. Kernels are cargo-built (Rust + core::arch intrinsics,
  asm where needed) and testable with no compiler involvement.
- **The artifact** is a shared object plus a JSON metadata sidecar (model
  hash, target hash, buffer plan, entry-point names, inferno version), cached
  under `~/.cache/inferno/`. Runtime `dlopen`s it.

## Compiler Pipeline

### Graph IR (`inferno-graph`)

A small closed op set — this is an LLM engine, not a general ML framework:
`Embed`, `MatMul` (quantized weights × float activations), `RmsNorm`, `Rope`,
`Attention` (single fused GQA-aware op), `SwiGlu`, `Residual`, elementwise ops,
`Logits`. Quantization formats are **first-class tensor dtypes**;
dequantization is never a graph op — it is fused into the consuming kernel by
construction.

Per-architecture builders map `ModelDesc` → graph. One data-driven Llama-family
builder covers Llama 3.x / Qwen / Mistral via hyperparameters (head counts,
GQA ratio, rope theta, norm eps, vocab size, etc.).

**Reference interpreter.** `inferno-graph` includes a slow, scalar,
obviously-correct interpreter over the graph IR. It exists before any codegen
and is the correctness oracle for everything else.

### What is baked at compile time

Target ISA and cache geometry, all weight shapes, quant formats, head layout,
fusion structure, memory plan, thread partitioning. Runtime-variable: sequence
position/length and nothing else (batch = 1 in v1). llama.cpp dispatches on
all of this at runtime; Inferno makes it constant and lets LLVM fold the world.

### Planning (`inferno-plan`)

All target-aware:

1. **Fusion:** partition the graph into fusion islands — e.g.
   `RmsNorm→MatMul→SwiGlu→MatMul→Residual` becomes one generated function with
   intermediates staying in registers/L1, never round-tripping through memory.
   Attention is one island per layer.
2. **Weight layout:** weights are repacked at compile time from file order
   into each microkernel's preferred tile layout (e.g. block-interleaved for
   AVX-512 Q4_K), written into the artifact's weight image in exact per-layer
   streaming order. Layout tuning is a compile-time transformation, never a
   load-time one.
3. **Memory plan:** a single activation arena with statically computed offsets
   (zero allocation during decode) plus the KV cache layout, sized from a
   compile-time `max_seq_len`.

### Codegen (`inferno-codegen`)

Each fusion island lowers to an explicit loop nest over tiles:

- Outer loops: parallelized across cores by static work partitioning over a
  pinned thread pool.
- Inner matmul tiles: calls into `inferno-kernels` microkernels.
- Everything else (norms, rope, softmax, elementwise): directly generated
  LLVM IR, which vectorizes well when all shapes are constants.

Emitted via `inkwell`. Two paths from the same module: LLJIT for setup-time
compilation, object-file emission for the cached artifact (and later AOT).

**Artifact cache:** keyed by
`hash(model file, quant config, TargetDesc, inferno version)`. Hit → skip
straight to `dlopen`. Compile budget: tens of seconds for a 4–8B model, once
per machine.

## Runtime

### Weights

The artifact's repacked weight image is `mmap`'d read-only; kernels read the
mapping directly (no copies). Layout is exact decode streaming order;
`madvise` sequential ranges at warm-up.

### Entry points

Two compiled entry points per model, generated from the same graph with
different loop strategies:

- `prefill(tokens[], kv, arena)` — batch-of-tokens matmuls, compute-bound,
  wide tiles.
- `decode_step(token, kv, arena, logits_out)` — GEMV-shaped, bandwidth-bound.

### KV cache

Contiguous per layer; logical shape `[layer][kv_head][seq][head_dim]` with the
physical layout chosen by the planner per target. F16 by default. Allocated
once at session start for `max_seq_len`. No paging in v1; sliding-window
models wrap.

### Generation loop (plain Rust, not generated)

```
tokenize → prefill → [sample → decode_step]* → detokenize (streaming)
```

- **Tokenizer:** GGUF-embedded vocab/merges parsed and executed natively
  (BPE and SentencePiece-style); MLX models use `tokenizer.json` via the
  `tokenizers` crate. One `Tokenizer` trait over both.
- **Sampling:** temperature, top-k, top-p, min-p, repeat penalty, seedable RNG.
- **Threading:** one pinned worker pool per engine, sized from `TargetDesc`
  topology (physical cores; compute avoids E-cores by default on hybrid x86).
  Static partitioning; no work-stealing in the hot loop.

### Public API (`inferno-core`)

```rust
let engine = Engine::load("model.gguf", Options::default())?; // detect → compile-or-cache → mmap
let mut session = engine.session(SessionOptions { max_seq_len: 8192, ..Default::default() })?;
for token in session.generate(prompt, sampler)? {             // streaming iterator
    print!("{token}");
}
```

Sessions own their KV cache and arena; one engine hosts multiple sequential
sessions.

## Targets & Hardware Detection

`TargetDesc`: ISA feature flags (x86-64-v3/v4 granularity; later aarch64
NEON/SME), per-level cache sizes, core topology (P/E, SMT), page size, memory
bandwidth class.

Sources: `cpuid` + `/sys` on Linux, `sysctl` on macOS, or a **named profile**
(TOML shipped in `inferno-target`, e.g. `m3.toml`, `snapdragon-8g3.toml`).
Detected and profile-loaded targets are the same struct — that equivalence is
the v2 cross-compile interface. (Note: iOS forbids JIT and loading unsigned
code, so phone delivery is necessarily AOT cross-compile; the design
accommodates it, v1 does not build it.)

## Development Environment & Repository Conventions

These follow the devkit skills (`developer-environment`,
`navigable-codebases`, `writing-clean-code`); the spec records the decisions,
not the commands — task definitions and pins are single-sourced in the files
named below.

### Toolchain: mise-first, devenv.nix for native deps

- **`mise.toml`** (committed) pins every tool to an exact version via
  `mise use --pin`: `rust`, and cargo-distributed dev tools via the cargo
  backend (e.g. `cargo:cargo-nextest`, `cargo:cargo-fuzz`,
  `cargo:cargo-audit`, `cargo:cargo-deny`, `cargo:cargo-insta`). Unpinned
  entries are treated as reproducibility bugs.
- **`devenv.nix`** (committed) provides what mise cannot: **LLVM** (the
  `llvmPackages_<N>` dev output that `llvm-sys`/`inkwell` link against, with
  `LLVM_SYS_<N>0_PREFIX` exported in the shell), supporting native libs
  (`libffi`, `libxml2`, `zlib`), and **llama.cpp** (nixpkgs package) as the
  pinned benchmark opponent. The LLVM major version is pinned here and must
  match the `inkwell` feature flag in `inferno-codegen`; devenv's lockfile
  pins the nixpkgs revision so LLVM and llama.cpp versions are deterministic.
- **Application dependencies** are owned by cargo: `Cargo.lock` is committed;
  exact or lockfile-resolved versions only; automated updater (Renovate or
  Dependabot) gated by CI keeps them current in small steps. Every dependency
  is a liability — prefer std or one small crate over overlapping ones.
- **Build orchestration:** native cargo only. No Bazel — single-language
  workspace, no remote-cache or hermeticity trigger holds.

### Repo front door (navigable-codebases)

Built in M0 and maintained as the repo evolves:

- **README quickstart:** clone → `devenv shell` (or direnv) → `mise install`
  → `mise run test`. The README references task *names*, never re-spells
  commands.
- **Named tasks** in `mise.toml` `[tasks]` for every repeated workflow:
  `test` (blocking tier), `test-full`, `lint`, `fmt`, `bench`, `fuzz`. Tasks
  are the single source of truth; CI invokes the same task names.
- **`AGENTS.md`** (with `CLAUDE.md` pointing at it) carries only the
  non-derivable: pointers to this spec, the threat model, the task names, and
  the LLVM-version-must-match-inkwell constraint.
- **`ARCHITECTURE.md`** is the codebase map: the crate diagram and boundary
  rules from this spec (why the boundaries fall where they do — formats are
  dumb, target is pure data, kernels are a fixed ABI), not a restated file
  tree.
- **Onboarding is verified by running it** — a CI job runs the documented
  clone-to-test sequence from scratch.

### Authoring conventions (writing-clean-code)

`rustfmt` + `clippy` are the source of truth for style (format-on-save,
lint-in-CI); one purpose per crate (the workspace layout above is the domain
decomposition — names come from the domain: `plan`, `target`, `kernels`);
rule-of-three before abstracting; YAGNI ruthlessly — v1 builds no
speculative generality for GPU, batching, or formats beyond the two chosen.

## Testing Strategy

Per `testing-practices`: match each oracle to what we can assert, keep most
weight on cheap layers, and tier the suite by speed.

### Static base

`rustfmt --check` + `cargo clippy` (warnings deny) on every PR; the compiler
and ownership model are the first validation layer. `unsafe` is confined to
`inferno-kernels` (intrinsics), mmap, and `dlopen` glue — each block carries a
documented invariant and gets extra review.

### Oracles

- **Derived (differential) — the load-bearing oracle:** the scalar reference
  interpreter in `inferno-graph` is the trusted implementation; compiled-model
  logits are compared against it (tolerance ~1e-2 on quantized paths, defined
  once per quant format and reused by every layer). llama.cpp output on
  identical GGUF files is a second, external differential oracle.
- **Invariant (property-based, `proptest`):** every microkernel vs the scalar
  reference on random inputs; quant pack/unpack round-trips; tokenizer
  encode/decode round-trips against HF `tokenizers` on the same vocab.
- **Recorded (golden, `insta`):** parsed `ModelDesc` snapshots for GGUF and
  safetensors fixtures; graph-IR dumps after builder and after planning
  (compiler IR is a classic snapshot target); CLI `inspect` output. Snapshots
  stay narrow and deterministic; every change is reviewed, never
  blind-accepted.
- **Specified (unit):** sampling (seeded RNG, exact expected picks), memory
  planner offsets, target detection parsing against captured `cpuid`/`sysctl`
  fixtures.
- **Fuzz (`cargo-fuzz`):** the GGUF and safetensors parsers take untrusted
  bytes — libFuzzer targets from M0, run as a nightly campaign with a corpus
  seeded from the golden fixtures.

Test fixtures include tiny handcrafted models (2 layers, dim 64) in both
formats so the blocking tier never downloads real models.

### Tiers

- **pre-commit / on-save:** fmt + clippy + fast unit tests (seconds).
- **PR (blocking):** unit + integration on tiny fixture models, incl.
  interpreter-vs-compiled differential on the tiny models. Explicit
  wall-clock budget: ≤5 minutes.
- **nightly / scheduled:** fuzz campaigns, real-model differential tests
  (llama.cpp cross-check), `cargo-mutants` on `inferno-plan` and
  `inferno-formats`, full bench protocol, fresh-clone onboarding job.

Flaky tests are quarantined out of the blocking tier immediately, then fixed
or deleted — never blind-retried.

### Performance

`inferno bench` reports prefill tok/s, decode tok/s, peak RSS; a checked-in
protocol benchmarks against the devenv-pinned llama.cpp on the same
model/quant/machine. Criterion micro-benches for kernels from M2.

## Security

Per `security-practices`, scaled to a local-inference library (no network
surface in v1):

- **Threat model** (committed at `docs/threat-model.md` in M0, linked from
  `AGENTS.md`): the primary trust boundary is **model files are untrusted
  input** — a downloaded GGUF/safetensors file may be malicious. Controls:
  parsers validate and canonicalize at the edge (bounds-checked offsets, no
  `unsafe` in `inferno-formats`, allocation limits derived from file size),
  plus the fuzz targets above. Second boundary: the artifact cache under
  `~/.cache/inferno/` is trusted-local — artifacts are keyed and verified by
  content hash before `dlopen`, and loading an artifact is documented as
  equivalent to running code from that cache directory. Out of scope:
  sandboxing generated code against a hostile *local* user.
- **Supply chain:** `cargo audit` (RustSec) + `cargo deny check` (advisories,
  bans, sources, licenses) as a CI gate **and** on a weekly schedule (new
  CVEs land on old code); lockfile committed; updater cadence per the
  environment section.
- **Secret scanning:** `gitleaks` in pre-commit and CI — non-negotiable
  hygiene even with no secrets expected.
- **SAST:** `clippy` security lints + `semgrep --config p/rust` in CI.
  Container/IaC scanning: none — no such artifacts exist.

## Milestones

Each milestone is its own spec → plan → implementation cycle.

- **M0 — Skeleton + formats.** Workspace scaffolding; dev environment
  (`mise.toml` pins, `devenv.nix` with LLVM + llama.cpp, named tasks); repo
  front door (README quickstart, `AGENTS.md`/`CLAUDE.md`, `ARCHITECTURE.md`);
  CI skeleton with the blocking/nightly tiers, scanners (gitleaks,
  cargo-audit/deny, semgrep), and fresh-clone onboarding job; threat model;
  `inferno-formats` parsing GGUF and safetensors/MLX into `ModelDesc` with
  fuzz targets; `inferno inspect` CLI.
- **M1 — Graph IR + interpreter.** Llama-family builder; scalar interpreter
  producing real (slow) tokens end-to-end, incl. tokenizer + greedy sampling.
  *First tokens out.*
- **M2 — Targets + kernels.** Hardware detection; `TargetDesc`; AVX2 (then
  AVX-512) quantized matmul microkernels with the reference-comparison rig.
- **M3 — Compiler.** Planner (fusion, layout, memory plan); codegen; LLJIT;
  artifact cache; `inferno compile`. *First fast tokens.*
- **M4 — Runtime polish + bench.** Full sampling suite; streaming CLI;
  `inferno bench` + llama.cpp comparison protocol. *v1 done when we win.*

**v2 direction (not spec'd here):** NEON/Apple Silicon target, AOT
cross-compilation (`--target <profile>`), then server mode with continuous
batching and prefix caching.

## Risks

- **Beating llama.cpp is hard.** Its kernels are heavily tuned. Mitigation:
  the win must come from specialization (constant shapes, fusion, layout) and
  from microkernels that are competitive, not miraculous. Kernel-level
  criterion benches exist from M2 (compared against llama.cpp's kernel
  throughput) so we know early if the bet is failing, before the full
  end-to-end protocol lands in M4.
- **LLVM dependency weight.** `llvm-sys` pins an LLVM version and slows cold
  builds. Mitigation: LLVM comes prebuilt and version-pinned from
  `devenv.nix` (nixpkgs `llvmPackages_<N>`), never built from source or
  installed ad hoc; the LLVM surface is isolated inside `inferno-codegen`,
  and the devenv-pinned major version must match `inferno-codegen`'s
  `inkwell` feature flag (recorded in `AGENTS.md`).
- **Two formats in v1** doubles loader work. Mitigation: the shared
  `ModelDesc` keeps it to parsing only; MLX safetensors parsing is simple
  compared to GGUF.
- **Quantized-numerics drift** between reference and compiled paths.
  Mitigation: per-format tolerance rules defined once in `inferno-graph` and
  used by every test layer.

## Amendments

### 2026-07-18 — M4 closed; v1 win criterion NOT MET (closing record)

M4 is closed. The v1 win criterion stated in §Scope Decisions — "Beat
llama.cpp tokens/sec (prefill **and** decode) on the same machine, model,
and quant" — is **NOT MET**. Standing ratios against llama.cpp
best-of-builds, from the M4b.16 protocol sessions of 2026-07-18
(Qwen2.5-0.5B-Instruct Q8_0, pp512/tg128, full-thread):

| Machine | pp512 | tg128 |
|---|---|---|
| d2.c1.medium — Xeon Gold 6336Y, 16c | 0.83x | 0.96x |
| s2.c2.medium — Xeon E-2388G, 8c | 0.69x | 0.86x |

The verdict, the M4b findings ledger, and the ceiling arithmetic that
closes each lever family are recorded in
[2026-07-18-v1-close-design.md](2026-07-18-v1-close-design.md). In summary:
every compiled-path streaming lever family is now measured at its wall —
prefill GEMM (M4b.13), prefill attention (M4b.14), decode attention
(M4b.15/M4b.16), and decode GEMV (M4b.17, where the shipping kernel runs
0.5% *above* its own GEMV-shaped roofline). The residual gap is structural,
not unexplored headroom.

§Risks called this outcome: "Beating llama.cpp is hard… the win must come
from specialization… and from microkernels that are competitive, not
miraculous." The specialization bet was tested to a measured ceiling rather
than abandoned, and the kernel-level criterion benches did the job that
section asked of them — showing where the bet stood before the end-to-end
protocol landed.

v1 closes as-built, with no release artifact: no version bump, no tag, no
publish. The v2 direction recorded in §Milestones (NEON/Apple Silicon, AOT
cross-compilation, then server mode) is the successor conversation and is
deliberately not designed here.

This entry is append-only and edits no recorded data point.
