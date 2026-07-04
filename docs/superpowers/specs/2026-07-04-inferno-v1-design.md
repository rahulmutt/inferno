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

## Testing Strategy

- **Kernel tests:** every microkernel vs the scalar reference on random
  inputs; property-based; explicit tolerance rules per quant format.
- **End-to-end golden tests:** compiled-model logits vs reference-interpreter
  logits (tolerance ~1e-2 for quantized paths); spot-check outputs against
  llama.cpp on identical GGUF files as an external oracle.
- **Tokenizer conformance:** round-trip against HF `tokenizers` for the same
  vocab.
- **Format tests:** golden GGUF/safetensors fixtures, including tiny
  handcrafted models (2 layers, dim 64) so CI stays fast without downloading
  real models.
- **Perf:** `inferno bench` reports prefill tok/s, decode tok/s, peak RSS;
  a checked-in protocol benchmarks against llama.cpp on the same
  model/quant/machine. Criterion micro-benches for kernels.

## Milestones

Each milestone is its own spec → plan → implementation cycle.

- **M0 — Skeleton + formats.** Workspace scaffolding; `inferno-formats`
  parsing GGUF and safetensors/MLX into `ModelDesc`; `inferno inspect` CLI.
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
  builds. Mitigation: prebuilt LLVM in the devcontainer; the LLVM surface is
  isolated inside `inferno-codegen`.
- **Two formats in v1** doubles loader work. Mitigation: the shared
  `ModelDesc` keeps it to parsing only; MLX safetensors parsing is simple
  compared to GGUF.
- **Quantized-numerics drift** between reference and compiled paths.
  Mitigation: per-format tolerance rules defined once in `inferno-graph` and
  used by every test layer.
