# M3 — Compiler Design

**Date:** 2026-07-05
**Status:** Approved design, pre-implementation
**Milestone:** M3 (see [inferno v1 design](2026-07-04-inferno-v1-design.md) §Milestones)

M3 is the compiler: it connects the graph and kernels built in M0–M2 into a
specialized, native, cached artifact and executes it. *First fast tokens.*

## Scope Decisions (M3)

| Decision | Choice |
|---|---|
| Spec/plan shape | One design spec (this doc) covering the whole compiler; the implementation plan is **phased** (planner → codegen → cache/CLI wiring) |
| Success criterion | Compiled path produces tokens matching the scalar interpreter within the per-quant tolerance, end-to-end via `inferno compile`, and runs **materially faster than the interpreter**. Beating llama.cpp, the full bench protocol, and perf tuning are **M4** |
| Prefill entry point | Compiled as **looped GEMV** — the single-token forward wrapped in a token loop, reusing the M2 GEMV kernels. Real GEMM/prefill tiling is M4 |
| Threading | **Single-threaded** generated code. Kernels are already row-range partitioned, so the static multi-thread partition over a pinned pool is an additive **M4** drop-in |
| Fusion | **Framework + conservative islands** — build the real fusion-island + arena machinery, partition along the design's named boundaries, no cost-model / cross-op register tricks (M4) |
| Activation arena | **Single arena, liveness-packed** — values share byte ranges when lifetimes don't overlap (the v1 design's stated model) |
| Compilation path | **Object-emit + `dlopen` only** — emit an object from the inkwell module, link to `model.so`, `dlopen` it; the written artifact *is* the cache entry. The v1 design's separate in-memory LLJIT path is a **documented deviation**, deferred as a later cold-start latency optimization (see Deviations) |

**Explicitly out of scope for M3** (deferred, mostly to M4): AVX-512 kernels,
GEMM/prefill tiles, multi-threaded generated code, aggressive/cost-model fusion,
in-memory LLJIT, the full `inferno bench` protocol and llama.cpp performance
comparison, the full sampling suite (M4), AOT cross-compile (`--target`, v2),
and any GPU/server/batching work (out of v1 entirely).

## Deviations from the v1 design, resolved during brainstorming

The v1 design (2026-07-04) describes the compiler at the end state. Two points
are narrowed for M3 and amend the v1 design's implementation expectation, not
its architecture:

1. **Object-emit + `dlopen` is the single compilation path.** The v1 design
   names both LLJIT (in-memory, setup-time) and object emission (for the
   cache). M3 implements only the latter: on a cache miss it compiles to
   `model.so` in the cache dir and `dlopen`s it. "Setup-time compilation"
   therefore happens via disk. In-memory LLJIT becomes a later cold-start
   latency optimization; dropping it removes a second execution path and
   failure mode from the riskiest milestone. The inkwell module and object
   emission it needs are exactly the durable interface an LLJIT path would
   reuse.
2. **Prefill is not GEMM-tiled.** M2 built only GEMV (`rs8`, decode-shaped)
   kernels. M3's compiled prefill loops those over the prompt tokens rather
   than introducing GEMM microkernels. Correct and fully compiled; not
   optimally tiled. Real prefill tiles are M4 perf work.

## What M3 Adds

Two new crates and one public-API crate, plus runtime wiring:

- **`inferno-plan`** — target-aware planning. `(&ModelDesc, &Graph,
  &TargetDesc, max_seq_len) → Plan`. Pure Rust, **no LLVM**.
- **`inferno-codegen`** — `Plan → Loop IR → LLVM IR (inkwell) → object → linked
  model.so`. The only crate that links LLVM.
- **`inferno-core`** — the embeddable public API: `Engine::load` (detect →
  compile-or-cache → mmap → backend) and sessions.

The interpreter in `inferno-graph` is **unchanged** and remains the correctness
oracle every compiled result is measured against.

### Data flow (new in M3)

```
ModelDesc + Graph ──inferno-plan──▶ Plan ──inferno-codegen──▶ Artifact dir
   (M0/M1)                          (islands,     (Loop IR → LLVM      (model.so +
                                     repacked      via inkwell;         weights.bin +
                                     weight image, object emit +        meta.json),
                                     memory plan)  link)                cached under
                                                                        ~/.cache/inferno/
                                                                              │
                                                          inferno-core Engine │
                                                    verify hashes → dlopen ───┘
                                                          → CompiledBackend
```

## `inferno-plan`

Pure Rust, target-aware, no LLVM. Produces a `Plan` — data only, with a
`Plan::dump()` text form (mirroring `Graph::dump()`) as the snapshot surface.
Three responsibilities:

### 1. Fusion islands (conservative)

Partition the flat, topologically-ordered node list into islands along the
v1 design's named boundaries, using fixed rules keyed on `Op` — **not** a
general cost-model fusion engine (that is M4):

- Each **`Attention` node is its own island** (one per layer): it is already a
  fused GQA op and owns the KV cache.
- The **attention projection block** `RmsNorm → MatMul(qkv) → Rope` fuses into
  one island.
- The **FFN block** `RmsNorm → MatMul(gate/up) → SwiGlu → MatMul(down) → Add`
  fuses into one island so intermediates stay in the arena.
- `Embed` and the final `RmsNorm → MatMul(logits)` are their own islands.

Islands are `Vec<Island { nodes: Range<usize>, .. }>`. The partitioner is a
single forward pass with fixed pattern rules; its output is snapshot-stable.
The exact rule set is validated against the tiny-fixture graphs during
implementation and recorded in `Plan::dump()` snapshots.

### 2. Weight-layout repacking

For every `MatMul` weight, call the M2 registry
`kernels_for(dtype, target.isa).pack(bytes, rows, k)` to obtain the `rs8`-packed
`AlignedBuf`, and lay all packed weights end-to-end into a single **weight
image** (`weights.bin`) in decode-streaming order. The `Plan` records per
weight `(offset, len, rows, k, dtype, isa)`.

This is the compile-time layout transform the v1 design mandates (never a
load-time one). It is also **required**: the kernels accept only packed
weights, so repacking is the load-bearing part of the planner.

### 3. Static memory plan

- **Activation arena:** one buffer; each graph value gets a byte offset from a
  **liveness sweep** over the node list, so a value may reuse a dead value's
  slot. Sized for the compile-time `max_seq_len` prefill footprint (decode is
  `Seq=1`). The arena also reserves **scratch slots for quantized activations**
  (`q8a`/`q8k`) consumed by the quantized-MatMul kernel calls. Zero allocation
  during decode.
- **KV cache:** physical layout `[layer][kv_head][seq][head_dim]`, F16,
  contiguous per layer, sized from `max_seq_len`. The planner picks concrete
  strides (a single sensible layout for M3) and records total bytes + per-layer
  offsets.

### Output & testing surface

`Plan { islands, weight_image_layout, arena_layout, kv_layout }`, snapshotted
via `Plan::dump()` on the tiny fixtures (the classic "IR after planning"
golden). Unit tests assert arena offsets/liveness and cover the KV layout math;
a property test asserts repacked weights round-trip through the kernel's own
unpack (extending M2's pack/unpack property tests).

## `inferno-codegen`

Owns the **Loop IR** and the LLVM backend. `Plan → Loop IR → LLVM IR (inkwell)
→ object → link → model.so`.

### Loop IR

Each island lowers to a `LoopNest`: an ordered list of steps over the arena.
Each step is either:

- a **kernel call** — `quantize_row` (`q8a`/`q8k`) then `gemv`
  (`inferno_gemv_{f32,q8_0,q4_k}_rs8_{scalar,avx2}`), with `rows`, `k`, and
  offsets baked as constants and `row_start=0, row_end=rows` (single-threaded);
  the f32 kernel skips the quantize step (activations are raw f32 LE bytes); or
- a **generated op** — `Embed`, `RmsNorm`, `Rope`, `Attention`, `SwiGlu`, `Add`
  lowered directly to LLVM IR. All shapes are compile-time constants, so LLVM
  vectorizes these well.

The Loop IR has a text dump → **blocking-tier snapshot, LLVM-independent**. This
gives a cheap correctness/regression surface before any LLVM is involved.

### Entry points — prefill is the decode body in a token loop

The single-token forward is causal: processing prompt token `t` after writing
the KV cache for positions `0..t` is exactly correct. So codegen produces **one
island lowering** and two entry points from it:

- `decode_step(token, pos, weights, kv, arena, logits_out)` — the body once.
- `prefill(tokens, n, pos_off, weights, kv, arena, logits_out)` — the same body
  inside `for t in 0..n`, writing logits only for the final token.

`pos`/`pos_off` (sequence position) is the only runtime-variable input; batch is
1. No GEMM — every matmul is a GEMV kernel call.

### LLVM / linking

Emit an object file from the inkwell module, then invoke the system linker
(from the devenv toolchain) to produce `model.so`. LLVM major version is pinned
by devenv (18) and **must** match the `inkwell` feature flag — the standing
AGENTS.md constraint. The LLVM surface is confined to this crate.

## Artifact, cache, and security

**Artifact = a cache directory** `~/.cache/inferno/<key>/` containing:

- `model.so` — the compiled entry points.
- `weights.bin` — the packed weight image, `mmap`'d read-only at run time;
  kernels read the mapping directly (no copies).
- `meta.json` — sidecar: model hash, target hash, weight-image hash,
  entry-point names, buffer/arena/KV plan, and inferno version.

**Cache key** = `hash(model file, quant config, TargetDesc, inferno version)`.
On load, `inferno-core` verifies every hash in `meta.json`; on any mismatch it
refuses the artifact and recompiles.

**Security** (per [threat model](../../threat-model.md), trusted-local
boundary): `dlopen` of `model.so` is native code execution. The control is
hash-verification before load; loading an artifact is documented as equivalent
to running code from the cache directory. Sandboxing generated code against a
hostile *local* user is out of scope, consistent with the v1 design.

## Runtime integration

### Backend trait

Today `Generator` (`inferno-runtime`) hard-codes `interp: Interpreter` and calls
`interp.run(&desc, &graph, tokens, &mut kv) -> Tensor` for both prefill and each
decode step. M3 extracts a small backend abstraction —

```rust
trait Backend {
    /// Advance the sequence by `tokens`, returning logits for the last token.
    fn forward(&mut self, tokens: &[u32]) -> Result<Tensor>;
}
```

with two implementations:

- **`InterpBackend`** — wraps today's `Interpreter` + `KvCache`; behaviour
  unchanged. Stays reachable as the oracle.
- **`CompiledBackend`** — owns the `mmap`'d weight image, the `dlopen`'d entry
  points, and its **own** arena + KV cache in the planner's physical layout.

The generation loop (`tokenize → prefill → [sample → decode]* → stream`) becomes
backend-agnostic; each backend hides its own KV/arena. Sampling, tokenizer, and
streaming are untouched.

### `inferno-core`

`Engine::load(model, options)` → detect target (or named profile) →
compile-or-load-cache → `mmap` → build `CompiledBackend`. `Engine::session(..)`
owns a session's KV/arena. This is the v1 `Engine`/`session` API, introduced now
because M3 is where compile-or-cache-then-execute first exists.

### CLI

- `inferno compile <model>` — run plan + codegen, populate the cache, print the
  artifact path.
- `inferno run <model> <prompt>` — defaults to the **compiled** backend
  (compile-if-missing via `Engine::load`); `--interp` forces the interpreter
  (for differential / debugging).
- `inferno diff` — gains a **compiled-vs-interpreter** mode alongside the
  existing interpreter-vs-llama.cpp teacher-forced differential.

## Testing Strategy

Leans on the existing oracle (the scalar interpreter) and the tiered suite from
the v1 design.

- **Differential — the gate (blocking):** compiled logits vs the interpreter on
  the tiny fixtures (2-layer, dim-64, **both** GGUF and MLX), within the
  per-quant tolerance already defined in `inferno-graph/src/tolerance.rs`. Add a
  fixture with `n_kv_heads < n_heads` so the generated **Attention** (fused GQA
  — the trickiest generated op) is exercised.
- **Snapshots (insta):** `Plan::dump()` after planning and the **Loop-IR dump**
  after codegen — both LLVM-independent, blocking tier. LLVM IR text is
  nightly-only (verbose, toolchain-sensitive).
- **Unit:** arena offsets/liveness; weight-image layout; cache-key determinism;
  `meta.json` verification (tamper a hash → load refuses).
- **Property:** planner weight-repack round-trips through the kernel's own
  unpack (extends M2's pack/unpack property tests).
- **Faster-than-interpreter — the success gate (nightly):** on a real small
  model (reuses M1's model-download infra), assert compiled decode tok/s beats
  interpreter tok/s by a margin. Tiny fixtures are too small to time
  meaningfully, so the blocking tier proves *correctness* and nightly proves
  *faster*.
- **End-to-end CLI test:** `inferno run` (compiled) and `inferno run --interp`
  produce identical greedy tokens on a fixture.

The blocking tier must still run without downloading a real model — the tiny
fixtures compile fast. Codegen tests need the devenv shell (LLVM) present.

## Implementation Phases

The plan (written next, via the writing-plans skill) follows three sequential
phases, each independently testable:

1. **`inferno-plan`** — islands + weight repack + liveness memory plan +
   `Plan::dump()` snapshots. Pure Rust, no LLVM. De-risks half of M3 before
   inkwell.
2. **`inferno-codegen`** — Loop IR + LLVM lowering + object-emit + `dlopen` +
   the two entry points. Differential vs interpreter on tiny fixtures. The
   LLVM-risk phase.
3. **Cache + `inferno-core` + CLI + backend trait** — `Engine::load`, cache
   key/verify, `inferno compile`, `run --interp`, backend-agnostic generation
   loop, the faster-than-interp nightly, and docs updates (AGENTS.md,
   ARCHITECTURE.md, README status).

## Risks

- **LLVM/inkwell is the project's hardest dependency.** Isolated to
  `inferno-codegen`; LLVM 18 (devenv) must match the inkwell feature flag (the
  AGENTS.md constraint). Cold-build weight is the known, accepted cost.
- **New runtime dependency: a linker / C toolchain** for the *first* compile
  (devenv provides it; cached runs need nothing). Documented and surfaced
  clearly when absent.
- **Quantized-numerics drift** compiled-vs-interpreter — mitigated by the
  per-quant tolerance already tuned in `tolerance.rs`; the differential is the
  guard, never loosened to make a red test green.
- **Arena liveness bugs → memory corruption** — guarded by offset unit tests
  and the differential; AddressSanitizer on the compiled path as an optional
  nightly.
- **Generated Attention correctness** (fused GQA over the KV cache) is the
  trickiest lowering — the GQA-shaped multi-layer fixture differential is the
  specific guard.

## Amendments

_(Record data points and post-implementation corrections here, as M2 did.)_

- **KV stored as f32 in M3 (not F16)** to keep the compiled-vs-interpreter
  differential free of an F16 rounding term; F16 KV is deferred to M4.
- **First speedup data point (Task 17, dev Ryzen 9 3900, 24 logical CPUs,
  2026-07-05):** `inferno bench-compiled` on Qwen2.5-0.5B-Instruct Q8_0 GGUF
  (the pinned nightly model), prompt "The capital of France is", 48 decode
  tokens, release build, decode tok/s via `GenStats::decode_secs`:
  compiled 26.10 tok/s vs interpreter 2.68 tok/s — **9.72x** (a repeat run:
  26.11 vs 2.71, 9.64x). This is real, locally-observed, not fabricated —
  the nightly's own first run is the durable record going forward.
  `MARGIN = 3.0` (`cli/src/bench.rs`) was left at the plan's conservative
  starting point rather than raised toward the observed ~9.6-9.7x: it stays
  a floor with generous headroom against CI-runner noise while still
  catching a real regression, per the Task 17 brief's discipline (set
  conservatively *below* the observed value, never tuned down to pass).
