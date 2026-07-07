# M4b.2 — Per-Thread Gap Design (Profiler, Prefill Tiles, Decode Lever)

**Date:** 2026-07-07
**Status:** Approved design, pre-implementation
**Milestone:** M4b.2 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones and the [M4b.1 spec](2026-07-06-m4b1-threading-design.md))

M4b.1 multi-threaded the generated code but its ≥6x scaling exit
criterion proved **unevaluable in this environment**: the dev machine is
a devpod container CPU-quota'd to 8 CPUs with proven mid-run throttling,
and llama.cpp's own prefill scales negatively there. The scaling
questions (exit-criterion evaluation, any parallel-attention amendment)
stay parked on M4b.1's required bare-metal rerun.

The **per-thread gap is measurable right now**: t=1-vs-t=1 comparisons
run both engines in the same environment and are not confounded by the
quota. Recorded baseline (M4b.1 data point, Q8_0 pinned model): inferno
t=1 pp 15.12 vs llama.cpp t=1 pp 54.60 (**0.28x**); tg 9.81–10.50 vs
23.42–23.93 (**0.42–0.44x**). M4b.2 attacks that gap, entirely at t=1.

What the evidence already rules in and out:

- **Prefill has a known structural cause.** Compiled prefill is the
  single-token forward wrapped in a token loop (`lower_prefill`,
  `inferno-codegen/src/llvm/ops.rs`) — every weight matrix is re-read
  from memory once *per token*, 512 full model passes for pp512.
  llama.cpp runs prefill as batched GEMM. Prefill tiles are the
  highest-confidence lever and this spec commits to them.
- **Decode is NOT losing in the GEMV kernel.** The M2 rig recorded the
  Q8_0 rs8 AVX2 GEMV at **1.09–1.72x ggml's throughput** on the hot
  shapes, yet end-to-end decode runs at 0.42–0.44x. The gap must live in
  generated attention/elementwise code, per-token overhead, activation
  quantize, or f32-KV memory traffic — and **no per-op profile exists
  anywhere in the project** (every attempt was blocked: no `perf` binary
  in the container, `perf_event_paranoid=3`). M4b.1's "~5–10% serial
  work" figure is an assertion, not a measurement. So decode levers are
  chosen from data this milestone first produces, not guessed.
- **AVX-512 kernels are out.** They chiefly fix the Q4_K microkernel
  (0.73–0.80x ggml, shuffle-port-bound, needs VNNI per the M2 spec), and
  the pinned nightly model is Q8_0. The `Isa::X86_64v4` / `Feature::Vnni`
  detection plumbed in M2 sits ready for whenever that work is scoped.

## Scope Decisions (M4b.2)

| Decision | Choice |
|---|---|
| Measurement basis | Everything t=1-vs-t=1 under the standard protocol (devenv shell, release, pinned Q8_0 model). Threading/scaling claims remain gated on M4b.1's bare-metal rerun |
| Attribution | **Profiler first.** A built-in, flag-gated per-op profiler lands before any optimization; the baseline decode+prefill profile is recorded in Amendments before tiles work starts |
| Exit criterion (prefill) | **Hard:** t=1 pp ≥ **0.7x** llama.cpp t=1, same protocol, data point recorded in Amendments |
| Exit criterion (decode) | **Scoped:** tg target set by explicit spec amendment *after* the profiler attributes the decode gap, against measured headroom; then met or explicitly re-scoped. No blind number — that is what made M4b.1's criterion unevaluable |
| Levers committed | Prefill/GEMM tiles (incremental: M-loop batch first, register-blocked tiles only by escalation amendment) |
| Levers contingent | Exactly one decode lever picked by the pre-registered attribution fork below: F16 KV, targeted fusion on the blamed ops, or a batched/AVX quantize path |
| Standing invariants | Bit-identity across thread counts (extends to `par_gemm`); scalar-vs-SIMD kernel bit-identity; compiled-vs-interpreter differential green with **no tolerance loosening** (an F16-KV adoption re-derives tolerances, documented, both sides switching together) |

**Explicitly out of scope:** AVX-512 kernel bodies (revisit when Q4_K
matters or bare metal exists), parallel attention (gated on the M4b.1
bare-metal rerun), batched *decode* / continuous batching (v2), NUMA and
thread affinity, in-memory LLJIT, prefix caching, any CI perf gate
(never — AGENTS.md).

## What M4b.2 Adds

- **`inferno-codegen`** — profile instrumentation variant (per-op rdtsc
  accumulation); `lower_prefill` restructured to tile-of-T batching.
- **`inferno-kernels`** — `gemm_*_rs8` batched siblings of the three
  GEMV kernels (f32, q8_0, q4_k), scalar + AVX2; rig extension.
- **`inferno-plan`** — tile size T as a planner constant; prefill
  activation-arena slots sized ×T; matmul byte counts exported for the
  profiler's GB/s column.
- **`inferno-pool`** — `inferno_par_gemm` dispatcher (row-partitioned,
  same shard rules as `inferno_par_gemv`).
- **`inferno-core`** — profile buffer ownership + symbol retention;
  artifact cache key grows the profile flag and T.
- **`cli`** — `--profile` on `inferno run` and `inferno bench`; profile
  table output.
- One decode lever (implementation scoped by amendment).

## The profiler

The compiled forward function is one generated entry point, so only
codegen can see op boundaries. When compiled with profiling enabled,
codegen wraps **each lowered op** — not just each island; GEMV must be
separable from elementwise inside the Ffn island — with `rdtsc` reads
and accumulates the delta into a per-op slot of a `u64` accumulator
array passed through the runtime context. No `perf`, no kernel
counters: it works in this container, which is the point.

**Granularity:** per op kind × site — embed; quantize; each matmul site
by name (qkv, o, gate/up, down, logits); rmsnorm; rope; swiglu; add;
attention. Not per layer (op-kind totals are enough to take the
attribution fork).

**Surface:** `inferno run --profile` and `inferno bench --profile`
print a table: per-op total time, share of wall clock, and — for matmul
sites — **achieved GB/s** computed from planner-known weight bytes. The
GB/s column is the load-bearing one: in-situ GEMV throughput vs the M2
microbench number (43.7 GiB/s for the hot Q8_0 shape) directly separates
"decode loses to per-op overhead" from "decode loses to memory
bandwidth."

**Constraints:**

- Profiling is a compile-time variant: the artifact cache key includes
  the profile flag; a profiled `model.so` is a distinct artifact.
- Instrumentation reads clocks and never touches math: logits must be
  **bit-identical** with and without `--profile` (asserted by test).
- Profiles are self-measurements that guide scoping; they never gate CI.
- rdtsc overhead (~tens of cycles) is negligible against µs-scale ops.

**Non-goals:** sampling, flamegraphs, per-layer breakdown.

## Prefill tiles

**Kernel layer.** Each rs8 GEMV kernel gains a batched sibling
`gemm_*_rs8(m, …)`: outer loop over weight strips (the strip/block inner
code is unchanged), inner loop over the `m` activation rows. Each weight
strip is read once per *batch* instead of once per token — the
structural win. Activations for the `m` tokens are a contiguous panel — quantized
row-by-row with the existing `quantize_row_*` for the quantized dtypes,
raw f32 rows for the f32 kernel (as today). `m` is a runtime argument.

**Determinism by construction.** Per-output-element dot order is
identical to GEMV (same strip/block k-order), so:

- `gemm(m=1)` must **bit-equal** `gemv` (new rig invariant);
- scalar and AVX2 GEMM must be bit-identical (existing rig rule);
- the compiled-vs-interpreter differential tolerances are untouched.

**Codegen.** `lower_prefill` moves from `for token { single-token
forward }` to a tile loop with compile-time tile size **T** (planner
constant, default 64; part of the artifact cache key). The final ragged
tile runs the same code with runtime `m ≤ T` — no second body. Within a
tile:

- embed and elementwise ops (rmsnorm, rope, swiglu, add, quantize) gain
  an outer m-loop over tile rows;
- matmuls call `gemm` with `m` = tile length via `inferno_par_gemm`;
- **attention stays a serial per-position loop**: the tile's K/V rows
  are all written first, then position `p` attends over `[0..p]`.
  Causality holds (K/V for positions ≤ p exist) and the math is
  unchanged from today;
- logits stay last-token-only (GEMV).

**Decode path untouched.** Decode keeps GEMV exactly as-is; zero
regression surface there.

**Memory plan.** The f32 intermediate arena already sizes every
`[Seq, ..]` value to `max_seq_len` rows (liveness-packed), so a tile of
T ≤ n ≤ max_seq_len tokens writes to already-allocated distinct rows —
**no f32-arena change is needed**. The one region that must grow is the
**quantized-activation scratch** (`act_scratch`, `inferno-plan/src/memory.rs`):
today it holds one quantized row; the GEMM activation panel needs T
contiguous quantized rows, so it is sized ×T. Decode is untouched. F32
weights need no panel at all — their per-token source rows are already a
contiguous `stride = k` panel in the arena, read straight by the GEMM.

**Pool.** `inferno_par_gemm` partitions rows across threads exactly like
`inferno_par_gemv`: each output row computed wholly by one thread with
the kernel's fixed combine order, so thread count never changes output
bits — same argument, same exact-equality assertion as M4b.1.

**Escalation clause (pre-registered).** If, with the M-loop GEMM landed,
t=1 pp is still < 0.7x, upgrading hot shapes to register-blocked tiles
(activation panel in registers, M×8 tiles, cache blocking over k)
becomes an **explicit scoped amendment** targeted at the shapes the
profiler blames — the same no-silent-scope rule as M4b.1's
parallel-attention clause.

## Decode attribution fork (pre-registered)

After the profiler lands, the baseline decode profile is recorded in
Amendments and picks **exactly one** lever via explicit amendment:

1. **Attention/KV dominates and in-situ GB/s sits near the memory
   ceiling → F16 KV.** The compiled path *and* the interpreter switch KV
   storage to F16 **together**, so the differential compares like
   against like. `logits_abs_tol` gets a principled re-derivation for
   the one new F16 rounding term in K/V — a documented derivation in
   this spec, never loosened-to-green. This deliberately retires the M3
   note ("KV stays f32 to keep the differential clean") rather than
   eroding it silently.
2. **Generated elementwise/norm/rope/overhead dominates → targeted
   fusion/codegen work** on exactly the blamed ops; the amendment lists
   the op set.
3. **Activation quantize dominates → batched/AVX quantize path.**

The tg exit target is set in the same amendment, against measured
headroom.

## Error handling

No new failure surface beyond existing patterns: `--profile` on a model
compiled without instrumentation recompiles (distinct cache key), it
does not error; `gemm` shares the GEMV kernels' contract (caller
validates shapes at plan time); `inferno_par_gemm` inherits
`inferno_par_gemv`'s dispatch guards.

## Testing Strategy

- **Kernel rig:** GEMM scalar-vs-AVX2 bit-identity over an m×rows×k
  grid including ragged shapes; `gemm(m=1) ≡ gemv` bitwise; proptest
  shapes (M2 pattern).
- **Differential gates (unchanged, must stay green):** the
  `inferno-codegen` compiled-vs-interpreter differential and the
  `inferno-core` artifact test pass with prefill tiles **without any
  tolerance change**.
- **Determinism:** bit-identity across thread counts extends to
  `par_gemm`; logits bit-identical with/without `--profile`; logits
  bit-identical across tile sizes T (new differential axis — tiling
  must not change math).
- **Perf context:** `bench-kernels` gains GEMM criterion benches vs
  ggml. End-to-end numbers only via the manual `mise run bench`
  protocol, recorded in Amendments. No CI perf gates.

## Implementation Phases

1. **Profiler** — codegen instrumentation, cache-key change, CLI
   `--profile`, table output; record the **baseline decode + prefill
   profile** in Amendments before any optimization.
2. **GEMM kernels** — M-loop `gemm_*_rs8` (scalar + AVX2, three
   dtypes), rig extension.
3. **Prefill tiling** — `lower_prefill` tile loop, arena ×T,
   `inferno_par_gemm`, differential + determinism tests.
4. **Prefill data point** — protocol run, recorded in Amendments;
   escalation amendment if pp < 0.7x.
5. **Decode amendment** — attribution fork taken from the recorded
   profile; tg target set; chosen lever implemented; tg data point
   recorded.
6. **Closing data point** — full protocol run recorded in Amendments.

## Risks

- **The M-loop GEMM may not reach 0.7x** if prefill is limited by more
  than weight re-reads (e.g. serial attention share grows as matmuls
  shrink). Mitigation: the escalation clause is pre-registered, and the
  profiler measures prefill too, so the amendment is scoped from data.
- **Decode attribution may be split** (no single dominant cost).
  Mitigation: the fork's amendment picks the largest measured item; the
  tg target is set against that item's headroom, not against a hope.
- **rdtsc-based self-measurement is noisy in a throttled container.**
  Mitigation: shares are meaningful even when absolute times wobble;
  profiles guide scoping and never gate anything.
- **F16 KV touches the differential's foundations.** Mitigation: both
  sides switch together; tolerance change is a documented re-derivation
  reviewed against observed error distributions (the `gemv_rel_tol` /
  `LOGIT_TIE_EPSILON` precedent).

## Amendments

*(Recorded protocol data points and scoped amendments land here; never
edit a recorded data point.)*
