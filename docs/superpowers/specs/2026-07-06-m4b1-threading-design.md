# M4b.1 — Multi-Threaded Generated Code Design

**Date:** 2026-07-06
**Status:** Approved design, pre-implementation
**Milestone:** M4b.1 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones and the [M4a spec](2026-07-06-m4a-bench-sampling-design.md))

M4b ("perf work until we beat llama.cpp prefill **and** decode") is
decomposed by what M4a's first data point showed. The headline gap (pp
0.36x, tg 0.39x) compares inferno at 1 thread to llama.cpp at 12 physical
cores, and even per-thread inferno trails (~0.3–0.5x pp, ~0.44–0.7x tg).
Two workstreams follow: **M4b.1 (this spec) multi-threads the generated
code** — the single biggest lever; **M4b.2 closes the per-thread gap**
(GEMM/prefill tiles, AVX-512 kernels, F16 KV, cost-model fusion), planned
after M4b.1's re-measurement shows where the remaining gap lives.

One expectation set by the M4a data: llama.cpp's own decode barely scales
with threads on the protocol machine (t=1 tg 15.2–23.8 vs t=12 tg 25.0 —
decode is memory-bandwidth-bound), while prefill scales well. Threading's
payoff is therefore lopsided: large for pp, modest for tg. M4b.1's exit
criterion is written accordingly — a prefill *scaling* target, not an
absolute vs-llama.cpp win (that stays with M4b as a whole).

## Scope Decisions (M4b.1)

| Decision | Choice |
|---|---|
| Milestone split | M4b.1 = intra-op row parallelism for the compiled path; M4b.2 = per-thread kernel/codegen quality, planned from M4b.1's re-measured data point |
| Exit criterion | Bit-identity across thread counts asserted in CI; prefill scaling **≥6x at 12 threads** vs inferno t=1 on the protocol machine; new data point recorded in this spec's Amendments. No absolute vs-llama.cpp gate |
| Hook point | **Parallel wrapper symbol** (`inferno_par_gemv`): codegen swaps the direct kernel call for a host-side dispatcher that partitions the row range and calls the unchanged single-threaded kernel per shard. No codegen outlining |
| Thread pool | **Hand-rolled, std-only** persistent fork-join pool (~200 lines): spin-then-park workers, static up-front partitioning, no work-stealing, no queues. No rayon — GEMV shards are ~10µs, dispatch latency and pin/spin behavior must be ours to control, and the repo's precedent (xoshiro) is to own small, stable primitives |
| Crate placement | New crate `inferno-pool` — the existing boundary rule says kernels are single-threaded and *the caller partitions*; this crate is that caller. Third sanctioned-`unsafe` crate (after `inferno-kernels`, `inferno-core`) |
| Determinism | Row-partitioned shards mean each output row is computed entirely by one thread with the kernel's fixed combine order — **thread count never changes output bits**, asserted exactly (not within tolerance) |
| Ops parallelized | `Gemv` only. `Quantize`/`Bias`/`RmsNorm`/`Rope`/`SwiGlu`/`Add` and inline `Attention` stay serial (~5–10% of per-token work combined). If pp scaling stalls below 6x and the profile blames serial attention, parallel attention becomes a scoped follow-up task **inside** M4b.1 — an explicit spec amendment, not silent scope growth |
| Prefill batching | Out of scope. Compiled prefill stays per-token sequential; batched GEMM prefill is M4b.2's "prefill tiles" |

**Explicitly out of scope for M4b.1** (M4b.2 unless noted): GEMM/prefill
tiles, AVX-512 kernels, F16 KV, aggressive/cost-model fusion, in-memory
LLJIT, any CI perf gate (never — AGENTS.md), NUMA awareness and thread
affinity/pinning (single-socket protocol machine; revisit only if the data
demands it).

## What M4b.1 Adds

- **`inferno-pool`** (new crate) — the persistent fork-join pool and the
  `inferno_par_gemv` C-ABI dispatcher.
- **`inferno-codegen`** — the `Gemv` step lowers to a `inferno_par_gemv`
  call (kernel fn ptr as an argument) instead of a direct kernel call.
- **`inferno-core`** — symbol retention for `inferno_par_gemv` (the
  `ensure_kernels_linked` pattern); pool initialization at
  `CompiledBackend` construction.
- **`cli`** — `--threads N` on `inferno run` and `inferno bench`; bench
  report gains a matched-threads inferno headline row and keeps a t=1
  inferno diagnostic row.

## The wrapper (`inferno_par_gemv`)

One symbol, not a per-dtype/ISA family. Generated code currently calls
e.g. `inferno_gemv_q8_0_rs8_avx2(y, xq, w, k, row_start, row_end)` with
the full `0..rows` range. Codegen instead emits:

```c
void inferno_par_gemv(
    GemvFn kernel,      // the same kernel symbol codegen already selects
    float*         y,   // output, one f32 per row
    const uint8_t* xq,  // (quantized) activation
    const uint8_t* w,   // packed weights
    size_t k,
    size_t rows);       // full row count — the wrapper owns the range split
```

The kernel-selection logic in `loopir.rs` (`gemv_symbol`) is untouched —
the selected symbol is now passed as a function-pointer argument rather
than called directly. The wrapper computes the shard table and dispatches;
each shard invocation is a plain call to the single-threaded kernel with
that shard's `row_start`/`row_end`.

Trust model is unchanged from today's direct kernel calls: the wrapper is
called only from generated code whose pointers and shapes M3's codegen
already guarantees; the raw `extern "C"` surface stays unchecked.

## Pool design (`inferno-pool`)

**Topology.** N−1 worker threads plus the calling thread, which executes
shard 0 itself. `threads == 1` (or an uninitialized pool) short-circuits
to a direct single-threaded kernel call on the caller's thread — a
compiled artifact works serially in any host that never initializes the
pool.

**Dispatch (fork-join over an epoch).** The caller publishes one job — a
`SendPtr`-wrapped argument block (kernel fn ptr, dst/w/x pointers, k) plus
the shard table — then bumps an atomic epoch. Workers observe the epoch,
pick their shard by worker index, run it, and decrement an atomic
remaining-count. The caller runs shard 0, then spins on the count (the
other shards are equal-sized, so the wait is short). No queues, no
stealing: every dispatch is one job fully partitioned up front, and the
shard→thread map is a pure function of `(rows, threads)` — deterministic
run-to-run.

**Spin-then-park.** Workers spin for a bounded window (~50–100µs) before
`std::thread::park`. Spinning covers the decode hot loop, where GEMVs
arrive every few hundred microseconds and park/unpark latency would
dominate the ~10µs shards; parking keeps idle CPUs quiet between
generations and in non-generating hosts. The caller unparks only workers
it knows are parked (a per-worker state flag), so the hot path stays
syscall-free.

**Partitioning.** Static contiguous shards, boundaries rounded to the
kernels' 8-row stride (the `rs8` in the symbol names); remainder rows go
to the last shard; `shards = min(threads, ceil(rows/8))` so tiny GEMVs
don't pay idle wakeups. `rows == 0` returns immediately.

**Lifecycle.** Process-global `OnceLock`, initialized explicitly with a
thread count at `CompiledBackend` construction. A second initialization
with a different count is an error at the Rust API level, never a silent
reconfigure. Thread count is clamped to `1..=logical_cores`; the default
is **physical cores** from `inferno-target`'s existing topology detection
(an explicit input, never re-probed — the existing rule).

**Safety.** The crate ships raw pointers across threads (manual
`Send`-wrapper around the job block) and exposes an `extern "C"` entry
point — the sanctioned-`unsafe` surface. Workers run only kernel fn ptrs,
which are panic-free `extern "C"` by existing contract; no unwinding can
cross the ABI.

## Product surface (CLI + bench)

- `--threads N` on `inferno run` and `inferno bench`, clamped to
  `1..=logical_cores`, default physical cores. `inferno run` gets faster
  by default — it is the product surface.
- The **nightly `bench-compiled` speedup gate stays pinned at
  `--threads 1`**: it measures codegen quality against the interpreter,
  and letting threading inflate the ratio would hide codegen regressions
  behind parallelism.
- `inferno bench` (the llama.cpp protocol) runs inferno at **physical
  cores** as the headline row — apples-to-apples with llama.cpp's
  full-thread row for the first time — and keeps an inferno `t=1`
  diagnostic row so per-thread parity stays visible for M4b.2. The JSON
  blob gains the inferno thread count (it already records
  `inferno_threads: 1` today; it becomes the real value, plus the t=1
  diagnostic fields).

## Error handling

Nothing in `inferno_par_gemv` can fail dynamically; degenerate cases are
handled structurally:

- `rows == 0` → immediate return.
- `threads == 1` or pool uninitialized → direct serial kernel call.
- Pool double-init with a different count → Rust-level error at backend
  construction (the only fallible surface).
- Worker panics are impossible by contract (kernels are panic-free
  `extern "C"`); the pool's own worker loop contains no panicking
  operations after startup.

## Testing Strategy

- **Pool unit tests** (no kernels): shard-boundary math (8-alignment,
  remainder, `rows < 8·threads`, `rows == 0`), determinism of the shard
  map, fork-join correctness under repeated dispatch, park/unpark after
  idle windows.
- **Bit-identity tests** (kernel rig): the same GEMV at t=1 / t=4 / t=12
  asserted **exactly equal** per dtype (f32, q8_0, q4_k) — extends the
  existing exact-equality contract ("ISA variants are bit-identical") to
  thread count.
- **End-to-end differential**: the existing compiled-vs-oracle gate runs
  with threads > 1; plus one compiled-vs-compiled run asserting t=1 and
  t=physical produce identical token streams.
- **Stress, not loom/miri**: the concurrency surface is one
  epoch/countdown protocol (~30 lines of atomics); a `--release` stress
  loop (thousands of dispatches, randomized shard counts) is
  proportionate. No CI perf gates — scaling is measured only by the
  manual protocol.

## Implementation Phases

1. `inferno-pool`: pool + shard math + unit/stress tests (pure Rust, no
   integration).
2. `inferno_par_gemv` wrapper + bit-identity tests against the kernel rig.
3. Codegen swap (`Gemv` lowering) + `ensure_kernels_linked` retention +
   differential gates at threads > 1.
4. CLI `--threads`, pool init from `CompiledBackend`, nightly-gate t=1
   pin, bench report/JSON changes.
5. Protocol run on the quiet 3900 box: scaling measurement + new data
   point in Amendments (same rules as M4a: quiet hardware, devenv shell,
   release build, never edit a recorded point; sandboxed agents stop and
   hand back to the human — perf numbers come only from real runs).

## Risks

- **Prefill scaling stalls below 6x.** Serial attention (~5% at pp512) or
  memory-bandwidth saturation could cap scaling short of the target. The
  spec's rule: profile first; if attention is the blocker, parallel
  attention (across heads) becomes an explicit scoped amendment inside
  M4b.1. If bandwidth is the blocker, that is a finding, not a failure —
  record it and let M4b.2 (which changes the compute/byte ratio via tiles
  and F16 KV) inherit it.
- **Decode gains are modest by construction.** The M4a data predicts tg
  scales weakly on this machine. The exit criterion deliberately does not
  gate on tg scaling; the recorded data point quantifies it.
- **Spin windows burn CPU in embedding hosts.** `inferno-core` is the
  embeddable API; a host that generates rarely would see workers spin
  ~100µs per idle transition, then park. Bounded spin + park is the
  mitigation; the window is a named constant to tune if a real host
  complains.
- **Pool double-init in multi-backend hosts.** Two `CompiledBackend`s in
  one process share the global pool; construction with a mismatched
  thread count errors loudly rather than silently reconfiguring. If a
  real embedding use case needs per-backend pools, that is a v2 API
  question, not M4b.1.

## Amendments

*(Recorded protocol data points and scoped amendments land here; never
edit a recorded data point.)*
