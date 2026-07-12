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

- **M4b.1 scaling data point (Task 10, dev machine, 2026-07-06; probes
  2026-07-07):** Ran the protocol inside `devenv shell` (release build via
  `mise run bench`) against the pinned nightly model
  `qwen2.5-0.5b-instruct-q8_0.gguf` (qwen2 1B Q8_0,
  `/home/dev/.cache/inferno-tests/qwen2.5-0.5b-instruct-q8_0.gguf`),
  defaults pp=512, tg=128, reps=5, matched threads t=12. inferno commit
  `c7533a1`, llama.cpp build `6f4f53f` (devenv-pinned, BLAS + CPU-haswell
  backends). CPU reports as AMD Ryzen 9 3900 12-Core Processor (12
  physical / 24 logical cores) — but see the environment finding below;
  this run was NOT on quiet bare metal.

  Load conditions, honestly: a first table-run attempt was killed at 300s
  by the harness wrapper's own timeout (no numbers were taken from it);
  the recorded table run started while the machine's load average was
  still decaying from that kill (~20 falling), with no other compute
  processes running. The `--json` run and the thread sweep ran later at
  load ~13 (decaying) and ~3.5 respectively. Runs were serialized, never
  concurrent.

  Table run (`mise run bench -- <MODEL>`):

  ```
  model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
  cpu: BLAS, AMD Ryzen 9 3900 12-Core Processor (12 physical / 24 logical cores)
  inferno 0.1.0 (c7533a1) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

  engine                 threads        pp512 tok/s        tg128 tok/s
  inferno (compiled)          12       13.35 ± 2.84        10.50 ± 1.22
  inferno (t=1 diag)           1       15.12 ± 1.32         9.81 ± 0.95
  llama.cpp                   12       42.97 ± 0.76        23.62 ± 0.31
  llama.cpp (t=1 diag)         1       54.60 ± 1.30        23.93 ± 0.03

  ratio (inferno/llama.cpp): pp 0.31x | tg 0.44x
  ```

  `--json` run (independent invocation, same protocol):

  ```json
  {
    "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
    "model_type": "qwen2 1B Q8_0",
    "cpu_info": "BLAS, AMD Ryzen 9 3900 12-Core Processor",
    "physical_cores": 12,
    "logical_cores": 24,
    "inferno_version": "0.1.0",
    "inferno_git": "c7533a1",
    "llama_build_commit": "6f4f53f",
    "pp": 512,
    "tg": 128,
    "reps": 5,
    "inferno_threads": 12,
    "llama_threads": 12,
    "inferno_pp_tok_s": 16.76005294016336,
    "inferno_pp_stddev": 1.1772039645048684,
    "inferno_tg_tok_s": 11.013822165886967,
    "inferno_tg_stddev": 0.6783627100663344,
    "llama_pp_tok_s": 43.379163,
    "llama_pp_stddev": 1.553997,
    "llama_tg_tok_s": 25.58065,
    "llama_tg_stddev": 1.276047,
    "llama_t1_pp_tok_s": 51.06514,
    "llama_t1_pp_stddev": 2.673074,
    "llama_t1_tg_tok_s": 23.42188,
    "llama_t1_tg_stddev": 0.300833,
    "inferno_t1_pp_tok_s": 14.453983947533555,
    "inferno_t1_pp_stddev": 1.5966267328710673,
    "inferno_t1_tg_tok_s": 10.014636769269071,
    "inferno_t1_tg_stddev": 1.0616405783546694
  }
  ```

  **Computed scaling factors (headline vs inferno t=1):** table run pp
  **0.88x**, tg **1.07x**; `--json` run pp **1.16x**, tg **1.10x**. The
  two runs disagree on whether t=12 prefill is slightly slower or
  slightly faster than t=1; either way threading is a wash.
  **The ≥6x prefill-scaling exit criterion is NOT met in this
  environment** — but read the environment finding before drawing any
  design conclusion from that.

  **Attribution (perf unavailable: no `perf` binary in the container and
  `perf_event_paranoid=3`; used a real thread sweep + cgroup accounting
  instead).** Thread sweep via `inferno run --threads N` (648-token
  prompt, 64 generated, 2 reps each, means):

  ```
  threads   pp tok/s   pp scale   tg tok/s   tg scale
  1         14.70      1.00x      9.48       1.00x
  2         17.72      1.21x      10.70      1.13x
  4         19.17      1.30x      11.11      1.17x
  8         18.24      1.24x      10.86      1.15x
  12        15.32      1.04x      10.03      1.06x
  ```

  Prefill rises to a shallow peak at t=4 (~1.3x) then regresses.

  **Environment finding (the headline of this data point): the "quiet
  dev machine" is a devpod container whose root cgroup is CPU-quota'd to
  8 CPUs** (`/sys/fs/cgroup/cpu.max` = `800000 100000`), with recorded
  throttling (`nr_throttled` 13k+ periods, ~6400s cumulative
  `throttled_usec`) and nonzero external CPU pressure
  (`/proc/pressure/cpu` some avg10 ≈ 11–15 during the runs). Two direct
  probes sampled `cpu.stat` around single `inferno run` invocations:

  - t=12 prefill (45.7s): `nr_throttled` +164, `throttled_usec` +14.5s —
    the quota actively throttles the 12-thread configuration, which
    explains the t=8→12 regression (spin-synced fork-join degrades
    badly when shard-holding workers are descheduled mid-dispatch).
  - t=4 prefill (37.6s): **zero** additional throttling, yet scaling is
    still only ~1.2–1.3x — so below the quota there is a second,
    environment-or-engine ceiling this data cannot separate.

  Corroboration that the ceiling is (at least partly) environmental:
  **llama.cpp's own prefill scales NEGATIVELY here** — t=1 pp 54.60 vs
  t=12 pp 42.97 (0.79x) in this run, reproducing the same signature in
  the M4a amendment's recorded rows (t=1 pp 53.71/31.30 vs t=12 pp
  45.56/43.37). A mature engine whose prefill scales near-linearly on
  real hardware does not scale in this container; the environment cannot
  demonstrate multicore scaling for ANY engine. Retroactive note for the
  ledger: this means M4a's data point was taken under the same 8-CPU
  quota; its inferno-vs-llama ratios compare both engines within the
  same environment and remain internally meaningful, but its "llama.cpp
  prefill scales well with threads" expectation is contradicted by its
  own recorded numbers.

  **Conclusion, plainly:** measured prefill scaling at t=12 is 0.88x
  (table) / 1.16x (json) vs the ≥6x target — not met here. However, the
  protocol machine assumption (quiet bare-metal 12-core) is violated:
  the container's 8-CPU quota (throttling proven mid-run) plus host CPU
  pressure confound the measurement, and llama.cpp's negative scaling in
  the same runs shows the environment itself cannot exhibit scaling.
  Whether inferno's threading meets the 6x criterion is **unknowable
  from this environment**. The spec-mandated attribution fork (serial
  attention → scoped amendment vs memory bandwidth → M4b.2 finding)
  cannot be taken from confounded data. Per the spec's no-silent-scope
  rule, no optimization was started. **Required follow-up: rerun this
  protocol on genuinely quiet, unquota'd bare metal (or a container with
  ≥12 dedicated CPUs) before evaluating the M4b.1 exit criterion or
  scoping any parallel-attention amendment.** The headline vs llama.cpp
  ratios (pp 0.31–0.39x, tg 0.43–0.44x) are essentially unchanged from
  M4a, consistent with threading contributing nothing in this
  environment.

### 2026-07-11 — quiet-hw verdict (M4b.7 gate-prefill-scaling, bare metal): ≥6x @ t=12 NOT MET

First clean read of the exit criterion — quiet bare metal via `mise run
metal` (d2.c1.medium, Xeon Gold 6336Y, 16 physical / 32 logical, PREFLIGHT
FIT: unquota'd, throttled_delta 0), inferno @ 6b0df49. **Prefill scale @
t=12 = 4.06x against the ≥6x target — NOT MET.** Full sweep: 1.76x@2,
2.62x@4, 3.56x@8, 4.06x@12 — sublinear from t=2 onward, so this is not a
tail-off at high t but a slope deficit throughout.

**The attribution fork (serial attention vs memory bandwidth) is now due
but could not be taken from this run alone:** the llama.cpp corroboration
column is confounded — its pp *falls* with threads (407.7 @ t=1 → 285.8 @
t=12) and the bench header reports a BLAS cpu backend, i.e. llama-bench
likely multithreads internally at t=1, so it cannot serve as the "does a
mature engine scale on this box" control. Follow-up: take the fork with a
clean control (non-BLAS llama build) or direct attention-fraction
profiling before scoping any parallel-attention amendment.

```
# gate-prefill-scaling (M4b.1 ≥6x @ t=12) — 2026-07-11T12:25:44Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-11

| t | pp tok/s | pp scale | tg tok/s | tg scale | llama pp (corrob.) | llama tg (corrob.) |
|---|---|---|---|---|---|---|
| 1 | 61.5623236240181 | 1.00x | 22.749259780322767 | 1.00x | 407.726001 | 23.172656 |
| 2 | 108.23660143376955 | 1.76x | 28.45561921587664 | 1.25x | 161.24184 | 35.229527 |
| 4 | 161.18434631113706 | 2.62x | 28.359880981833737 | 1.25x | 174.658449 | 43.444397 |
| 8 | 219.05631710077432 | 3.56x | 28.39016188663262 | 1.25x | 217.505463 | 51.740286 |
| 12 | 249.8517236661377 | 4.06x | 45.78206946658722 | 2.01x | 285.810925 | 54.773961 |

gate: prefill scale @ t=12 = 4.06x (target ≥6x) -> NOT MET
note: on a MET=no result, take the M4b.1 spec's attribution fork (serial attention vs memory bandwidth) — see its Amendments.
```

### 2026-07-11 — second quiet-hw session (fixed comparator): NOT MET reproduced at 4.11x; ATTRIBUTION FORK TAKEN → serial fraction, not bandwidth

Same box type as the morning session (d2.c1.medium, Xeon Gold 6336Y,
PREFLIGHT FIT), inferno @ 1804d9f, now with the pure-CPU llama.cpp
comparator (the BLAS confound is fixed — see the M4a same-day
amendments). **Prefill scale @ t=12 = 4.11x (vs 4.06x this morning) —
NOT MET, reproduced.**

**The fork is now decidable and is taken: the deficit is inferno's
serial fraction (serial attention), NOT memory bandwidth.** The clean
control scales on the same silicon in the same session: llama.cpp pp
goes 118.6 → 1109.8 tok/s @ t=12 (~9.4x) while inferno goes 61.4 →
252.6 (4.11x). A bandwidth-bound box cannot produce a 9.4x pp curve, so
the spec's bandwidth branch is eliminated. Per the no-silent-scope rule
this authorizes scoping the parallel-attention amendment as follow-up
work (not started here). Footnotes for the record: the llama curve is
mildly superlinear (t=1 may be depressed by its own overhead floor) and
its t=16 pp carries a ±242 stddev — neither changes the direction by
enough to matter against a 2.3x gap.

```
# gate-prefill-scaling (M4b.1 ≥6x @ t=12) — 2026-07-11T20:20:55Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-11

| t | pp tok/s | pp scale | tg tok/s | tg scale | llama pp (corrob.) | llama tg (corrob.) |
|---|---|---|---|---|---|---|
| 1 | 61.445261793030646 | 1.00x | 22.149875294979513 | 1.00x | 118.604394 | 22.963362 |
| 2 | 109.70235179363456 | 1.79x | 28.582096839747617 | 1.29x | 232.769258 | 30.105505 |
| 4 | 159.99859332670934 | 2.60x | 28.34614277313001 | 1.28x | 445.223471 | 49.738081 |
| 8 | 220.20366614545551 | 3.58x | 28.35285556520114 | 1.28x | 810.536708 | 58.216329 |
| 12 | 252.56605806102783 | 4.11x | 45.60576209003862 | 2.06x | 1109.83946 | 60.881269 |

gate: prefill scale @ t=12 = 4.11x (target ≥6x) -> NOT MET
note: on a MET=no result, take the M4b.1 spec's attribution fork (serial attention vs memory bandwidth) — see its Amendments.
```

### 2026-07-11 — parallel-attention amendment scoped as M4b.8

The follow-up the fork authorized is designed in
[M4b.8 — Parallel Prefill Attention](2026-07-11-m4b8-parallel-attention-design.md)
(own doc per repo convention). The ≥6x @ t=12 exit criterion stays owned
by this spec; M4b.8's quiet-hw verdict will be recorded here.

### 2026-07-12 — third quiet-hw session, post-M4b.8: 5.67x @ t=12 — NOT MET; attribution: remaining serial ops

Same box type as both prior sessions (d2.c1.medium → Xeon Gold 6336Y,
16 physical / 32 logical, PREFLIGHT FIT), inferno @ 823437f — the first
run with M4b.8's parallel prefill attention. **Prefill scale @ t=12 =
5.67x against the ≥6x target — NOT MET, but up from 4.06x/4.11x on the
same silicon**, with t=1 unchanged (61.2 vs 61.4–61.6 tok/s — no serial
regression) and absolute t=12 throughput 249.9→346.8 tok/s (+39%).

**The attribution fork is taken this session, both ways ruled decisive:**

- *Memory bandwidth of the box*: ruled out. The pure-CPU llama.cpp
  control (fixed comparator, second-session follow-up) scales 117.7 →
  1082.6 pp tok/s = **9.2x @ t=12 on this box** — the hardware supports
  well past 6x at these throughput levels.
- *Dispatch overhead vs remaining serial ops* (the M4b.8 spec's item-3
  fork): the sweep shape is a **slope deficit persisting across t**
  (1.90x@2, 3.07x@4, 4.72x@8, 5.67x@12 — near-ideal at t=2, deficit
  growing with t), not low-t degradation with high-t recovery. That is
  the remaining-serial-ops signature. Amdahl fit: residual serial
  fraction ≈ 10.2%; the 6x line needs ≈ 9.1% — the top of the M4b.8
  spec's predicted 5–10% band, missed by ~1pp of serial work.

Per the M4b.8 spec's Risks section, the recorded attribution authorizes
**rope/norm/append parallelization as the next lever** — not loosening
the gate. Scoping deferred to a dedicated design (M4b.9 candidate).

```
# gate-prefill-scaling (M4b.1 ≥6x @ t=12) — 2026-07-12T08:06:10Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-12

| t | pp tok/s | pp scale | tg tok/s | tg scale | llama pp (corrob.) | llama tg (corrob.) |
|---|---|---|---|---|---|---|
| 1 | 61.180962962430556 | 1.00x | 24.36003144785473 | 1.00x | 117.727526 | 24.548485 |
| 2 | 116.05132249378607 | 1.90x | 30.262036892614805 | 1.24x | 233.513123 | 31.238757 |
| 4 | 187.9838116436776 | 3.07x | 29.934194425207068 | 1.23x | 449.758712 | 49.208576 |
| 8 | 288.520376217837 | 4.72x | 30.09539046002045 | 1.24x | 827.517343 | 57.428537 |
| 12 | 346.8010463910017 | 5.67x | 47.17741791944487 | 1.94x | 1082.552108 | 59.288741 |

gate: prefill scale @ t=12 = 5.67x (target ≥6x) -> NOT MET
note: on a MET=no result, take the M4b.1 spec's attribution fork (serial attention vs memory bandwidth) — see its Amendments.
```

### 2026-07-12 — fourth quiet-hw session, post-M4b.9: 10.63x @ t=12 — **MET**. Gate closed.

Same box type as all three prior sessions (d2.c1.medium → Xeon Gold 6336Y,
16 physical / 32 logical, PREFLIGHT FIT), inferno @ 2387266 — the first
run with M4b.9's parallel serial tail (rmsnorm/rope/add/swiglu/bias/embed,
KV-append, activation-quantize all token-sharded through
`inferno_par_token_loop`). **Prefill scale @ t=12 = 10.63x against the ≥6x
target — MET**, up from 5.67x (M4b.8) / 4.11x / 4.06x on the same silicon.
The M4b.1 exit criterion is satisfied; no further lever is authorized or
needed for this gate.

- **t=1 unchanged: 61.37 pp tok/s** vs 61.18 (M4b.8) and 61.4–61.6 (earlier
  sessions). The outlining regression risk the M4b.9 spec named
  (call-boundary opacity costing single-thread codegen quality) **did not
  materialize** — this is the same measurement the `bench-compiled` t=1
  nightly guards, taken on quiet hardware.
- **Absolute t=12 throughput 346.8 → 652.4 pp tok/s (+88%).**
- **The dispatch-count-growth risk did not materialize either.** M4b.9
  predicted its detector as low-t degradation with high-t recovery; the
  observed low-t rows are at or near ideal (2.00x @ t=2, 3.92x @ t=4), so
  the ≈2.5x growth in joins per layer per tile is not visible even where it
  would be most costly. Job fusion (approach C) stays unexercised.
- **Amdahl residual serial fraction ≈ 1.2%**, down from ≈10.2% at M4b.8 —
  the serial tail was the deficit, exactly as the third session's
  attribution fork concluded. (The 6x line needed ≈9.1%.)
- **Scaling slope vs llama.cpp:** the pure-CPU control scales 118.7 →
  1064.8 = 8.97x @ t=12 on this box; inferno now out-scales it (10.63x),
  though from a lower t=1 base. Absolute pp gap at t=12 narrowed from 3.12x
  behind to 1.63x behind (652.4 vs 1064.8).
- Decode (tg) is untouched by M4b.9 by construction; the tg column moved
  47.2 → 45.6 tok/s @ t=12, within the cross-session spread and not a
  code-path change. Decode's own verdict lives in the M4b.5 gate.

```
# gate-prefill-scaling (M4b.1 ≥6x @ t=12) — 2026-07-12T14:36:00Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-12

| t | pp tok/s | pp scale | tg tok/s | tg scale | llama pp (corrob.) | llama tg (corrob.) |
|---|---|---|---|---|---|---|
| 1 | 61.371907264832984 | 1.00x | 22.333697855392508 | 1.00x | 118.720164 | 16.419338 |
| 2 | 122.81768427088039 | 2.00x | 28.465759382766066 | 1.27x | 230.487011 | 40.107204 |
| 4 | 240.2871894480028 | 3.92x | 28.19155852084818 | 1.26x | 444.989896 | 46.910133 |
| 8 | 460.1357171111038 | 7.50x | 28.403966472986003 | 1.27x | 801.32421 | 53.084828 |
| 12 | 652.3823384803542 | 10.63x | 45.604824021519946 | 2.04x | 1064.793598 | 54.868367 |

gate: prefill scale @ t=12 = 10.63x (target ≥6x) -> MET
```
