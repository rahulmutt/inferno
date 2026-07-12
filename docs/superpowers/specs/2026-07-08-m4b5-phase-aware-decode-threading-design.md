# M4b.5 — Phase-Aware Decode Threading Design

**Date:** 2026-07-08
**Status:** Approved design, pre-implementation
**Milestone:** M4b.5 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.4](2026-07-08-m4b4-decode-gemv-mlp-design.md))

M4b.1 multi-threaded the compiled path by row-sharding every `Gemv`
dispatch across the host thread pool. That machinery is already live on
the decode path: every decode-step matmul is dispatched through
`inferno_par_gemv`, sharded across `active_threads()` (physical cores by
default). This milestone does **not** add parallel decode — it
*right-sizes* the parallelism that already runs, because the M4b.1 data
shows the current default is decode's **worst** configuration.

## Motivation — decode is bandwidth-bound, and full-core sharding regresses

Decode is a single-row GEMV: one token, each weight byte read exactly
once, no reuse. That makes it **memory-bandwidth-bound**, not
compute-bound — adding cores past the point that saturates DRAM
bandwidth buys nothing and eventually *costs* (fork-join barrier overhead
+ CPU-quota throttling on shard-holding workers). The M4b.1 thread sweep
(recorded amendment, dev box, qwen2.5-0.5b Q8_0) shows exactly this on
the decode (tg) axis:

| threads | tg tok/s | scale |
|---|---|---|
| 1 | 9.48 | 1.00× |
| 2 | 10.70 | 1.13× |
| 4 | 11.11 | **1.17× (knee)** |
| 8 | 10.86 | 1.15× |
| 12 | 10.03 | **1.06× (regresses)** |

Two facts follow:

1. **The default is the worst realistic choice.** `inferno run` defaults
   `--threads` to `0` → physical cores → decode runs at t=12 (1.06×)
   instead of its measured optimum near t=4 (1.17×). We leave ~10% of
   decode throughput on the floor *and* pay the high-thread regression,
   for free, today.
2. **Threading is not the lever that closes the llama.cpp decode gap.**
   llama.cpp's own decode does not scale with threads either (recorded:
   t=1 tg 23.9 ≈ t=12 tg 23.6). The ~2.2× tg gap vs llama.cpp is a
   *single-thread bytes-per-token* gap, not a thread-count gap.
   Row-sharding more cores cannot close it, and this milestone does not
   claim to. Closing that gap is a separate lever (single-thread
   bytes/token: the M4b.4 GEMV follow-through, or F16 KV) — see Out of
   Scope.

So M4b.5's job is narrow and honest: cap decode's thread count at a
tuned, bandwidth-saturating value so decode stops running past its knee,
while prefill keeps every core. It is a bit-neutral, reversible,
regression-removing change — hygiene plus a foundation that pays off on
high-bandwidth (many-channel / multi-CCX / server) hardware where the
knee sits at a higher core count.

## Scope Decisions (M4b.5)

| Decision | Choice |
|---|---|
| Lever | **Phase-aware thread count.** Decode (`inferno_par_gemv`) shards over `min(active_threads, decode_cap)`; prefill (`inferno_par_gemm`) stays at full `active_threads`. Single global pool, one new cap |
| Dispatch boundary | Pure **pool-side** change at the decode dispatch site. `inferno_par_gemv` is emitted **only** on the compiled `decode_step` entry point (prefill tiles lower to `inferno_par_gemm`), so capping the GEMV dispatcher caps decode and nothing else |
| ABI / codegen | **No change.** `inferno_par_gemv`'s `extern "C"` signature is untouched; it reads the cap from the global pool it already resolves. No codegen edit, no `HOST_ABI_VERSION` bump, **no recompile** — an existing cached `model.so` benefits immediately |
| Cap default | **Deferred to a quiet-hardware sweep** (Task 1), same discipline as M4b.4's `PF_DIST`. Ship a reversible starting hypothesis, `clamp(active/3, 2, active)` (reproduces the one known knee, 4/12), overridable by `INFERNO_DECODE_THREADS` |
| Tuning surface | Env var `INFERNO_DECODE_THREADS=N` (mirrors `PF_DIST`/`PF_DIST_F32`). No CLI flag in v1 |
| Prefill path | **Unchanged.** `par_gemm` keeps full `active_threads`; the M4b.1 ≥6× prefill-scaling gate stays separately gated on its own quiet-hardware rerun |
| Correctness — scalar vs AVX2 | **Exact equality**, unchanged (kernels untouched) |
| Correctness — thread count vs output | **Bit-identical for every cap value.** `shard_table` computes each output row entirely within one lane; the cap only regroups rows, never changes per-row math |
| Correctness — compiled vs interpreter | Differential + artifact stay green with **no `logits_abs_tol` / `gemv_rel_tol` loosening**. No new tolerance constant, no `observed_error_*` sweep — bits are identical |
| Exit criterion | Two-legged: **(1)** correctness gates green + new cap-invariance test, no tolerance touched (provable on this box); **(2)** a quiet-hardware decode-thread sweep confirming the capped default meets-or-beats the best fixed thread count and removes the high-thread regression, t=1 decode unchanged, prefill unchanged (deferred) |
| Standing invariants | scalar-vs-SIMD GEMV bit-identity; gemm(m=1)-vs-gemv bit-identity; compiled-vs-interpreter differential green with no tolerance loosening; the interpreter unchanged |

**Explicitly out of scope:**

- **Closing the tg gap vs llama.cpp** — single-thread bytes-per-token, a
  different lever (F16 KV / GEMV bytes). This milestone makes no tg-win
  claim.
- **Prefill threading** (`par_gemm`) — untouched; the M4b.1 ≥6×
  prefill-scaling gate stays on its own gate.
- **Auto-tune at model load** (measure the knee per-machine at startup) —
  the most per-machine-correct option, but adds startup cost and
  nondeterminism. Documented follow-up, not v1.
- **`memory_bw_class`-keyed default** — a natural refinement once profile
  builds carry it (`None` on auto-detect today). Optional enhancement,
  see Design §Default.
- **CLI `--decode-threads` flag** — the env var is the v1 surface; a flag
  is a later add if wanted.
- **F16 KV cache, register-blocked GEMM** — unchanged existing gates.

## Design

### Mechanism

One knob on the global pool, applied at exactly one dispatch site.

**`inferno-pool` (`pool.rs`, `lib.rs`):**

- `Shared` gains `decode_cap: AtomicUsize` alongside the existing
  `active`.
- `Pool::par_gemv` changes its shard-count line from
  `let active = self.active_threads();` to
  `let active = self.active_threads().min(self.decode_threads());`.
  The job/epoch protocol, `shard_table`, wakeups, and spin/park loop are
  all untouched.
- `Pool::par_gemm` (prefill) is **not touched** — it keeps
  `active_threads()`.
- New `set_decode_threads(n)` / `decode_threads()` mirror the existing
  `set_active_threads` / `active_threads`, clamped to `[1, capacity]`.
- `extern "C" fn inferno_par_gemv`'s signature is **unchanged**; it reads
  `decode_cap` from the global pool. `inferno_par_gemm` is likewise
  unchanged.

**`inferno-core` (`CompiledBackend`):** at pool init, set `decode_cap`
from `INFERNO_DECODE_THREADS` if present and parseable to a value in
`[1, capacity]`, else the heuristic default.

Because the cap lives on the runtime pool and the kernel ABI is
untouched, this composes with `set_active_threads(1)` (the `bench` t=1
diagnostic): `min(1, decode_cap) = 1`, so the t=1 path is unaffected.

### Default value

The optimal cap is `total_DRAM_bandwidth / per_core_streaming_bandwidth`,
and inferno auto-detects neither. Any shipped default is therefore a
hypothesis, and — following the `PF_DIST` precedent — the **final default
is deferred to a quiet-hardware decode-thread sweep** (Task 1). We ship a
reversible, env-overridable default meanwhile.

- **Starting hypothesis:** `decode_cap = clamp(active / 3, 2, active)`.
  This reproduces the only knee we have measured (t=4 on 12 physical
  cores ≈ ⅓) and scales with core count instead of being a magic
  constant, degrading sensibly on both small and large machines.
- **Env override:** `INFERNO_DECODE_THREADS=N` forces the cap — the
  tuning surface, matching `PF_DIST` / `PF_DIST_F32`.
- **Optional refinement (noted, not built):** when a profile build
  carries `TargetDesc.memory_bw_class` (`Consumer` / `Workstation` /
  `Server`), key the default off it instead of the fraction. Left out of
  v1 because auto-detect yields `None`.

The default only has to clear a low bar to be a strict improvement: the
status quo (`active` = physical cores) is *proven* to regress (t=12 =
1.06× vs t=4 = 1.17×). Any cap at or below the knee removes that
regression.

### Correctness & bit-neutrality

The cap regroups output rows into shards; it never changes the per-row
computation. `shard_table` already guarantees each output row is computed
entirely within one lane, for any thread count, so decode output is
**bit-identical regardless of the cap value**. The full numeric contract
carries over untouched:

- **No tolerance constant touched, no `observed_error_*` sweep** — there
  is no numeric surface (identical bits).
- Existing gates stay green as-is: scalar≡AVX2 rig proptest,
  `inferno-codegen` differential (5/5), `inferno-core` artifact (4/4).
- **New unit test** in `inferno-pool/tests/par_rig.rs`: `par_gemv` output
  is invariant across `decode_cap` values (sweep the cap `1..=capacity`
  for a representative `(rows, k)`, assert byte-identical `y`). This locks
  the bit-neutrality claim at the unit level.

### Tasks (rough order)

1. **Diagnostic (measurement).** Decode-thread sweep on the pinned
   qwen2.5-0.5b Q8_0 model; record the knee. Directional-only on the
   shared devpod, load-bearing on quiet hardware → finalizes the default
   formula/constant.
2. **Pool change.** `decode_cap` atomic + `set/get` + env init + the one
   `min` in `par_gemv`. `par_gemm` untouched.
3. **Wiring.** `CompiledBackend` sets the default cap (heuristic) unless
   `INFERNO_DECODE_THREADS` overrides.
4. **Tests.** New `par_rig` cap-invariance test; confirm differential +
   artifact stay green.
5. **Docs.** Document `INFERNO_DECODE_THREADS` (`README`, `AGENTS.md`);
   record the quiet-hardware sweep in this spec's Amendments.

## Exit Criterion

Two-legged, split by what is provable where.

**Leg 1 — Correctness (load-bearing, provable on this box):**
`inferno-codegen` differential 5/5 + `inferno-core` artifact 4/4 green
with no tolerance loosened; the new `par_gemv` cap-invariance test green;
scalar≡AVX2 rig unaffected. This is the milestone's hard contract and it
is fully verifiable on the shared devpod.

**Leg 2 — Performance (deferred to quiet hardware):** a decode-thread
sweep on quiet bare metal recording (a) the measured knee, (b) that the
capped default meets-or-beats the best fixed thread count *and*
eliminates the high-thread regression, (c) t=1 decode unchanged, (d)
prefill (`par_gemm`) throughput unchanged. The shipped default is a
reversible hypothesis; the constant/formula is finalized from this
recorded sweep. On the quota'd devpod any before/after is
directional-only — not a verdict — as in M4b.1–M4b.4.

## Risks

- **The one knee is machine-specific.** `active/3` is fit to a single
  12-core consumer box; a many-channel server's knee sits higher and a
  dual-channel laptop's lower. Mitigation: the env override is the escape
  hatch, and the default is explicitly a deferred-to-quiet-hardware
  hypothesis, not a claim. The `memory_bw_class` refinement is the
  principled successor once detection exists.
- **Regression measurement is unavailable on the devpod.** The box is
  CPU-quota'd and throttled (M4b.1 environment finding), so the perf leg
  cannot be judged here. Mitigation: Leg 1 (correctness) is fully
  provable on-box; Leg 2 is honestly deferred, same as every M4b perf
  verdict since M4b.1.
- **Reading this milestone as a decode "win."** It is not — it removes a
  regression and right-sizes; it does not close the tg gap. Mitigation:
  stated plainly in Motivation and Out of Scope so no downstream ledger
  entry over-claims.

## Amendments

*(Recorded protocol data points and scoped amendments land here; never
edit a recorded data point.)*

### 2026-07-08 — Task 6: exit criterion — correctness PROVEN, performance verdict DEFERRED (directional knee reproduces)

Final M4b.5 task. Splits into a load-bearing correctness gate (reliable on
this box) and a performance verdict (deferred to quiet hardware per the
standing controller/user decision that this shared, CPU-quota'd 24-core
devpod cannot produce a trustworthy absolute perf number). The directional
sweep this time is unusually clean because varying `INFERNO_DECODE_THREADS`
at a fixed `--threads` is a *controlled same-box manipulation* (only the
decode cap moves), which the standing discipline treats as the trustworthy
signal — but the final default constant and the formal Leg 2 verdict still
defer to quiet bare metal.

**Leg 1 — Correctness (LOAD-BEARING, PROVEN here, no tolerance touched):**
the cap only regroups output rows into shards; `shard_table` keeps each row
on one lane, so decode output is bit-identical for every cap.

- `cargo nextest run -p inferno-codegen -E 'binary(differential)'` →
  **5/5 PASS** (`differential_tiny_{gguf,mlx,bias}`,
  `profiling_does_not_change_logits`,
  `prefill_tiling_is_bit_invariant_to_tile_size`).
- `cargo nextest run -p inferno-core -E 'binary(artifact)'` → **4/4 PASS**
  (`compiled_prefill_matches_interpreter`,
  `concurrent_compile_publishes_atomically`,
  `prefill_past_max_seq_len_panics_not_oob`, `tampered_meta_is_rejected`
  — the last being the pre-existing ~1-ULP flaky integrity test carried
  since M4b.3; passed this run).
- `cargo nextest run -p inferno-pool -E 'test(decode_cap) + test(bit_invisible) + test(bit_invariant)'`
  → **6/6 PASS**, including the new `q8_0_decode_cap_is_bit_invisible`
  (sweeps `decode_cap` 1..=12 on a 12-lane pool, asserts every capped
  dispatch is `.to_bits()`-identical to one serial kernel call) plus the
  three existing dtype `*_thread_count_is_bit_invisible` locks.
- No tolerance constant, ABI, or `HOST_ABI_VERSION` touched on the branch:
  `git diff main..HEAD -- crates/inferno-graph/src/tolerance.rs crates/inferno-codegen/src/lib.rs`
  = empty. Whole-workspace `mise run test` = 265/265 (3 skipped) at Task 4.
  The milestone's numeric contract holds.

**Leg 2 — Performance (DIRECTIONAL only — NOT a verdict):** decode-cap
sweep on the pinned qwen2.5-0.5b Q8_0, short prompt, 64 decode steps,
`--threads 12` fixed, `--profile`, single run per cap (no repetition; noisy
shared devpod):

| `INFERNO_DECODE_THREADS` | decode wall (s) | decode GEMV `lm_head` GB/s | decode scale vs cap=1 | prefill wall (s) |
|---|---|---|---|---|
| 1 | 3.542 | 10.3 | 1.00× | 0.035 |
| 2 | 2.052 | 18.3 | 1.73× | 0.033 |
| **4** | **1.556** | **25.3** | **2.28× (knee)** | 0.032 |
| 8 | 1.822 | 23.2 | 1.94× | 0.030 |
| 12 | 2.723 | 16.5 | 1.30× (regresses) | 0.038 |

Directional reading (same-box relative, single-shot — no error bars):

- **The predicted knee reproduces at cap = 4.** Decode is fastest at cap=4
  (1.556 s, ~25.3 GB/s single-row GEMV), matching the spec's `active/3`
  hypothesis for a 12-core box exactly. The shipped heuristic
  `(12/3).max(2).min(12) = 4` lands on the measured optimum.
- **The high-thread regression the milestone removes is visible.** cap=12
  (the pre-M4b.5 default = physical cores) is 2.723 s / 16.5 GB/s — **1.75×
  slower decode than cap=4** — i.e. the old default ran decode at its worst
  point on this box, as M4b.1 predicted. cap=8 already regresses off the
  cap=4 peak (1.822 s), so the knee is genuinely near 4, not a plateau.
- **Prefill is unaffected by the decode knob (structural, reliable here).**
  Prefill wall stays flat (~0.030–0.038 s, noise) across every cap value —
  `par_gemm` is uncapped and always uses full `--threads 12`, exactly as
  designed. Varying a decode-only cap leaves prefill flat: this is a
  same-box structural observation, not a throttled absolute, so it holds
  independent of box contention.
- **t=1 decode is unchanged by construction.** cap=1 forces single-thread
  decode (`min(active, 1) = 1`); the bench t=1 diagnostic composes as
  before (`min(1, decode_cap) = 1`).

**Why still DEFERRED despite the clean knee:** absolutes are
contention-depressed (peak ~25 GB/s here vs the box's known throttling; a
quiet Ryzen would read higher) and this is a single unrepeated run per cap.
The *shape* (knee at ~⅓ cores, high-thread regression, prefill-flat) is
trustworthy same-box; the *exact optimal constant* and the formal Leg 2
sign-off (meets-or-beats best fixed count with error bars, t=1 explicitly
held, prefill throughput unchanged under load) are finalized on a quiet,
unquota'd bare-metal re-run — the reversible `active/3` default and the
`INFERNO_DECODE_THREADS` escape hatch ship meanwhile.

**Verdict:**
- **Correctness — MET (proven here):** `decode_cap` bit-invisible (new
  `par_gemv` cap sweep 1..=12) + compiled≡interpreter differential 5/5 +
  artifact 4/4, no tolerance loosened.
- **Performance — DEFERRED (directional knee reproduces at 4):** the shipped
  `active/3` default is a reversible hypothesis that lands on this box's
  measured optimum and removes the proven high-thread regression; the final
  constant/formula and formal Leg 2 verdict are deferred to a quiet-hardware
  re-run, together with the `memory_bw_class`-keyed refinement if built.

### 2026-07-11 — quiet-hw Leg 2 verdict (M4b.7 gate-decode-cap, bare metal): default does NOT meet-or-beat; formula revision deferred

Quiet bare metal via `mise run metal` (d2.c1.medium, Xeon Gold 6336Y, 16
physical / 32 logical, PREFLIGHT FIT), inferno @ 6b0df49, reps=3
interleaved. Leg 2 sub-verdicts:

- **(b) default meets-or-beats best fixed: NOT MET.** Knee = cap 13
  (63.98 tok/s median); shipped `clamp(active/3, 2, active)` → cap 5 on
  this box → 55.93 tok/s, **−9.82% vs best fixed**. The measured knee is
  ≈ 0.8×active here vs the ⅓×active hypothesis (whose only prior support
  was the knee=4-on-12-cores point from the quota'd devpod).
- **(b) high-thread regression: gone.** Uncapped-equivalent cap=16 =
  61.03 tok/s, −4.6% vs knee — a mild tail, not the cliff the cap was
  built against; on this box the cap prevents little and costs 9.8%.
- **(c) t=1 decode unchanged:** t1 = 23.79 median (one noisy rep at
  17.17), consistent with cap=1 (23.49) — no t=1 damage.
- **(d) prefill unchanged:** not re-measured here; covered by
  gate-prefill-scaling the same session (M4b.1 amendment).

**Verdict: the `active/3` constant is wrong on this machine class, but
one quiet knee point is not enough to pick a replacement — the formula
revision is deferred to a scoped follow-up with at least one more quiet
machine (ideally a different core count / memory class).**
`INFERNO_DECODE_THREADS` remains the override; users on 16-core Ice
Lake-SP-class boxes should set it ≈ 13 meanwhile.

```
# gate-decode-cap (M4b.5 default-vs-best sweep) — 2026-07-11T12:42:47Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-11
sweep: caps={1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 } + default + t1 | reps=3 (interleaved rounds) | max-tokens=128

| cap | decode tok/s (median of 3) | per-rep |
|---|---|---|
| 1 | 23.49 | 23.53 23.44 23.49 |
| 2 | 40.88 | 33.78 41.42 40.88 |
| 3 | 42.06 | 42.28 42.06 41.02 |
| 4 | 50.61 | 50.61 51.80 49.81 |
| 5 | 56.28 | 56.66 56.28 55.91 |
| 6 | 57.82 | 57.82 60.03 57.52 |
| 7 | 60.07 | 60.07 61.38 58.76 |
| 8 | 59.89 | 59.89 62.83 59.27 |
| 9 | 60.63 | 60.63 63.60 59.55 |
| 10 | 60.69 | 60.56 63.09 60.69 |
| 11 | 63.22 | 61.73 63.22 64.31 |
| 12 | 62.7 | 61.56 64.33 62.70 |
| 13 | 63.98 | 61.92 64.06 63.98 |
| 14 | 63.53 | 61.45 63.53 63.85 |
| 15 | 60.99 | 60.94 60.99 62.15 |
| 16 | 61.03 | 61.03 59.20 63.91 |
| default | 55.93 | 55.84 55.93 58.00 |
| t1 | 23.79 | 24.14 23.79 17.17 |

knee (best fixed cap): 13 (63.98 tok/s median)
default clamp(active/3,2,active): 55.93 tok/s median -> -9.82% vs best fixed (median of per-rep ratios)
gate inputs (human verdict to M4b.5 Amendments): default meets-or-beats
best-fixed? high-thread regression gone (compare cap=16 row vs knee)?
t=1 decode unchanged (t1 row vs prior recorded t=1)?
```

### 2026-07-11 — second quiet-hw session: Leg 2 miss reproduced (knee 12, default −11.19%)

Second session on the same box type (d2.c1.medium, PREFLIGHT FIT),
inferno @ 1804d9f. Knee = cap 12 (63.88 tok/s; morning session said 13 —
the 8–16 plateau is flat, both sit on it), default = 56.73 →
**−11.19% vs best fixed (morning: −9.82%). The Leg 2 miss is
reproduced; verdict unchanged** — the `active/3` constant undershoots
the knee on this machine class, formula revision still deferred to a
different machine class. Noted for the record: t=1 medians differ
between sessions (23.79 vs 16.94, with one 24.40 rep inside this
session) — single-thread decode on these boxes is bimodal across runs
(frequency/turbo behavior suspected); cap=1 tracks t1 within each
session, so the cap plumbing is not implicated.

```
# gate-decode-cap (M4b.5 default-vs-best sweep) — 2026-07-11T20:38:31Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-11
sweep: caps={1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 } + default + t1 | reps=3 (interleaved rounds) | max-tokens=128

| cap | decode tok/s (median of 3) | per-rep |
|---|---|---|
| 1 | 17.06 | 17.01 23.73 17.06 |
| 2 | 30.65 | 30.42 40.98 30.65 |
| 3 | 42.11 | 42.11 42.13 41.75 |
| 4 | 51.88 | 51.88 50.76 52.10 |
| 5 | 56.88 | 56.88 56.24 56.98 |
| 6 | 59.72 | 60.13 58.04 59.72 |
| 7 | 61.03 | 61.03 59.27 61.08 |
| 8 | 62.08 | 62.20 60.28 62.08 |
| 9 | 61.07 | 63.44 60.40 61.07 |
| 10 | 63.4 | 63.58 61.33 63.40 |
| 11 | 63.86 | 63.86 61.84 64.09 |
| 12 | 63.88 | 64.16 61.21 63.88 |
| 13 | 63.7 | 63.96 61.37 63.70 |
| 14 | 63.61 | 61.13 64.11 63.61 |
| 15 | 62.78 | 61.35 63.99 62.78 |
| 16 | 63.47 | 61.66 63.75 63.47 |
| default | 56.73 | 55.81 57.31 56.73 |
| t1 | 16.94 | 24.40 16.86 16.94 |

knee (best fixed cap): 12 (63.88 tok/s median)
default clamp(active/3,2,active): 56.73 tok/s median -> -11.19% vs best fixed (median of per-rep ratios)
gate inputs (human verdict to M4b.5 Amendments): default meets-or-beats
best-fixed? high-thread regression gone (compare cap=16 row vs knee)?
t=1 decode unchanged (t1 row vs prior recorded t=1)?
```

### 2026-07-12 — third quiet-hw session: Leg 2 miss reproduced again (knee 13, default −11.76%)

Third session, same box type (PREFLIGHT FIT), inferno @ 823437f (with
M4b.8 — which does not touch decode dispatch). Knee = cap 13
(63.49 tok/s; sessions 1–2 said 13/12 — the 8–16 plateau stays flat),
default = 55.83 → **−11.76% vs best fixed. Third consecutive Leg 2
miss; verdict unchanged** — `active/3` undershoots the knee on this
machine class, formula revision still deferred to a different machine
class. The other two human gate inputs: high-thread regression gone —
cap=16 (62.66) sits −1.3% off the knee, a flat plateau, no cliff; t=1
unchanged — t1 median 18.28 with a 22.86 rep, inside the recorded
bimodal band (16.94/23.79 across sessions), cap=1 tracking t1
within-session as before.

```
# gate-decode-cap (M4b.5 default-vs-best sweep) — 2026-07-12T08:23:33Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-12
sweep: caps={1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 } + default + t1 | reps=3 (interleaved rounds) | max-tokens=128

| cap | decode tok/s (median of 3) | per-rep |
|---|---|---|
| 1 | 18.39 | 25.47 18.26 18.39 |
| 2 | 32.25 | 43.76 32.25 32.03 |
| 3 | 44.51 | 44.88 44.06 44.51 |
| 4 | 52.58 | 52.58 49.70 53.17 |
| 5 | 56.26 | 56.26 56.73 55.94 |
| 6 | 58.21 | 58.21 58.15 58.98 |
| 7 | 60.13 | 58.80 60.13 60.15 |
| 8 | 61.28 | 60.07 61.39 61.28 |
| 9 | 61.78 | 60.36 61.85 61.78 |
| 10 | 62.39 | 60.76 62.39 62.43 |
| 11 | 63.3 | 61.25 63.64 63.30 |
| 12 | 63.29 | 62.69 63.29 63.49 |
| 13 | 63.49 | 63.49 63.27 63.61 |
| 14 | 62.71 | 62.87 62.31 62.71 |
| 15 | 62.59 | 63.34 62.59 60.26 |
| 16 | 62.66 | 62.66 63.98 61.26 |
| default | 55.83 | 55.70 55.83 56.21 |
| t1 | 18.28 | 18.22 18.28 22.86 |

knee (best fixed cap): 13 (63.49 tok/s median)
default clamp(active/3,2,active): 55.83 tok/s median -> -11.76% vs best fixed (median of per-rep ratios)
gate inputs (human verdict to M4b.5 Amendments): default meets-or-beats
best-fixed? high-thread regression gone (compare cap=16 row vs knee)?
t=1 decode unchanged (t1 row vs prior recorded t=1)?
```
