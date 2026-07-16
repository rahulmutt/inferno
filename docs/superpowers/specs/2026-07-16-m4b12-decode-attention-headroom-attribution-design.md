# M4b.12 — Decode Attention Headroom Attribution (Instrumented Dispatch Split, Gated Levers) Design

**Date:** 2026-07-16
**Status:** Approved design, pre-implementation
**Milestone:** M4b.12 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.11](2026-07-16-m4b11-decode-attention-f16kv-design.md))

This milestone takes up the finding M4b.11's closing verdict recorded for
"any successor decode-attention work": head-sharding realized an 11.3%/11.8%
decode-wall reduction against Gate 1's 41–52% ceiling, and the gap between
the two is unexplained. M4b.12 builds the instrument that can explain it,
measures, and gates levers on what the measurement blames.

## Motivation

M4b.11's attribution measured attention's share of decode wall correctly
(S = 55.8% at t=16 on the 16c 6336Y, 46.6% at t=8 on the 8c E-2388G) and
Gate 1's ceiling arithmetic was sound *given its assumption* — that the
per-head work scales ~`min(t_best, 14)`-way when sharded. It does not: the
landed lever recovered roughly a quarter of the ceiling, tg vs llama.cpp
best-of now stands at 0.96x (16c) / 0.86x (8c), and the v1 win criterion
(tg > 1x) is still open.

The op profiler cannot say where the other three quarters went, because it
measures the whole host call via `rdtsc` on the calling thread. Everything
inside `par_attention_heads` (`inferno-pool/src/pool.rs`) lands in one
"attention" number:

- **publish** — the job write, epoch bump, and unpark loop the dispatcher
  pays per call, 24 times per decoded token;
- **wake latency** — the gap between dispatch and each worker's first cycle
  inside its shard, which is µs-scale if the worker was spinning and
  scheduler-scale if it had parked (`SPIN_ITERS` ≈ 50µs of spin, then park);
- **per-lane kernel compute** — the hspan kernel itself, which today also
  includes a `vec![0f32; pos + 1]` heap allocation per lane per call
  (`run_attn_heads_span`, pool.rs) — up to 14 allocations × 24 layers per
  decoded token;
- **drain wait** — the dispatcher spinning on `remaining` for the slowest
  lane after finishing its own shard.

First-principles arithmetic says the head-loop math is a minority of the
recorded attention cycles (14 heads × ~640 visible positions × 64-dim dot +
exp per call, against ~1.3M cycles/call observed serial at t=16), and
M4b.11's Gate 2 already proved the KV *bytes* are negligible (F16 KV
projected ≤ 0.25% of decode wall). The missing cycles are therefore in the
dispatch machinery, the allocation, memory stalls, or lane imbalance — four
hypotheses the current profiler cannot distinguish. Guessing among them is
the sand M4b.10 refused to build on; this milestone measures instead.

## Scope Decisions (M4b.12)

| Decision | Choice |
|---|---|
| Phase | **Decode only.** Prefill is closed (M4b.1 gate MET at 10.63x @ t=12, M4b.9) |
| Structure | **Attribution-first, gated** (M4b.11 pattern): Task 1 builds the dispatch-split instrument, Task 2 measures on quiet hw, each lever ships only if its pre-registered gate fires |
| Lever depth | **Pool-side + one-call codegen swaps only.** Wait-strategy, per-lane scratch, dispatch-protocol tightening. No decode-step restructuring, no op-sequence change. The menu below happens to be pool-side only — no `HOST_ABI_VERSION` bump, no recompile; a cached `model.so` benefits immediately (M4b.5/M4b.10 precedent) |
| Machines | 16c `d2.c1.medium` (Xeon Gold 6336Y, primary — deepest recorded history) + 8c `s2.c2.medium` (Xeon E-2388G, check) |
| Riders | The same sessions capture an attention **shard-count sweep** (the scaling-shape curve Gate 1 assumed and never measured) and a **`perf stat` topdown capture** (the only view of worker-side costs calling-thread self-measurement cannot see) |
| Fork arity | **Independent gates** per lever (M4b.11 precedent): the levers compose and each has its own governing bucket |
| Exit target | **Per-lever gates own every ship decision** (M4b.11's headroom-set closing target overpromised from a ceiling; not repeated). The closing bench records tg vs llama best-of as v1 context, **never the gate** |
| Correctness | Every lever is **bit-invisible by construction** (wait strategy, scratch reuse, and dispatch protocol never touch math); the full differential suite passes with **zero tolerance change** |
| Measurement discipline | Quiet bare metal via `mise run metal`; interleaved reps; ratios computed within-session (standing M4b discipline) |

**Explicitly out of scope:**

- **Flash-decoding position-split** — standing escalation (M4b.11 §Scope),
  still contingent on a profile that blames irreducible serial attention.
  If this milestone's gates all STOP on a memory-stall finding, the
  recorded evidence *is* the escalation record — but the escalation itself
  is its own milestone.
- **F16 KV** — STOP'd on recorded arithmetic (M4b.11 gate verdict); stays
  closed unless a future measurement overturns the bandwidth premise.
- **An attention shard-count cap.** If the shard sweep shows a knee, that
  is a recorded diagnostic. Planting a tuning constant two milestones after
  M4b.10 buried one requires an M4b.10-style measured decision rule, not a
  rider here.
- **NUMA-aware threading** — out of scope since M4b.1.
- **Prefill anything** — the prefill gate is MET and closed.
- **Any CI perf gate** — standing rule (AGENTS.md).
- **Any pos-/size-threshold heuristic** in decode dispatch (M4b.11 rule).

## The dispatch-split instrument

A pool-side cycle-accounting surface, compiled only under a new
`pool-profile` cargo feature on `inferno-pool` (forwarded by the CLI), so
the shipping hot loop is textually untouched when the feature is off.
Timestamps are raw `rdtsc`; both target Xeons have invariant, synchronized
TSC (`constant_tsc nonstop_tsc`), and the quiet-hw preflight gains an
assert on those cpuinfo flags.

Per `par_attention_heads` call, four buckets:

1. **publish** — dispatcher cycles from call entry to the last `unpark`
   returning (job write + epoch bump + unpark loop).
2. **wake** — per lane: dispatch timestamp → that worker's first cycle
   inside its shard. Recorded alongside a **parked bit**: whether the lane
   had parked or was still spinning when the epoch arrived. The per-call
   effective wake latency is the max over lanes. (This single bit is what
   most likely decides Lever W.)
3. **kernel** — per lane: cycles inside `run_attn_heads_span`, with the
   scratch allocation bracketed separately so hypothesis **H-alloc** gets
   its own number rather than an inference.
4. **drain** — dispatcher cycles spinning on `remaining` after finishing
   shard 0.

Aggregation is preallocated and fixed-size (per-bucket sums plus a
log-scaled histogram of per-call totals; per-lane sums for the kernel and
wake buckets) and is dumped once at end of run: `inferno run --profile` on
a feature-enabled build gains a `pool [decode attention]` section alongside
the existing op table.

**Sum identity (admissibility check #1).** Per session:

```
publish + max_lane(wake + kernel) + drain  ≈  op-profiler attention cycles
```

within **10%** (the dispatcher's own shard-0 kernel time sits inside the
max when shard 0 is the wall). If the split does not sum to the whole, the
instrument is broken and **no gate is evaluated**.

**Perturbation A/B (admissibility check #2, pre-registered).** One
interleaved A/B per machine: shipping build vs feature build with recording
on, the standing bench protocol. If tg moves by more than **1%**, the
instrumentation is reworked before any attribution is trusted.

## Attribution protocol (Task 2)

One quiet-hw session per machine, in this order:

1. **Dispatch-split profile** at `t_best` (16 / 8): the M4b.11 protocol run
   (pp=512, tg=128) on the feature build with recording on. Yields the
   **blame table**: each bucket's share of *decode wall* (bucket cycles ÷
   total decode cycles — decode-wall shares, not shares-of-attention, so
   projections translate directly to tg), parked-vs-spinning wake counts,
   H-alloc's share, and per-lane kernel sums (the 8c box's 14-heads-over-8
   split makes lane imbalance directly visible here).
2. **Shard-count sweep** — decode attention forced to 1 / 2 / 4 / 7 / 14
   spans via a new probe-only env var **`INFERNO_ATTN_SHARDS`** (decode
   attention only — deliberately *not* `INFERNO_DECODE_THREADS`, which
   would move the GEMV cap in the same run and confound the curve). Yields
   the scaling-shape curve, measured with the same instrument on.
3. **`perf stat` capture** — topdown level-1 plus
   context-switches/cpu-migrations around the standing protocol run, on the
   **shipping** build (counters see what actually ships). We own the metal;
   no PMU-access caveats apply.

The admissibility checks run inside the same sessions. Shares guide scoping
and never gate CI (M4b.2 rule).

## The pre-registered gates

Written down before any sweep runs (standing discipline: M4b.2, M4b.8,
M4b.9, M4b.10, M4b.11). Gates are evaluated **once**, from the Task 2 blame
tables — no re-rolls.

**Projections (ceilings).** For each lever, the projected decode-wall
recovery is its governing bucket's decode-wall share:

```
P_W = wake share        (portion attributable to parked lanes)
P_A = H-alloc share
P_D = publish share
```

Unlike M4b.11's Gate 1, each ceiling is judged next to a *measured*
scaling curve (rider 2), not an assumed one.

**Thresholds (M4b.6's STOP gate, verbatim), applied per lever:**

- **≥ 3% projected on both machines → lever authorized.**
- **< 3% on both machines → STOP** — the finding is recorded and the lever
  is not built.
- **Split verdict → judgment call**, argument recorded in the amendment.

**Menu guard (evaluated first).** Let `C(n)` be the wall-clock kernel
component (max-lane kernel cycles per call) at `n` shards from the sweep.
If **`C(max shards) > C(1) / 2` on both machines** — kernel compute scales
less than 2x despite 14- or 8-way sharding — the buckets above are noise
around a memory-stall wall: **all three gates STOP**, and the finding —
with the topdown capture as evidence — is recorded as the escalation
record the flash-decoding note (M4b.11 §Scope) has been waiting for.

## The lever menu

Each lever independent, composable, bit-invisible.

### Lever W — decode wait-strategy

Fires on the wake bucket **when the parked bit correlates** (the calls that
pay wake latency are the calls whose slowest lane had parked). The change:
workers that have just run a **decode-kind** shard (`Gemv` / `AttnHeads`)
extend their spin window before parking, keeping the pool hot across the
serial gaps inside a decode step. Keyed off the job kind the worker just
ran — no new phase signal, no API change. `SPIN_ITERS` remains one named
constant; only the decode-window multiple is new, and an idle host (no
decode dispatch arriving) still parks on the same schedule as today.
Numerics-free by construction.

### Lever A — per-lane scratch reuse

Fires on the H-alloc bucket. Replace `vec![0f32; j.pos + 1]` in
`run_attn_heads_span` with a grow-only per-worker-slot scratch buffer. The
kernel writes `scores[t]` before any read of it, so stale bytes beyond a
shorter call's `pos + 1` are unreachable — sound by argument, and checked
by test (below). Same values written, same kernel, bit-invisible.

### Lever D — publish slimming

Fires on the publish bucket. Cheapen the dispatch itself: cache the shard
table per `(n_heads, active)` instead of rebuilding the `Vec` every call,
and trim redundant `SeqCst` traffic in the publish sequence —
**protocol-preserving only**; the epoch/remaining SAFETY argument in
pool.rs must survive verbatim, re-read against the final ordering.

**Landing protocol per authorized lever (M4b.11 pattern):** within-session
parent-vs-lever A/B on both machines, long (pp=512 tg=128) and short
(pp=16 tg=32) contexts, five reps interleaved; the lever lands only if tg
does not regress on either machine, else the pre-registered revert fires.
Levers land one at a time so each keeps its own data point, in
pre-registered order **A → D → W** (ascending blast radius: a scratch swap,
then a protocol-preserving cache, then a wait-strategy change); composition
is read from the final closing bench.

## Correctness & testing

- **Standing guard:** the full differential suite passes with zero
  tolerance change after every lever. The existing `par_attention_heads`
  bit-invariance and fallback tests remain in force.
- **Lever A:** a pool unit test drives a pos-varying call sequence
  (long → short → long) and asserts bit-identity between reused-scratch and
  fresh-alloc outputs.
- **Lever W:** liveness, not numerics — a stress test dispatches decode-
  and prefill-kind jobs across the window boundary under load and asserts
  completion (lost-wakeup hunt); the park/unpark SAFETY comment is
  re-stated for the window.
- **Lever D:** the cached shard table is property-tested identical to the
  freshly computed table over `(n_heads, active)` ∈ full range; the
  existing pool concurrency stress tests gate the ordering trim.
- **Instrument:** the `pool-profile` feature is compiled and unit-tested in
  CI (a smoke test asserts the sum identity on a small synthetic run) so it
  cannot rot; it is **off** in every bench/shipping build.
- **`INFERNO_ATTN_SHARDS`:** probe-only; a unit test asserts it is
  inert when unset and that any forced shard count reproduces the unsharded
  output bit-for-bit (it can only regroup heads).

## Exit criteria

All recorded in this spec's §Amendments (standing protocol; bench outputs
verbatim in the M4a spec §Amendments where the protocol lives):

1. **Blame table, shard-count sweep, and `perf stat` capture recorded on
   both machines**, with both admissibility checks passing.
2. **Gate verdicts recorded once, arithmetic shown**, including the menu
   guard.
3. **Every authorized lever landed with its within-session data point** (or
   its revert recorded); **every STOP recorded.** An all-STOP with the
   memory-stall finding is a successful outcome — it is the flash-decoding
   escalation record.
4. **Closing quiet-hw re-bench vs llama.cpp best-of** recorded as v1
   context on both machines (tg ratio, alongside the 0.96x / 0.86x
   baseline).

## Risks

- **The instrument perturbs what it measures.** rdtsc pairs and record
  writes sit in a µs-scale hot path. Mitigation: preallocated fixed-size
  aggregation, the ≤ 1% perturbation A/B, and the sum identity; gates are
  simply not evaluated on inadmissible data.
- **Cross-thread TSC comparison.** Wake latency subtracts a dispatcher
  timestamp from a worker timestamp. Both boxes have invariant synchronized
  TSC (preflight now asserts it), and the quantities of interest are µs- to
  ms-scale against ns-scale skew.
- **All gates fire small.** W + A + D may compose to well under the ~16%
  the 8c box needs to reach tg 1x. Honest outcome: tg ≥ 1x is context,
  never the gate; the residual gap plus the topdown capture is exactly the
  evidence the escalation decision needs.
- **Blame outside the menu.** The 8c box's 14-heads-over-8-lanes shard
  table (six 2-head spans, two 1-head) makes the 2-head lanes the drain
  wall — a ~7x ceiling instead of 8x, visible in the per-lane records but
  fixed by no menu lever. Pre-registered handling: recorded finding, no
  improvised lever (the no-new-tuning-constant rule above).
- **Lever W burns idle CPU inside the window.** Bounded: the window is
  keyed to decode-kind dispatches and collapses to today's behavior on an
  idle host; the constant stays named and tunable (existing SPIN_ITERS
  note).

## Amendments

*(Recorded protocol data points and scoped amendments land here; never
edit a recorded data point.)*
