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

### 2026-07-17 — attribution session A (d2.c1.medium / Xeon Gold 6336Y, 16c, PHX)

Quiet-hw pass on merged main `783b453` (box cloned pinned SHA from origin;
local `git_dirty=true` flag was an untracked `models/` dir only — nothing
uncommitted reached the box). Preflight FIT (probe 5 invariant-TSC ok);
smoke pass first, then this real pass; workload exit 0.
Raw: `target/metal/d2.c1.medium-20260717T002043Z/target/quiet-hw/20260717T005054Z`.

**Admissibility (both must pass before any gate is evaluated):**
- Sum identity (instrument total vs op-profiler decode attention, best-t profile):
  **99.4%** — within 90–110%. ADMISSIBLE.
- Instrument perturbation (ship vs pool-profile-recording, 5 interleaved rep
  pairs): ship mean tg 58.069, prof-recording mean tg 57.792, ratio
  **0.9952 (−0.48%)** — within 1%. ADMISSIBLE.

#### gate-attn-split.out (verbatim)

```
# gate-attn-split (M4b.12 attribution: dispatch-split profile + shard sweep) — 2026-07-17T01:40:43Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

--- dispatch-split profile at --threads 16 ---
profile [prefill] 4.358s wall, 13506695392 cyc total
  op                                   cycles   share        GB/s
  attention                        3788042224   28.0%           -
  matmul:lm_head.weight            2073916502   15.4%       469.4
  matmul:layers.*.ffn.up_proj.weight     1880760872   13.9%       397.7
  matmul:layers.*.ffn.gate_proj.weight     1780266194   13.2%       420.1
  matmul:layers.*.ffn.down_proj.weight     1778487286   13.2%       420.6
  matmul:layers.*.attn.q_proj.weight      489097952    3.6%       281.7
  swiglu                            460692932    3.4%           -
  matmul:layers.*.attn.o_proj.weight      408837680    3.0%       337.0
  quantize                          205722668    1.5%           -
  rmsnorm                           142690958    1.1%           -
  rope                              117390926    0.9%           -
  matmul:layers.*.attn.k_proj.weight       99185156    0.7%       198.5
  matmul:layers.*.attn.v_proj.weight       96974260    0.7%       203.0
  add                                84325356    0.6%           -
  bias                               67781994    0.5%           -
  kv_append                          22183716    0.2%           -
  embed                              10338716    0.1%           -
profile [decode] 1.407s wall, 4266740470 cyc total
  op                                   cycles   share        GB/s
  attention                        1247288240   29.2%           -
  matmul:lm_head.weight             718769312   16.8%        41.3
  matmul:layers.*.ffn.gate_proj.weight      570713728   13.4%        40.0
  matmul:layers.*.ffn.up_proj.weight      569991536   13.4%        40.1
  matmul:layers.*.ffn.down_proj.weight      565197776   13.2%        40.4
  matmul:layers.*.attn.q_proj.weight      143484938    3.4%        29.3
  swiglu                            140701940    3.3%           -
  matmul:layers.*.attn.o_proj.weight      130104940    3.0%        32.3
  rmsnorm                            41696146    1.0%           -
  rope                               40224978    0.9%           -
  matmul:layers.*.attn.k_proj.weight       32777402    0.8%        18.3
  matmul:layers.*.attn.v_proj.weight       32136536    0.8%        18.7
  add                                18957638    0.4%           -
  bias                               13350140    0.3%           -
  embed                                974840    0.0%           -
  quantize                             370380    0.0%           -
  kv_append                                 0    0.0%           -
pool [decode attention] 1536 calls, 1240035282 cyc instrumented
  sum identity vs op-profiler attention: 99.4% (admissible: 90-110%)
  bucket                   cycles   share
  publish                 7633510    0.6%
  kernel(shard0)        898523068   72.5%
  drain                 333878704   26.9%
  per-call max-lane sums: wake 3058240 | wake-parked 0 (0 calls) | kernel-max 1229484238 | alloc-max 1978810
  lane             wake         kernel          alloc  parked-calls
  0                   0      898523068        1386456             0
  1             1991482     1099127620        1156932             0
  2             2801766     1169539780        1372874             0
  3             2790020      924404668        1372198             0
  4             2797248      992259776        1415718             0
  5             2824754     1006209340        1455978             0
  6             1971924     1176420180        1229062             0
  7             2796936     1183796480        1468944             0
  8             1980136     1193920426        1244452             0
  9             1987446      900351496        1283384             0
  10            2000444     1087488108        1362924             0
  11            2792568     1169902266        1422462             0
  12            2795238      935635538        1424826             0
  13            1996432      999067796        1271894             0
  14                  0              0              0             0
  15                  0              0              0             0
  per-call cycles histogram: 2^19:1510 2^20:26

--- shard sweep (INFERNO_ATTN_SHARDS; pool sections only) ---
--- INFERNO_ATTN_SHARDS=1 ---
pool [decode attention] 1536 calls, 4213100122 cyc instrumented
  sum identity vs op-profiler attention: 99.9% (admissible: 90-110%)
  bucket                   cycles   share
  publish                       0    0.0%
  kernel(shard0)       4213100122  100.0%
  drain                         0    0.0%
  per-call max-lane sums: wake 0 | wake-parked 0 (0 calls) | kernel-max 4213100122 | alloc-max 1507240
  lane             wake         kernel          alloc  parked-calls
  0                   0     4213100122        1507240             0
  1                   0              0              0             0
  2                   0              0              0             0
  3                   0              0              0             0
  4                   0              0              0             0
  5                   0              0              0             0
  6                   0              0              0             0
  7                   0              0              0             0
  8                   0              0              0             0
  9                   0              0              0             0
  10                  0              0              0             0
  11                  0              0              0             0
  12                  0              0              0             0
  13                  0              0              0             0
  14                  0              0              0             0
  15                  0              0              0             0
  per-call cycles histogram: 2^21:1536

--- INFERNO_ATTN_SHARDS=2 ---
pool [decode attention] 1536 calls, 2186943456 cyc instrumented
  sum identity vs op-profiler attention: 99.8% (admissible: 90-110%)
  bucket                   cycles   share
  publish                 6009120    0.3%
  kernel(shard0)       2179745588   99.7%
  drain                   1188748    0.1%
  per-call max-lane sums: wake 2264244 | wake-parked 0 (0 calls) | kernel-max 2180818404 | alloc-max 1794478
  lane             wake         kernel          alloc  parked-calls
  0                   0     2179745588        1530342             0
  1             2264244     2144741778        1321562             0
  2                   0              0              0             0
  3                   0              0              0             0
  4                   0              0              0             0
  5                   0              0              0             0
  6                   0              0              0             0
  7                   0              0              0             0
  8                   0              0              0             0
  9                   0              0              0             0
  10                  0              0              0             0
  11                  0              0              0             0
  12                  0              0              0             0
  13                  0              0              0             0
  14                  0              0              0             0
  15                  0              0              0             0
  per-call cycles histogram: 2^20:1536

--- INFERNO_ATTN_SHARDS=4 ---
pool [decode attention] 1536 calls, 1759328234 cyc instrumented
  sum identity vs op-profiler attention: 99.7% (admissible: 90-110%)
  bucket                   cycles   share
  publish                 7163542    0.4%
  kernel(shard0)       1695966036   96.4%
  drain                  56198656    3.2%
  per-call max-lane sums: wake 3101770 | wake-parked 0 (0 calls) | kernel-max 1748615292 | alloc-max 2049026
  lane             wake         kernel          alloc  parked-calls
  0                   0     1695966036        1392876             0
  1             2701618     1708092860        1168380             0
  2             2701290     1477709054        1801200             0
  3             2704638     1391416002        1287770             0
  4                   0              0              0             0
  5                   0              0              0             0
  6                   0              0              0             0
  7                   0              0              0             0
  8                   0              0              0             0
  9                   0              0              0             0
  10                  0              0              0             0
  11                  0              0              0             0
  12                  0              0              0             0
  13                  0              0              0             0
  14                  0              0              0             0
  15                  0              0              0             0
  per-call cycles histogram: 2^20:1536

--- INFERNO_ATTN_SHARDS=7 ---
pool [decode attention] 1536 calls, 1191304162 cyc instrumented
  sum identity vs op-profiler attention: 99.5% (admissible: 90-110%)
  bucket                   cycles   share
  publish                 6359568    0.5%
  kernel(shard0)       1128513530   94.7%
  drain                  56431064    4.7%
  per-call max-lane sums: wake 2352362 | wake-parked 0 (0 calls) | kernel-max 1180993058 | alloc-max 2171870
  lane             wake         kernel          alloc  parked-calls
  0                   0     1128513530        1590118             0
  1             1472260     1126388652        1267786             0
  2             2108558     1163220440        1510578             0
  3             2083864     1153988492        1427080             0
  4             1465616     1133534992        1329662             0
  5             2117416     1159651002        1464474             0
  6             2110242     1162149524        1522816             0
  7                   0              0              0             0
  8                   0              0              0             0
  9                   0              0              0             0
  10                  0              0              0             0
  11                  0              0              0             0
  12                  0              0              0             0
  13                  0              0              0             0
  14                  0              0              0             0
  15                  0              0              0             0
  per-call cycles histogram: 2^19:1496 2^20:38 2^21:2

--- INFERNO_ATTN_SHARDS=16 ---
pool [decode attention] 1536 calls, 1199613388 cyc instrumented
  sum identity vs op-profiler attention: 99.4% (admissible: 90-110%)
  bucket                   cycles   share
  publish                 7749384    0.6%
  kernel(shard0)        853426504   71.1%
  drain                 338437500   28.2%
  per-call max-lane sums: wake 2667858 | wake-parked 0 (0 calls) | kernel-max 1188607998 | alloc-max 2195624
  lane             wake         kernel          alloc  parked-calls
  0                   0      853426504        1548128             0
  1             2017170     1049363488        1367176             0
  2             2394756     1146868352        1497590             0
  3             2389886      914441288        1396286             0
  4             2018142     1073459540        1372106             0
  5             2383728     1088015656        1414480             0
  6             2039202     1115553004        1361192             0
  7             2049808      988931094        1430956             0
  8             2391864     1091737812        1256534             0
  9             2403896      915503388        1464850             0
  10            2386036      998614032        1313542             0
  11            2030238     1111379198        1307046             0
  12            2380964     1095896138        1375760             0
  13            2037646     1085702032        1371876             0
  14                  0              0              0             0
  15                  0              0              0             0
  per-call cycles histogram: 2^19:1491 2^20:45

```

#### gate-attn-perturb.out (verbatim)

```
# gate-attn-perturb (M4b.12 admissibility: ship vs pool-profile-recording A/B) — 2026-07-17T01:41:39Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

--- rep 1: ship ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "783b453",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 848.9445236060341,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 58.7405030310804,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 876.789248,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 62.528417,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 117.724479,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 16.493869,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 63.329763687296115,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 15.77228603136436,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 1: prof (recording on) ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "783b453",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 850.5025603017444,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 59.26852309405365,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 870.365513,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 62.278474,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 117.613917,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 16.410081,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 63.51264616541903,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 15.717115286372527,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 2: ship ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "783b453",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 847.1622771491026,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 56.93221860879559,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 876.8704,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 62.131272,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 118.430578,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 23.058096,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 63.02214820662615,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 22.734607314453484,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 2: prof (recording on) ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "783b453",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 841.4020489405282,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 56.64053334113887,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 876.905796,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 61.735528,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 118.377025,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 22.979728,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 63.54130628667714,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 22.629977413631263,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 3: ship ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "783b453",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 848.116284534678,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 58.54325470981239,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 865.233222,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 62.57931,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 117.753231,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 16.312925,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 63.2754700906178,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 15.833550843748641,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 3: prof (recording on) ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "783b453",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 850.1109085906357,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 58.78137274029949,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 871.473083,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 62.73818,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 117.830189,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 16.511388,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 63.465126008951586,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 16.19253772197079,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 4: ship ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "783b453",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 845.1197357553558,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 56.96758563336986,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 874.763688,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 62.995562,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 117.525208,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 22.998305,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 63.08472836592437,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 22.422880342978583,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 4: prof (recording on) ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "783b453",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 852.0059988677373,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 57.334122880942296,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 878.947879,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 62.293651,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 118.367975,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 23.049639,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 63.63169765797153,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 22.473039094559333,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 5: ship ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "783b453",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 850.4019482110696,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 59.163599224325644,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 869.889993,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 63.074936,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 117.538663,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 16.395299,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 63.26066417251713,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 15.782797427860979,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 5: prof (recording on) ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "783b453",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 507.725565089877,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 56.934303586266076,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 873.526475,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 62.434915,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 118.341633,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 22.964467,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 63.62266912343549,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 23.032094034725286,
  "inferno_t1_tg_stddev": 0.0
}

inferno tg per interleaved rep (ship | prof-recording):
58.7405030310804	59.26852309405365
56.93221860879559	56.64053334113887
58.54325470981239	58.78137274029949
56.96758563336986	57.334122880942296
59.163599224325644	56.934303586266076
```

#### gate-attn-perf.out (verbatim)

```
SKIPPED: perf not on PATH
```

### 2026-07-17 — attribution session B (s2.c2.medium / Xeon E-2388G, 8c, CHI)

Quiet-hw pass on `c2b8e04` (docs-only over merged main `783b453` — box code
identical to main; local `git_dirty=true` flag = untracked `models/` dir only).
PHX had no stock (API 406, nothing billed); ran in CHI. One CHI provision
before this one wedged in provisioning (status polls 401'd, box never ready,
deleted via gc — no workload ran). This session: preflight FIT (probe 5
invariant-TSC ok); smoke pass first, then this real pass; workload exit 0.
Raw: `target/metal/s2.c2.medium-20260717T023548Z/target/quiet-hw/20260717T030057Z`.

**Admissibility (both must pass before any gate is evaluated):**
- Sum identity (instrument total vs op-profiler decode attention, best-t profile):
  **99.6%** — within 90–110%. ADMISSIBLE.
- Instrument perturbation (ship vs pool-profile-recording, 5 interleaved rep
  pairs): ship mean tg 62.666, prof-recording mean tg 62.776, ratio
  **1.0018 (+0.18%)** — within 1%. ADMISSIBLE.

#### gate-attn-split.out (verbatim)

```
# gate-attn-split (M4b.12 attribution: dispatch-split profile + shard sweep) — 2026-07-17T03:40:33Z
machine: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (GenuineIntel) | 16 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

--- dispatch-split profile at --threads 8 ---
profile [prefill] 5.337s wall, 17033320175 cyc total
  op                                   cycles   share        GB/s
  attention                        6879355550   40.4%           -
  matmul:lm_head.weight            2482787750   14.6%       400.9
  matmul:layers.*.ffn.down_proj.weight     2106137511   12.4%       363.1
  matmul:layers.*.ffn.gate_proj.weight     1969097418   11.6%       388.3
  matmul:layers.*.ffn.up_proj.weight     1964999549   11.5%       389.1
  swiglu                            415661062    2.4%           -
  matmul:layers.*.attn.o_proj.weight      378654952    2.2%       372.0
  matmul:layers.*.attn.q_proj.weight      366687408    2.2%       384.1
  rmsnorm                           105281314    0.6%           -
  quantize                           88653302    0.5%           -
  rope                               72698084    0.4%           -
  matmul:layers.*.attn.k_proj.weight       63346644    0.4%       317.7
  matmul:layers.*.attn.v_proj.weight       62830868    0.4%       320.3
  add                                39904456    0.2%           -
  bias                               20098619    0.1%           -
  kv_append                          13666419    0.1%           -
  embed                               3459269    0.0%           -
profile [decode] 1.251s wall, 3943440779 cyc total
  op                                   cycles   share        GB/s
  attention                         914344418   23.2%           -
  matmul:lm_head.weight             776963816   19.7%        39.8
  matmul:layers.*.ffn.down_proj.weight      601274608   15.2%        39.5
  matmul:layers.*.ffn.gate_proj.weight      598533990   15.2%        39.7
  matmul:layers.*.ffn.up_proj.weight      596199492   15.1%        39.8
  swiglu                            115971725    2.9%           -
  matmul:layers.*.attn.q_proj.weight      114739482    2.9%        38.1
  matmul:layers.*.attn.o_proj.weight      112805546    2.9%        38.8
  rope                               30855192    0.8%           -
  rmsnorm                            27303755    0.7%           -
  matmul:layers.*.attn.k_proj.weight       20635985    0.5%        30.3
  matmul:layers.*.attn.v_proj.weight       20227737    0.5%        30.9
  add                                 7147134    0.2%           -
  bias                                5474856    0.1%           -
  embed                                527200    0.0%           -
  quantize                             435843    0.0%           -
  kv_append                                 0    0.0%           -
pool [decode attention] 1536 calls, 910913363 cyc instrumented
  sum identity vs op-profiler attention: 99.6% (admissible: 90-110%)
  bucket                   cycles   share
  publish                 1775264    0.2%
  kernel(shard0)        858271568   94.2%
  drain                  50866531    5.6%
  per-call max-lane sums: wake 1297782 | wake-parked 0 (0 calls) | kernel-max 907736149 | alloc-max 2241174
  lane             wake         kernel          alloc  parked-calls
  0                   0      858271568        1528468             0
  1             1052357      868214714        1252853             0
  2              932005      903368536        1251989             0
  3              936409      840895266        1392273             0
  4              949576      907503842        1523076             0
  5              918535      890582718        1317177             0
  6              913868      645606826        1435387             0
  7              918774      694121406        1548531             0
  per-call cycles histogram: 2^19:1536

--- shard sweep (INFERNO_ATTN_SHARDS; pool sections only) ---
--- INFERNO_ATTN_SHARDS=1 ---
pool [decode attention] 1536 calls, 2762261255 cyc instrumented
  sum identity vs op-profiler attention: 99.9% (admissible: 90-110%)
  bucket                   cycles   share
  publish                       0    0.0%
  kernel(shard0)       2762261255  100.0%
  drain                         0    0.0%
  per-call max-lane sums: wake 0 | wake-parked 0 (0 calls) | kernel-max 2762261255 | alloc-max 1551693
  lane             wake         kernel          alloc  parked-calls
  0                   0     2762261255        1551693             0
  1                   0              0              0             0
  2                   0              0              0             0
  3                   0              0              0             0
  4                   0              0              0             0
  5                   0              0              0             0
  6                   0              0              0             0
  7                   0              0              0             0
  per-call cycles histogram: 2^20:1533 2^21:3

--- INFERNO_ATTN_SHARDS=2 ---
pool [decode attention] 1536 calls, 1675973498 cyc instrumented
  sum identity vs op-profiler attention: 99.8% (admissible: 90-110%)
  bucket                   cycles   share
  publish                 1623540    0.1%
  kernel(shard0)       1673846749   99.9%
  drain                    503209    0.0%
  per-call max-lane sums: wake 846252 | wake-parked 0 (0 calls) | kernel-max 1674241887 | alloc-max 1564562
  lane             wake         kernel          alloc  parked-calls
  0                   0     1673846749        1478975             0
  1              846252     1649745032        1097902             0
  2                   0              0              0             0
  3                   0              0              0             0
  4                   0              0              0             0
  5                   0              0              0             0
  6                   0              0              0             0
  7                   0              0              0             0
  per-call cycles histogram: 2^20:1536

--- INFERNO_ATTN_SHARDS=4 ---
pool [decode attention] 1536 calls, 1111459505 cyc instrumented
  sum identity vs op-profiler attention: 99.7% (admissible: 90-110%)
  bucket                   cycles   share
  publish                 1636026    0.1%
  kernel(shard0)       1021127695   91.9%
  drain                  88695784    8.0%
  per-call max-lane sums: wake 937506 | wake-parked 0 (0 calls) | kernel-max 1108471799 | alloc-max 1699980
  lane             wake         kernel          alloc  parked-calls
  0                   0     1021127695        1432809             0
  1              824517     1108471799        1238926             0
  2              832357      827753069        1217497             0
  3              829609      933334629        1065978             0
  4                   0              0              0             0
  5                   0              0              0             0
  6                   0              0              0             0
  7                   0              0              0             0
  per-call cycles histogram: 2^19:1536

--- INFERNO_ATTN_SHARDS=7 ---
pool [decode attention] 1536 calls, 915024665 cyc instrumented
  sum identity vs op-profiler attention: 99.6% (admissible: 90-110%)
  bucket                   cycles   share
  publish                 1691469    0.2%
  kernel(shard0)        874943320   95.6%
  drain                  38389876    4.2%
  per-call max-lane sums: wake 1048308 | wake-parked 0 (0 calls) | kernel-max 912017063 | alloc-max 2042036
  lane             wake         kernel          alloc  parked-calls
  0                   0      874943320        1470033             0
  1              812664      896850566        1466418             0
  2              792697      907257915        1189555             0
  3              810604      876259144        1400547             0
  4              816351      852480105        1136996             0
  5              813696      899452128        1241472             0
  6              802651      911705191        1174435             0
  7                   0              0              0             0
  per-call cycles histogram: 2^19:1536

--- INFERNO_ATTN_SHARDS=8 ---
pool [decode attention] 1536 calls, 877432519 cyc instrumented
  sum identity vs op-profiler attention: 99.6% (admissible: 90-110%)
  bucket                   cycles   share
  publish                 1738397    0.2%
  kernel(shard0)        872655221   99.5%
  drain                   3038901    0.3%
  per-call max-lane sums: wake 1124267 | wake-parked 0 (0 calls) | kernel-max 874581649 | alloc-max 2138998
  lane             wake         kernel          alloc  parked-calls
  0                   0      872655221        1493778             0
  1              895777      858633727        1280133             0
  2              905955      873927903        1189055             0
  3              901658      841966747        1313266             0
  4              907501      862443287        1508787             0
  5              915003      839445681        1249532             0
  6              898348      616421183        1500277             0
  7              926953      630287654        1163710             0
  per-call cycles histogram: 2^19:1536

```

#### gate-attn-perturb.out (verbatim)

```
# gate-attn-perturb (M4b.12 admissibility: ship vs pool-profile-recording A/B) — 2026-07-17T03:41:31Z
machine: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (GenuineIntel) | 16 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

--- rep 1: ship ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "c2b8e04",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 491.92069545857106,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 62.210666271268636,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 1043.215544,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 72.675002,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 165.235707,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 33.757523,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 89.84133676704957,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 34.70110734581655,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 1: prof (recording on) ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "c2b8e04",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 627.0242988093505,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 62.59274417026805,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 1046.787346,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 72.57961,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 163.965256,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 33.655194,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 90.00416847977895,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 34.7711388780212,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 2: ship ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "c2b8e04",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 615.7457791471686,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 62.841750067839875,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 1045.893829,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 72.808958,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 163.909804,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 33.33743,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 88.33891073756496,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 34.76466290270114,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 2: prof (recording on) ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "c2b8e04",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 631.5101691077296,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 62.86431355364875,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 1049.979054,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 72.766355,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 163.90847,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 33.396586,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 91.2325199603207,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 34.682910285586175,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 3: ship ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "c2b8e04",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 626.1584780276019,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 62.80782193265688,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 1050.677433,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 72.639649,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 165.354873,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 33.270785,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 90.98818951743682,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 35.12811778923342,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 3: prof (recording on) ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "c2b8e04",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 631.6038741658907,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 62.989336071242256,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 1045.500522,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 72.715442,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 164.443919,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 33.622509,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 91.27448464702093,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 35.26328905691748,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 4: ship ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "c2b8e04",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 619.2903193960914,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 62.74003805514654,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 1043.725536,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 72.755603,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 164.706255,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 33.256219,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 91.04963863618349,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 35.120694983304475,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 4: prof (recording on) ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "c2b8e04",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 628.3776981140769,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 62.741371939856236,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 1045.426659,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 72.78744,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 164.931722,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 33.752706,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 90.88017084426997,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 35.305583677272764,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 5: ship ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "c2b8e04",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 625.4949297350454,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 62.731620396579835,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 1042.939366,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 72.491567,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 165.302039,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 33.667691,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 91.1483885415129,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 35.38751450248769,
  "inferno_t1_tg_stddev": 0.0
}
--- rep 5: prof (recording on) ---
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "c2b8e04",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 1,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 632.0703019798334,
  "inferno_pp_stddev": 0.0,
  "inferno_tg_tok_s": 62.69423893596733,
  "inferno_tg_stddev": 0.0,
  "llama_pp_tok_s": 1046.427447,
  "llama_pp_stddev": 0.0,
  "llama_tg_tok_s": 72.739329,
  "llama_tg_stddev": 0.0,
  "llama_t1_pp_tok_s": 164.790834,
  "llama_t1_pp_stddev": 0.0,
  "llama_t1_tg_tok_s": 33.332104,
  "llama_t1_tg_stddev": 0.0,
  "inferno_t1_pp_tok_s": 91.05812988280881,
  "inferno_t1_pp_stddev": 0.0,
  "inferno_t1_tg_tok_s": 35.1651278952034,
  "inferno_t1_tg_stddev": 0.0
}

inferno tg per interleaved rep (ship | prof-recording):
62.210666271268636	62.59274417026805
62.841750067839875	62.86431355364875
62.80782193265688	62.989336071242256
62.74003805514654	62.741371939856236
62.731620396579835	62.69423893596733
```

#### gate-attn-perf.out (verbatim)

```
SKIPPED: perf not on PATH
```

### 2026-07-17 — gate verdicts (pre-registered arithmetic, both machines)

Computed by the controller from the session A/B amendments above, exactly per
the pre-registered formulas (§Pre-registered gates). Formulas were fixed before
the data; lever tasks were not consulted.

**Step 1 — admissibility (gate precondition):** 16c sum identity 99.4%,
perturbation 0.9952 (−0.48%); 8c sum identity 99.6%, perturbation 1.0018
(+0.18%). All four within bounds → data admissible on both machines.

**Step 2 — menu guard.** `C(n) = kernel_max_cyc / calls` (1536 calls each):

| n | 16c C(n) cyc/call | 8c C(n) cyc/call |
|---|---|---|
| 1 | 2,742,904 | 1,798,347 |
| 2 | 1,419,804 | 1,090,001 |
| 4 | 1,138,421 | 721,661 |
| 7 | 768,876 | 593,761 |
| max (16 / 8) | 773,833 | 569,389 |

Guard: C(max) > C(1)/2? 16c: 773,833 vs 1,371,452 — **no** (ratio 0.282).
8c: 569,389 vs 899,174 — **no** (ratio 0.317). The menu guard does not fire
on either machine; per-lever gates are evaluated.

**Step 3 — per-lever decode-wall shares** (best-t profile; decode_total_cyc =
op table total: 16c 4,266,740,470 @ t=16; 8c 3,943,440,779 @ t=8):

| share | 16c | 8c | threshold | verdict |
|---|---|---|---|---|
| P_W = wake_parked_cyc / decode_total | 0 / 4,266,740,470 = **0.000%** | 0 / 3,943,440,779 = **0.000%** | <3% on both | **STOP — Lever W not authorized** |
| P_A = alloc_max_cyc / decode_total | 1,978,810 / 4,266,740,470 = **0.046%** | 2,241,174 / 3,943,440,779 = **0.057%** | <3% on both | **STOP — Lever A not authorized** |
| P_D = publish_cyc / decode_total | 7,633,510 / 4,266,740,470 = **0.179%** | 1,775,264 / 3,943,440,779 = **0.045%** | <3% on both | **STOP — Lever D not authorized** |

(P_W's numerator is literally zero on both machines: `wake-parked 0 (0 calls)`
in both best-t profiles — no lane was park-eligible during decode attention at
best-t. Shares are each computed against the same decode wall independently;
per the design they are not summed — the P_W and P_D brackets overlap in wall
time.)

**Verdict: all three gates STOP. Tasks 8–10 (Levers A, D, W) do not run.**
The dispatch-split blame table locates the headroom instead in (a) the
attention kernel itself — kernel(shard0) is 72.5% (16c) / 94.2% (8c) of the
instrumented call — and (b) on the 16c box, drain at 26.9%: the dispatcher
waiting on the slowest worker lane, with per-lane kernel sums spanning
0.90–1.19 Gcyc (load imbalance across head shards), while publish/wake/alloc
are all sub-0.2%. The C(n) sweep confirms the kernel scales to the box's lane
count (0.28×/0.32× C(1) at max shards, flattening past n=7). This is the
milestone's memory-stall/kernel-bound finding, recorded per §Attribution-first:
an all-STOP with the finding is a successful outcome. Milestone proceeds
directly to Task 11 (closing).
