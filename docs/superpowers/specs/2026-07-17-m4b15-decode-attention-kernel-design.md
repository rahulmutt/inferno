# M4b.15 — Decode Attention Per-Thread Kernel Quality (Phase-Marginal µbench, Gated KV-Split) Design

**Date:** 2026-07-17
**Status:** Approved design, pre-implementation
**Milestone:** M4b.15 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.14](2026-07-17-m4b14-prefill-attention-query-blocking-design.md))

This milestone takes up the **decode attention** finding that M4b.12's
closing verdict recorded as the starting point for any future
decode-attention work: the wall time is in the hspan kernel itself, not
in dispatch, not in bandwidth. It follows the attribution-first
discipline one level deeper than M4b.12 went — a phase-marginal µbench
that explains where the kernel's cycles go *inside* the kernel boundary,
a pair of bit-neutral per-phase levers gated on that blame, and one
order-changing lever (a KV-position-split kernel, flash-decoding shape)
pre-registered behind a mid-milestone gate.

## Motivation

At the M4b.14 closing benches (2026-07-17, both quiet-hw boxes, M4a
protocol), the v1 win criterion (pp > 1x AND tg > 1x vs llama.cpp at its
best) stands at:

| machine | pp512 vs best-of | tg128 vs best-of |
|---|---|---|
| d2.c1.medium (6336Y 16c) | 0.79x | 0.94x |
| s2.c2.medium (E-2388G 8c) | 0.70x | 0.86x |

M4b.14's closing verdict established that the remaining pp gap is not
closable from inside any single prefill bracket on the 8c box. **tg is
the closer gap** (0.94x on the 16c box), and it is the direction the
existing attribution record points at. That record, in full:

1. **Decode attention is not bandwidth-bound.** M4b.11's Gate 2
   arithmetic: at best-t the attention pass streams unique KV at
   2.41 GB/s (16c) / 3.91 GB/s (8c) — **4.4% / 8.5%** of each machine's
   measured bandwidth ceiling (54.39 / 45.95 GB/s, gate-bw-curve). F16
   KV was STOP'd on that arithmetic (P2 = 0.09% / 0.25%) and stays
   closed.
2. **Decode attention is not dispatch-bound.** M4b.12's dispatch-split
   blame tables: publish, wake-parked, and scratch-alloc are each
   **sub-0.2%** of decode wall on both machines; the menu guard did not
   fire (C(max)/C(1) = 0.28 / 0.32 — the kernel scales to the box's
   lane count).
3. **The wall is the kernel plus, on 16c, the drain.** kernel(shard0)
   is **72.5% (16c) / 94.2% (8c)** of the instrumented call; on the 16c
   box drain — the dispatcher waiting on the slowest lane — is
   **26.9%**, with per-lane kernel sums spanning 0.90–1.19 Gcyc.
4. **Attention's share of decode wall at best-t** (M4b.12 best-t
   profiles, slowest-lane kernel sum × 1536 calls / decode total):
   ~**27.9%** (16c: 773,833 × 1536 / 4,266,740,470) / ~**22.2%** (8c:
   569,389 × 1536 / 3,943,440,779).

A kernel that is neither streaming memory near its bandwidth roofline
nor waiting on dispatch is stalling on something *inside* the kernel
boundary — and nothing in the record says what. That unattributed
region is this milestone's target. Ceiling context (why tg, unlike pp,
is arithmetically in reach): removing a fraction `c` of the attention
bracket reaches `tg_ratio / (1 - attn_share·c)`; at c=½ that is 1.09x
(16c) / 0.97x (8c), and at c=1, 1.30x / 1.11x. The 16c box can close
from attention alone; the 8c box is tight — the headroom-set exit
criterion (below) is scoped for exactly that honesty.

### The kernel-level suspects (by inspection — the µbench decides)

`attn_core_avx2` (`inferno-kernels/src/attention.rs`) computes, per
head: a QK-dot pass over every visible position, a max pass, an
exp+denominator pass (`expf_avx2` 8-wide + scalar tail), and a weighted
AV-accumulation pass. Two serial-dependency structures stand out:

- **QK dot:** one YMM accumulator per position — head_dim 64 is a chain
  of 8 *dependent* FMAs (4–5 cyc latency each) followed by an `hsum8`
  reduction, per visible position. The loads could issue far ahead of
  the FMAs; the single accumulator serializes them. This is consistent
  with a kernel running at 4–9% of its bandwidth roofline.
- **AV accumulate:** the 64-float output row is loaded, FMA'd, and
  stored back to memory **once per visible position** — a
  store-to-load-forwarding chain the length of the visible range. The
  row fits in exactly 8 YMM registers.

Counter-hypotheses the µbench must be able to blame instead: the
exp/softmax pass (already 8-wide, but never measured in isolation), and
GQA K/V re-streaming (7 Q-heads per KV head re-read the same K/V; the
M4b.11 arithmetic says bandwidth is not the bound, but the µbench's
roofline anchors confirm or refute it at working-set scale). Inspection
is not attribution: **no lever ships without its phase being blamed by
measurement.**

### The 16c lane-imbalance reality (why "rebalancing" is not a lever)

`shard_table_aligned(n_heads=14, active=16, align=1)`
(`inferno-pool/src/shard.rs`) already produces **14 one-head shards** —
every lane gets exactly one head, and two lanes idle. There is nothing
to reassign at head granularity: the 26.9% drain and the 0.90–1.19 Gcyc
per-lane spread arise at *equal* shard sizes (candidate explanations:
dispatcher-lane double duty, GQA-group cache locality — 7 lanes stream
each of the 2 shared KV groups — or topology). The µbench's cold/warm
axis probes how much cache state can explain; the only *structural*
remedy is granularity finer than a head, which is precisely Lever 2.
The 16c drain fraction is therefore a **blame key for Lever 2**, not a
Lever 1 entry.

## Scope Decisions (M4b.15)

| Decision | Choice |
|---|---|
| Instrument | Phase-marginal µbench (`attn_decode_phases`, new bench in `inferno-kernels`): phase-isolating kernel variants + counterfactual probes + roofline anchors. Bench-local loop copies; zero shipping-code change |
| Levers | **Lever 1** = up to two bit-neutral per-phase kernel fixes (multi-position-blocked QK dot; register-resident AV accumulation), **independently gated** on the µbench blame — they address disjoint phases and compose. **Lever 2** = KV-position-split kernel (flash-decoding shape), authorized only by the mid-milestone gate |
| Exit criterion | **Headroom-set target (M4b.11 style):** closing tg per box judged against `baseline_tg × (1 + S × r)` with `S`, `r`, and the baseline all measured in-session before the closing runs. v1 tg ≥ 1.0x recorded as context, never the gate. Sanctioned STOP-outs below |
| Phase | **Decode attention (`m == 1` path) only.** Prefill attention (M4b.14's query-blocked kernel), every GEMM/GEMV path, and the pool dispatch machinery untouched |
| Dtype | **f32 KV.** M4b.11's F16 KV STOP stands; the recorded arithmetic (4–9% of bandwidth ceiling) already forecloses a bandwidth re-opening |
| Machines | 16c `d2.c1.medium` (6336Y) + 8c `s2.c2.medium` (E-2388G); one quiet-hw session per box doubling as mid-gate + closing (M4b.13/M4b.14 precedent); provision with `perf` on the image PATH (M4b.12 deviation, carried forward) |
| Standing invariants | Lever 1 kernels bit-equal the current ones (per-position math unchanged); scalar kernel untouched; scalar↔AVX2 per-ISA identity; `_hspan` ≡ whole-call over any head split; `m == 1` compiled-path identity; cross-thread bit-identity; `attn_rel_tol` / `logits_abs_tol` untouched by Lever 1; compiled-vs-interpreter and artifact differentials green throughout |

**Explicitly out of scope:**

- **Prefill attention.** M4b.14 closed it (all-STOP; on the 8c box no
  attention lever reaches the pp criterion). No prefill claim is made.
- **F16 KV / any KV dtype change** (M4b.11 gate verdict; stays closed
  unless a future attribution shows decode attention bandwidth-bound).
- **Decode GEMV / bandwidth levers.** The 8c ceiling arithmetic says
  attention alone may not reach tg 1.0x there; that is what the
  headroom-set exit absorbs. A GEMV milestone is a separate decision.
- **Dispatch/pool changes.** M4b.12 exonerated dispatch; publish, wake,
  and alloc are closed findings.
- **Shard reassignment.** Impossible at head granularity (14 one-head
  shards); subsumed by Lever 2.

## The instrument — `attn_decode_phases`

**Why variants, not timers.** The per-call kernel wall is too small for
intrusive sub-brackets — rdtsc pairs inside the per-position loops would
perturb the very latency chains under suspicion (M4b.12's brackets
worked because dispatch spans are µs-scale). The µbench instead times
**phase-isolating variants** and attributes by *marginals*, with a
pre-registered identity check standing in for M4b.12's sum-identity
admissibility.

**Structure.** New criterion bench `attn_decode_phases` in
`inferno-kernels/benches/`, registered with the existing
`mise run bench-kernels` task. Protocol geometry (14 Q / 2 KV heads,
head_dim 64, `kv_dim` 128), single-threaded, one full 14-head call per
iteration. Inputs deterministically varied (LCG fill — spread exp
inputs, no RNG dependency). All variants are bench-local copies of
`attn_core_avx2`'s loops; the shipping kernel is additionally timed via
its public symbol as the anchor:

1. **`full`** — the shipping kernel (public symbol).
2. **`full_local`** — bench-local copy (guards against a copy-drift
   artifact; must match `full` within noise or the run is inadmissible).
3. **`dot_only`** — QK dot + `hsum8` + max; no exp, no AV.
4. **`no_av`** — dot + max + exp + denom; no AV.
   Marginals: **dot** = `dot_only`, **softmax** = `no_av − dot_only`,
   **AV** = `full_local − no_av`.
5. **Counterfactual probes** (also the Lever 1 candidates):
   **`dot_blocked`** — the QK pass with N positions in flight on
   independent accumulators, each position's own partition order and
   `hsum8` tree unchanged (bit-identical per position);
   **`av_regacc`** — the AV pass with the out row held in 8 YMM
   registers across all positions, t-ascending accumulation order
   preserved (bit-identical); **`combined`** — both.
6. **Roofline anchors:** a pure K/V-stream loop over the same visible
   extent (the bandwidth bound at working-set size) and an FMA-peak
   microloop (the compute bound). Each phase marginal is read against
   both anchors.

**Sweep axes:** `pos ∈ {127, 511, 639, 1023, 2047}` — 511/639 bracket
the bench protocol's decode range (pp 512 + tg 128); the wider points
are for understanding, not gating. **Cold-vs-warm KV** at each pos:
one head timed with KV freshly evicted vs re-run — the probe at the 16c
per-lane-spread hypothesis (bounds how much cache state can explain; it
cannot reproduce the pool).

**Admissibility (pre-registered):** at every swept pos —
(a) monotonicity `dot_only ≤ no_av ≤ full_local`;
(b) every marginal ≥ 0 and `dot + softmax + AV` within ±10% of
`full_local`; (c) `full_local` within ±5% of `full`. Any failure → the
decomposition is not trusted, no lever gate may consume the run, and
the instrument finding is itself the milestone's recorded outcome
(sanctioned STOP-out).

**Where it runs:** local dev box first — the Lever 1 gate input
(M4b.14 Task 7 precedent: non-quiet, honestly labeled, gates on ratios
not absolutes) — then re-run verbatim in each quiet-hw session for the
record.

## Pre-registered gates

Formulas fixed here, before any measurement; verdicts recorded once in
§Amendments with the arithmetic shown. No lever ships without its gate.

**Gate 1a — dot blocking.** Ship iff, on the local µbench:
(a) the dot marginal ≥ **15%** of `full_local` at pos 511 and 639, AND
(b) `dot_blocked` beats `full_local` by ≥ **10%** whole-call at both of
those positions, at the best block size N over the swept {2, 4, 8} —
that N is the one that ships.

**Gate 1b — register AV.** Same rule with the AV marginal and
`av_regacc`.

Gates 1a and 1b are evaluated independently; zero, one, or both fire.
If both fire, `combined` must also beat each single probe (sanity; a
regression there is recorded and the better single lever ships alone).

**Softmax-blamed escape.** If the softmax marginal exceeds both other
marginals at pos ≥ 511, no scoped bit-neutral lever exists
(`expf_avx2` is already 8-wide): Lever 1 STOPs, the finding is
recorded, and the milestone proceeds to the sessions for the diagnostic
record.

**Headroom-set target** (fixed at each session's mid-gate, judged at
closing, per box):

```
tg_target = baseline_tg × (1 + S × r)
```

- `baseline_tg`: re-benched in-session on the same box instance before
  the Lever 1 binary runs (M4b.11 rule — within-session ratios are the
  comparable quantity).
- `S`: attention's share of decode wall from the session's fresh best-t
  split-bracket profile of the *baseline* binary.
- `r`: whole-call kernel reduction (`full` old vs new via public
  symbol), re-measured by the µbench **on that box**, averaged over
  pos {511, 639}.

Conservatism as in M4b.11: `(1 + S·r)` understates what a wall-share
reduction implies (`1/(1−S·r)`) but ignores second-order effects; the
target nets those against each other. If Lever 1 shipped nothing
(gates fail / softmax escape), the target degenerates to
`baseline × 1.0` and the milestone closes as a diagnostic on the
instrument findings alone.

**Gate 2 — KV-position-split kernel (Lever 2).** Evaluated at each
session's mid-gate from *post-Lever-1* measurements: a `pool-profile`
dispatch-split profile (M4b.12 instrument, rebuilt on the Lever 1
binary) and the fresh best-t profile.

```
P2 = S_residual × c
```

- `S_residual`: attention share of decode wall post-Lever-1 (slowest-
  lane kernel sum × calls / decode total, M4b.12 arithmetic).
- `c`: the drain fraction of the instrumented call — the share
  sub-head granularity can reclaim (16c baseline 26.9%; 8c ~6%, which
  forecloses authorization there absent a surprise).

Thresholds, M4b.11 verbatim: **≥ 5% on both machines → authorized;
< 3% on both → STOP; split → judgment call, argument recorded in the
amendment.** The expected outcome is a split (16c yes-ish, 8c no) — the
judgment call must weigh that Lever 2's tolerance re-derivation is paid
once but its win may exist on one box only.

**Lever 2 numerics (pre-committed, if authorized):** a KV-position
split merges per-split running maxima and exp-sums — reduction order
changes, so outputs change. `attn_rel_tol` (and `logits_abs_tol` if the
artifact differential moves) get a documented re-derivation against
observed error distributions **before** any win claim — never
loosened-to-green. The split kernel is a new pool entry point beside
`inferno_par_attention_heads` (dispatch machinery untouched); ABI and
artifact-cache-key impact assessed in the implementation plan. If the
re-derivation would exceed what the observed distributions justify,
Lever 2 reverts and the STOP is recorded (M4b.2 revert discipline).

## Lever 1 — the bit-neutral fixes

**Multi-position-blocked QK dot** (`dot_blocked` promoted): process N
positions per outer step, each with its own accumulator register; the
loads interleave and the FMA chains overlap. Each position's dot
retains today's partition order and `hsum8` tree — **bit-identical per
position by construction**. N is the Gate 1a winner over the swept
{2, 4, 8}, fixed before shipping.

**Register-resident AV accumulation** (`av_regacc` promoted): hold
`out[h·64 .. h·64+64)` in 8 YMM accumulators across the whole visible
loop; store once at the end. Accumulation order per output element
stays t-ascending — **bit-identical by construction**.

Both land inside `attn_core_avx2` only. The scalar kernel is untouched
(its outputs are already the reference; Lever 1 does not change any
output anywhere). The `_hspan` and whole-call public symbols keep their
signatures; the qblock prefill kernel is untouched.

**Testing:** rig exact-equality (M2 pattern) extended — blocked-dot and
register-AV paths vs `attention_reference()` bit-exact across the
geometry sweep including `visible` edge cases (1, 7, 8, 9, block-size
boundaries); `_hspan` ≡ whole-call over every head split (existing
hspan tiling tests re-run against the new core); `m == 1` compiled-path
identity; `cargo test -p inferno-codegen --test differential` and
`cargo test -p inferno-core --test artifact` green with tolerances
untouched.

## Tasks

1. **µbench** — build `attn_decode_phases` (variants, probes, anchors,
   cold/warm axis); run locally; record the full table + admissibility
   verdict in §Amendments.
2. **Local gate verdicts** — Gates 1a/1b (and the softmax escape if it
   fires) computed from the Task 1 table, arithmetic recorded, before
   any shipping-code change.
3. **Lever 1 implementation** — only gate-passed fixes; rig equality
   tests extended; both differentials green; `mise run test` +
   `mise run lint`.
4. **Local data point** — µbench re-run on the shipped kernel (public
   symbol) + local e2e bench, honestly labeled non-quiet.
5. **Quiet-hw session A (16c d2.c1.medium)** — baseline re-bench;
   baseline best-t profile → `S`; µbench (record + `r`); headroom
   target fixed; Lever 1 protocol runs; post-Lever-1 `pool-profile`
   dispatch-split profile; Gate 2 inputs recorded. Provisioned per the
   metal runbook, `perf` on PATH, `metal-gc` + zero-server check after.
6. **Quiet-hw session B (8c s2.c2.medium)** — same protocol.
7. **Gate 2 verdict** — computed once from both sessions' amendments,
   arithmetic shown.
8. **Lever 2** — only if authorized: KV-position-split kernel,
   documented tolerance re-derivation before any claim, its own local
   gate + rig tests, and a follow-up session pair for its closing data
   point. If Gate 2 STOPs or splits-to-no, Tasks 5–6 double as the
   closing sessions (M4b.13/M4b.14 precedent).
9. **Closing verdict** — exit-criteria walk in §Amendments; tg vs the
   headroom-set targets per box; v1 ratios recorded as context in the
   M4a spec §Amendments.

## Exit criteria

1. µbench built; admissibility recorded locally and on both boxes.
2. Every gate verdict (1a, 1b, softmax escape if fired, Gate 2)
   recorded once with the arithmetic shown; no lever shipped without
   its gate.
3. All standing invariants held: rig identities bit-exact, tolerances
   untouched — unless Lever 2 fired, in which case its re-derivation is
   documented in §Amendments with the observed distributions.
4. Closing tg judged against the headroom-set targets on both boxes;
   v1 context recorded, never the gate.
5. Every STOP recorded as a finding. Sanctioned STOP-outs: instrument
   inadmissible; softmax-blamed; both Lever 1 gates fail; Gate 2
   STOP/split-to-no. Each closes the milestone as a diagnostic
   (M4b.12 precedent: an all-STOP with the finding is a successful
   outcome).

## Risks

- **The marginals may not decompose cleanly** — out-of-order overlap
  means `dot_only` + softmax + AV can undershoot `full` (phases hide
  under each other). The ±10% identity check is the tripwire; an
  inadmissible decomposition is itself a finding (which phases overlap
  is information about the stall structure).
- **Compiler autovectorization drift in bench-local copies** —
  `full_local` vs `full` (±5%) guards it.
- **`r` measured single-threaded may not transfer to the pool** (14
  lanes share L2/DRAM differently than one). Mitigation: the headroom
  target uses in-session `r` on the criterion box, and the closing
  judgment is against the target, not the µbench.
- **The 8c box may again be the wall** — at c=½, attention arithmetic
  reaches only ~0.97x there. The headroom-set exit absorbs this
  honestly; the closing verdict must state what the residual decode
  wall is shaped like (GEMV bandwidth?) for the next milestone's
  scoping, exactly as M4b.14 did for prefill.
- **Lever 2's re-derivation cost is paid before its win is proven** —
  that is why it sits behind Gate 2 with M4b.11's thresholds, and why
  the split-verdict judgment must be recorded, not hand-waved.

## Amendments

Session records, gate verdicts, and the closing exit-criteria walk are
appended here as they land; recorded data points are never edited
(erratum pattern if a correction is needed).

### 2026-07-17 — Task 5: local µbench + Lever 1 gate verdicts (non-quiet dev box)

**Machine:** AMD Ryzen 9 3900 12-Core (24 threads), NON-QUIET dev box
(shared host; load average ~18.7 during the run — ambient noise high,
ratios are the quantity of record per the spec's local-gate rule).
Bench commit: 762e131. Full run: `cargo bench -p inferno-kernels --bench
attn_decode_phases`, bit-identity assert passed (all probes, pos 9/639).

**Criterion means (µs), full run:**

| pos | full | full_local | dot_only | no_av |
|----|------|-----------|----------|-------|
| 127 | 21.886 | 18.134 | 7.386 | 8.732 |
| 511 | 96.672 | 72.509 | 28.929 | 34.169 |
| 639 | 127.670 | 97.870 | 37.381 | 42.884 |
| 1023 | 206.739 | 134.299 | 67.298 | 76.144 |
| 2047 | 426.617 | 331.395 | 153.193 | 171.021 |

| pos | dot_blocked2 | dot_blocked4 | dot_blocked8 | av_regacc | combined |
|----|----|----|----|----|----|
| 127 | 15.332 | 14.484 | 14.530 | 12.935 | 10.568 |
| 511 | 66.241 | 56.206 | 65.732 | 61.705 | 52.554 |
| 639 | 79.206 | 73.097 | 72.923 | 73.929 | 65.804 |
| 1023 | 124.753 | 122.812 | 123.482 | 136.952 | 115.291 |
| 2047 | 250.970 | 249.325 | 250.149 | 273.728 | 238.617 |

Anchors: kv_stream 44.954 / 181.304 / 223.739 / 358.448 / 723.241 µs at
pos 127/511/639/1023/2047; fma_peak 373.675 µs. Cold/warm (one head):
pos 639 cold 69.991 vs warm 6.363 µs; pos 2047 cold 209.074 vs warm
21.823 µs.

**Admissibility (pre-registered, spec §The instrument):**
- (a) monotonicity dot_only ≤ no_av ≤ full_local: PASS at all five pos.
- (b) marginals ≥ 0 (all are; see below), sum = full_local exactly
  (telescoping identity — the sum check is vacuous by construction;
  noted for the record). Marginals (µs) dot/softmax/AV: 127:
  7.386/1.346/9.402; 511: 28.929/5.240/38.340; 639: 37.381/5.503/54.986;
  1023: 67.298/8.846/58.155; 2047: 153.193/17.828/160.374.
- (c) full_local within ±5% of full: **FAIL at every pos.**
  full_local/full = 0.8286 (−17.1%) at 127; 0.7501 (−25.0%) at 511;
  0.7666 (−23.3%) at 639; 0.6496 (−35.0%) at 1023; 0.7768 (−22.3%)
  at 2047.

**VERDICT: STOP (instrument finding).** Admissibility check (c) failed;
per the pre-registered rule the decomposition is not trusted, no lever
gate may consume the run, and the instrument finding is the recorded
outcome. **Gates 1a/1b: NOT EVALUATED (VOID — inadmissible run).**
Tasks 6–7 (Levers 1a/1b) are SKIPPED. The softmax-escape check is moot
for gating; for the record the softmax marginal is the smallest of the
three at every pos ≥ 511.

For transparency, the raw ratios the gates would have read (VOID, not
consumable, no lever ships from them): dot marginal 39.9%/38.2% of
full_local at 511/639; AV marginal 52.9%/56.2%; dot_blocked4 −22.5%/
−25.3% whole-call vs full_local; av_regacc −14.9%/−24.5%.

**Diagnostic evidence (stability + mechanism), recorded as evidence:**
1. Focused re-run (same box, later, `-- 'phases/full'`): full_local/full
   = 0.708 / 0.764 / 0.797 / 0.787 at pos 127/511/639/1023 — the gap
   reproduces with disjoint 95% CIs; systematic, not noise.
2. Disassembly of the bench binary (attn_decode_phases-cf66c9077dc73ab6):
   `inferno_attention_f32_avx2` is a thunk to a NON-INLINED
   `attn_core_avx2` whose QK dot compiles to a 4-instruction dynamic
   loop (vmovups, vfmadd231ps, add, cmp/jb — one FMA per iteration,
   runtime head_dim bound). The bench-local `full_local`, with
   HEAD_DIM=64/KV_DIM=128 as compile-time consts, compiles the same
   source loop to a fully-unrolled straight-line 8×FMA sequence with
   constant offsets and no loop control.
3. Interpretation (hypothesis, labeled): the frozen copy is
   const-geometry-specialized by LLVM; the shipping kernel pays a
   ~20–35% whole-call penalty for runtime-dim genericity at protocol
   geometry. Check (c) exists precisely to catch copy-vs-symbol
   codegen artifacts and did. The marginals in this table attribute the
   SPECIALIZED kernel's phases, not the shipping kernel's — hence
   inadmissible for Lever 1 gating.

**Roofline context (per pre-registration, feeds no gate):** at pos 639
the AV marginal (54.99 µs) and dot marginal (37.38 µs) sit well above
the kv_stream anchor scaled to their footprint (K-half ≈ 112 µs for
both K and V regions ≈ 224 µs total stream at serial-add latency —
the anchor is latency-chained, not bandwidth-limited, so it bounds
loosely on this noisy box). Cold/warm delta ≈ 11× at one-head
granularity (6.4 → 70.0 µs at pos 639) — cache state can explain
large per-lane spreads.

**Instrument finding for the milestone record:** the decode attention
µbench cannot gate bit-neutral micro-levers on this instrument design —
the bench-local copies measure a const-specialized variant of the
kernel. The actionable diagnostic is the gap itself: runtime-dim
genericity costs ~20–35% whole-call at protocol geometry on this box;
a const-specialized (or dim-monomorphized) kernel is the natural next
lever, but it is out of M4b.15's pre-registered scope and needs its own
plan. Per plan Task 5 Step 2.2: proceed to sessions for the quiet-hw
diagnostic record.

### 2026-07-17 — Task 8: local post-Lever-1 data point (non-quiet) — DEGENERATE (no lever shipped)

Lever 1 shipped nothing (Task 5 STOP), so the plan's Step 1 µbench
re-run for `r` is moot: `full` is the unchanged shipping kernel and
r := 0; per the spec the headroom target degenerates to
`baseline × 1.0` and the milestone closes as a diagnostic. Recorded
here: the local e2e context point only, honestly labeled.

**Local e2e (NON-QUIET dev box, AMD Ryzen 9 3900, ambient load high —
context only, feeds no gate):** inferno 0.1.0 (c45e19b) vs llama.cpp
6f4f53f, qwen2.5-0.5b-instruct-q8_0.gguf, pp=512 tg=128 reps=5:

```
engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          12      190.47 ± 6.94        17.34 ± 0.44
inferno (t=1 diag)           1       61.53 ± 4.45        12.24 ± 0.73
llama.cpp                   12      109.87 ± 30.93        6.71 ± 1.44
llama.cpp (t=1 diag)         1       74.63 ± 4.03        17.42 ± 1.07
ratio (inferno/llama.cpp): pp 1.73x | tg 2.59x
```

### 2026-07-18 — Session A — d2.c1.medium (16c Xeon Gold 6336Y), quiet-hw

**Machine:** Intel Xeon Gold 6336Y @ 2.40GHz, 32 logical CPUs, kernel
6.9.10+bpo-amd64. PREFLIGHT: FIT (psi 0.00, throttled_delta 0, tsc ok).
**DEVIATION:** perf unavailable (`linux-perf` not locatable on the box
image; recorded per plan). First provision attempt (server
6a5abc336df43c5b5fbca180) died to a transient SSH-tunnel drop mid-
workload after the baseline protocol table; runner delete + metal-gc
--force (409/403-then-clear quirk) confirmed zero servers before the
retry. Retry = this session. Raw artifacts:
`target/metal/d2.c1.medium-20260718T000851Z` (workload.log, criterion
tree, quiet-hw dir; local-only, gitignored).

**Protocol tables:** recorded verbatim in the M4a spec §Amendments
(standing rule), including the both-engines t=1 drift caveat.
`baseline_tg` = **57.26** (t=16, baseline binary 9883086);
lever-binary tg 59.12 / pp 924.67 (066bd45, same kernel — no lever).

**Baseline best-t attention share `S`** (gate-decode-attr, t=16 decode):
attention 1238854680 / 4250870946 cyc = **29.1%** (t=1: 34.0%).

**Headroom target:** no lever shipped → r := 0;
`tg_target = 57.26 × (1 + 0.291 × 0) = 57.26 × 1.0 = 57.26` (degenerate
per Task 5 STOP). Closing judges tg against this.

**µbench on this box (criterion means, µs):**

| pos | full | full_local | dot_only | no_av |
|----|------|-----------|----------|-------|
| 127 | 31.096 | 18.518 | 7.646 | 10.931 |
| 511 | 121.360 | 75.650 | 30.595 | 35.212 |
| 639 | 150.547 | 95.061 | 38.792 | 53.943 |
| 1023 | 241.093 | 151.257 | 75.397 | 74.637 |
| 2047 | 795.832 | 570.608 | 157.242 | 172.609 |

Probes (µs): dot_blocked2 17.049/67.606/84.473/134.390/568.041;
dot_blocked4 15.416/65.810/82.113/131.077/577.720; dot_blocked8
18.267/67.244/84.007/134.070/572.658; av_regacc 17.629/72.265/90.607/
143.831/581.444; combined 15.711/62.878/78.543/125.257/538.831 (pos
127/511/639/1023/2047). Anchors: kv_stream 38.776/155.235/194.027/
310.003/621.163 µs; fma_peak 294.164 µs. Cold/warm one-head: 639:
33.031 vs 6.440 µs; 2047: 99.844 vs 42.016 µs.

**µbench admissibility on this box:**
- (c) full_local/full = 0.5955 (−40.4%) / 0.6233 (−37.7%) / 0.6314
  (−36.9%) / 0.6274 (−37.3%) / 0.7170 (−28.3%) at pos 127/511/639/
  1023/2047 — **FAIL at every pos** (bound ±5%).
- (a) monotonicity: **FAIL at pos 1023** (no_av 74.637 < dot_only
  75.397; softmax marginal −0.76 µs, noise-level inversion) — PASS at
  the other four pos.
- **The Task 5 instrument finding REPRODUCES on quiet hardware,
  stronger:** the shipping kernel's runtime-dim genericity gap vs the
  const-specialized copy is −37% at the protocol positions on this
  Intel box (vs −23…−25% on the AMD dev box). Machine-independent
  codegen artifact, not noise and not load.

**Dispatch-split (gate-attn-split, lever binary):** sum identity vs
op-profiler attention 99.4% (admissible 90–110%). Buckets: publish
0.8%, kernel(shard0) 73.7%, **drain 25.5%** → Gate 2 `c = 0.255`.
Slowest-lane kernel spread visible (lane sums 0.88–1.14 Gcyc across 14
active lanes).

**`S_residual`** = S = **29.1%** (identical kernel — no lever shipped;
the lever-binary split profile is the same binary's dispatch view).

### 2026-07-18 — Session B — s2.c2.medium (8c Xeon E-2388G), CHI, quiet-hw

**Machine:** Intel Xeon E-2388G, 8 physical cores, CHI (PHX 406 —
no stock; catalog showed NLD,SGP,CHI; `--location CHI` per runbook).
PREFLIGHT: FIT. **DEVIATION:** perf unavailable (same as session A).
First attempt was the PHX 406 (no server created); gc-confirmed zero
before retry. Raw artifacts:
`target/metal/s2.c2.medium-20260718T010355Z` (local-only, gitignored).

**Protocol tables:** recorded verbatim in the M4a spec §Amendments.
`baseline_tg` = **62.40** (t=8, baseline 9883086); lever-binary tg
62.52 / pp 737.23 (066bd45, same kernel). Ratios 0.70x/0.86x baseline,
0.72x/0.86x lever (best-of-builds 0.70x/0.87x and 0.71x/0.87x).

**Baseline best-t attention share `S`** (gate-decode-attr, t=8 decode):
attention 779547772 / 3781606717 cyc = **20.6%** (t=1: 33.6%).

**Headroom target:** r := 0 (no lever) →
`tg_target = 62.40 × 1.0 = 62.40` (degenerate per Task 5 STOP).

**µbench on this box (criterion means, µs):**

| pos | full | full_local | dot_only | no_av |
|----|------|-----------|----------|-------|
| 127 | 21.904 | 13.429 | 5.891 | 6.917 |
| 511 | 88.900 | 62.199 | 21.059 | 25.282 |
| 639 | 121.502 | 87.761 | 29.204 | 30.993 |
| 1023 | 223.636 | 189.547 | 62.339 | 59.385 |
| 2047 | 449.419 | 373.214 | 156.959 | 148.495 |

Probes (µs, pos 127/511/639/1023/2047): dot_blocked2 13.119/50.464/
85.557/160.703/389.041; dot_blocked4 12.802/53.314/81.747/173.312/
358.116; dot_blocked8 13.355/55.065/89.450/171.659/314.038; av_regacc
12.018/50.981/91.823/178.579/405.840; combined 10.748/46.096/76.037/
144.778/335.517. Anchors: kv_stream 27.638/110.635/138.758/221.227/
443.458 µs; fma_peak 210.027 µs. Cold/warm one-head: 639: 30.427 vs
6.378 µs; 2047: 91.289 vs 27.153 µs.

**µbench admissibility on this box:**
- (c) full_local/full = 0.6131 (−38.7%) / 0.6997 (−30.0%) / 0.7223
  (−27.8%) / 0.8476 (−15.2%) / 0.8304 (−17.0%) at pos 127/511/639/
  1023/2047 — **FAIL at every pos** (bound ±5%).
- (a) monotonicity: **FAIL at pos 1023 AND 2047** (no_av < dot_only;
  softmax marginal −2.95 µs and −8.46 µs) — PASS at 127/511/639.
- **The instrument finding reproduces on the third machine:** the
  const-specialized copy beats the shipping kernel by −28…−30% at the
  protocol positions on E-2388G (AMD dev box −23…−25%, 6336Y −37%).

**Dispatch-split (gate-attn-split, lever binary, default shards):**
1536 calls, sum identity 99.6% (admissible). Buckets: publish 0.2%,
kernel(shard0) **97.7%**, **drain 2.1%** → Gate 2 `c = 0.021` (spec
§Pre-registered gates forecast "8c ~6%"; measured even lower).

**`S_residual`** = S = **20.6%** (identical kernel — no lever shipped).
