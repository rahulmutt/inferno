# M4b.10 — Decode-Cap Formula Revision Design

**Date:** 2026-07-12
**Status:** Approved design, pre-implementation
**Milestone:** M4b.10 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; closes the exit-criterion leg 2 that the
[M4b.5](2026-07-08-m4b5-phase-aware-decode-threading-design.md) spec has
carried as DEFERRED since 2026-07-08)

M4b.5 shipped `decode_cap = clamp(active / 3, 2, active)` as an explicitly
**reversible starting hypothesis**, with its performance leg deferred to a
quiet-hardware sweep. That sweep has now run three times on quiet bare metal
and the hypothesis has failed three times. This milestone replaces the
formula — but only after collecting the evidence that the recorded verdict
said was missing.

## Motivation — three recorded misses, and a cap that may prevent nothing

The `gate-decode-cap` sweep (d2.c1.medium → Xeon Gold 6336Y, 16 physical /
32 logical, PREFLIGHT FIT, reps=3 interleaved) has been recorded three
times in the [M4b.5 spec](2026-07-08-m4b5-phase-aware-decode-threading-design.md)
§Amendments:

| session | inferno @ | knee (best fixed cap) | shipped default (cap 5) | regret |
|---|---|---|---|---|
| 2026-07-11 (1st) | 6b0df49 | 13 → 63.98 tok/s | 55.93 tok/s | **−9.82%** |
| 2026-07-11 (2nd) | 1804d9f | 12 → 63.88 tok/s | 56.73 tok/s | **−11.19%** |
| 2026-07-12 (3rd) | 823437f | 13 → 63.49 tok/s | 55.83 tok/s | **−11.76%** |

Two facts fall out, and the second is the more interesting one.

**The shipped default leaves ~10–12% of decode on the table**, reproduced
three times on the same silicon. Every decode measurement inferno currently
takes — including the M4a headline tg ratio — is taken on a baseline
throttled by our own heuristic.

**The high-thread cliff the cap exists to prevent does not exist on this
box.** Uncapped-equivalent (cap = 16) lands at 61.03 / 63.47 / 62.66 tok/s
— a regret of 4.6% / 0.6% / 1.3% against the knee, i.e. a flat plateau from
cap 8 to 16, not a cliff. The *only* evidence for a cliff was the original
knee-at-4-on-12-cores point, and that was measured on the **8-CPU
cgroup-quota'd shared devpod** — where running 12 threads against an 8-CPU
quota throttles. A quota-induced throttle is exactly what a fake knee looks
like. The cap may be solving a problem that only ever existed inside a
container.

The recorded verdict declined to pick a replacement on this evidence:

> the `active/3` constant is wrong on this machine class, but one quiet knee
> point is not enough to pick a replacement — the formula revision is
> deferred to a scoped follow-up with at least one more quiet machine
> (ideally a different core count / memory class).

This is that follow-up.

### Why we cannot look the answer up

M4b.5 names the physically motivated optimum:
`total_DRAM_bandwidth / per_core_streaming_bandwidth`. That is essentially a
cores-per-memory-channel ratio — but we cannot obtain it from spec sheets:

- **Channel *population* is not published.** PhoenixNAP does not document
  DIMM layout, and channels supported ≠ channels populated.
- **The hardware table has already drifted from reality.** `d2.c1.medium`
  was catalogued as a Xeon 5315Y and delivers a 6336Y (commit f72d67c,
  observed twice). The delivered box reports 16 physical / 32 logical, which
  matches neither part's published core count.
- **`d2.c5.large` is dual-socket** (the Platinum 8352Y is a 32-core part, so
  64 physical cores means two sockets), which makes "cores per channel"
  ambiguous — two sockets carry 16 channels, but a NUMA-local allocation
  sees 8.

A formula keyed on looked-up hardware facts would therefore be built on
sand. So this design **measures the model's input directly** instead of
looking it up, which also makes the model falsifiable.

## Scope Decisions (M4b.10)

| Decision | Choice |
|---|---|
| Phase | **Decode only.** Prefill (`inferno_par_gemm`, `inferno_par_attention`, `inferno_par_token_loop`) is untouched |
| Lever | Replace the `decode_cap` formula in `inferno-core`. **Pool-side only** — no codegen edit, no `HOST_ABI_VERSION` bump, **no recompile**; an existing cached `model.so` benefits immediately (M4b.5 precedent) |
| Evidence | Three quiet-hw sessions, each collecting **two** curves on the same box in the same session: the decode-knee sweep (`gate-decode-cap.sh`) and a new **bandwidth-saturation curve** (`gate-bw-curve.sh`) |
| Machines | `d2.c1.medium` (16c — re-run: we have three knee sessions but *no* bandwidth curve), `s2.c2.medium` (Xeon E-2388G, 8c), `d2.c5.large` **pinned to one socket** via `numactl --cpunodebind=0 --membind=0` (32c, NUMA-free) |
| Formula choice | **Pre-registered decision rule** (below), fixed before any data is taken. Candidates: remove the cap, refit the constant, or ship a runtime bandwidth probe |
| Correctness | The cap is **provably bit-invisible** — `shard_table` computes each output row entirely within one lane, so the cap only regroups rows. The existing `par_gemv` cap-invariance test remains the standing guard |
| Tolerances | **None touched.** No numerics change; the differentials must pass as-is |
| Override | `INFERNO_DECODE_THREADS` survives in every outcome |
| Measurement discipline | Quiet bare metal via `mise run metal`; regret computed **within-session** from per-rep ratios in the same interleaved round (standing M4b discipline) |

**Explicitly out of scope:**

- **Decode attention parallelism and F16 KV** — M4b.2's decode attribution
  fork is still open and is the *next* item. It is deliberately sequenced
  after this milestone so it is measured against a de-throttled baseline
  rather than a cap-5 one.
- **NUMA-aware threading** — the dual-socket box is socket-pinned precisely
  to keep NUMA out of scope, as it has been since M4b.1.
- **Prefill anything** — the M4b.1 prefill scaling gate is MET (10.63x @
  t=12, M4b.9) and closed.
- **Any CI perf gate** — standing rule (AGENTS.md).

## The pre-registered decision rule

Written down **before** any sweep runs. This project pre-registers its
attribution forks (M4b.2, M4b.8, M4b.9) and that discipline is why the
M4b.9 verdict was trustworthy; the same applies here.

**Definitions.** For each machine `M`, the sweep yields a median throughput
per cap value. Let `best_fixed(M)` be the cap with the highest median and
`T_best(M)` its throughput. For a candidate cap `c`:

```
regret(c, M) = (T_best(M) − T(c, M)) / T_best(M)
```

Regret is always computed **within a session**, from per-rep ratios taken in
the same interleaved round. This cancels the frequency/turbo drift that makes
these boxes' absolute single-thread decode bimodal across sessions (t=1
medians of 23.79 / 16.94 / 18.28 tok/s are already recorded in the M4b.5
ledger), and the existing gate script already computes ratios this way.

**One session per machine is authoritative for the rule.** `d2.c1.medium`
will have four knee sessions once its bandwidth curve is taken; the rule
consumes **the new session** (the only one carrying a paired bandwidth curve,
which rule 2 requires). The three prior sessions serve as corroboration — if
the new session's regret figures fall outside their recorded spread, that is
a measurement problem to resolve *before* applying the rule, not a datum to
average away.

**Candidates.**

- **U** (uncapped): `c = active`.
- **K_k** (static): `c = clamp(round(k · active), 2, active)` for
  `k ∈ {⅓ (current), ½, ⅔, ¾, 1}`.
- **P** (bandwidth model): the smallest lane count reaching ≥95% of peak
  aggregate streaming bandwidth on the measured bandwidth curve.

**The rule, applied in order:**

1. If **regret(U) ≤ 5% on all three machines** → **remove the cap.** It
   prevents nothing; delete the cap and its heuristic.
2. Else if **P is validated** (regret(P) ≤ 5% on all three machines) **and**
   P beats the best static `K_k` on worst-case regret by **≥3pp** → **ship
   the runtime bandwidth probe.** The physical model is real and generalizes
   to hardware we have never rented.
3. Else → **ship the static `K_k` with the lowest worst-case regret** across
   the three machines. If it ties with U within 2pp, prefer **U**
   (simplicity).
4. If **regret(U) > 15% on any machine**, a genuine cliff exists and a cap is
   mandatory; rules 2 and 3 decide which one.

**The 5% threshold is derived, not invented.** Uncapped's regret on the
6336Y spans 0.6–4.6% across three sessions, so a candidate within 5% is
indistinguishable from optimal at this measurement precision — while the
shipped default's 9.8–11.8% sits clearly outside it.

**The rule is falsifiable.** Rule 2 fires only if the measured bandwidth
curve *predicts* the measured decode knee. If it does not, the physical
model is refuted and we record that, rather than fitting a constant and
calling it a theory.

For reference, the candidates' regret on the evidence we already hold
(6336Y, three sessions):

| candidate | cap on this box | regret (1st / 2nd / 3rd) |
|---|---|---|
| shipped `K_⅓` | 5 | −9.82% / −11.19% / −11.76% |
| `K_¾` | 12 | −2.00% / 0.00% / −0.31% |
| **U** (uncapped) | 16 | −4.61% / −0.64% / −1.31% |

Both replacements beat the shipped default decisively on this box. They
**diverge maximally on the 32-core box** (U says 32 lanes, `K_¾` says 24),
which is what the new machines buy.

## Design

### `crates/inferno-pool/examples/bw_curve.rs` (new)

Drives the **real Q8_0 GEMV kernel** through the real `Pool` at lane counts
`1..capacity`, over a synthetic weight matrix sized past L3 so it genuinely
streams from DRAM — decode's actual access pattern, not a synthetic memcpy.
`inferno-pool` already carries `inferno-kernels` as a dev-dependency, so this
needs no new dependency edge.

Reports aggregate GB/s per lane count, speedup vs one lane, and derives **P**
(smallest lane count reaching ≥95% of peak). The artifact serves double duty:
it is the session's curve-2 measurement, and it is the prototype of the
runtime probe that rule 2 would ship.

### `scripts/quiet-hw/gate-bw-curve.sh` (new)

Wraps the example with the same `lib.sh` / `machine_block` /
interleaved-median discipline as the other four quiet-hw gates. Verdict
destination: this spec's Amendments.

### `scripts/quiet-hw/gate-decode-cap.sh` (edit)

- **Coarse grid above 16 cores.** It currently sweeps `seq 1 $PHYS`, which is
  32 cap values on the pinned 8352Y. Sweep 1..16 fine-grained, then step 4, to
  bound session time.
- **NUMA pinning.** Honor `QHW_NUMA_NODE=N`: wrap each run in
  `numactl --cpunodebind=N --membind=N` and derive `phys_cores` from the
  pinned node. This is what makes the socket-pinned 8352Y session honest.

### `crates/inferno-core/src/lib.rs` (edit)

`decode_cap` takes whichever shape the rule selects:

- **remove** → `override.unwrap_or(active)`;
- **static** → `clamp(round(k · active), 2, active)` with the selected `k`;
- **probe** → a bounded call to a new `inferno_pool::measure_decode_knee()`
  at pool init, clamped to `[1, capacity]`, with a deterministic fallback to
  `active` if the probe fails.

The `INFERNO_DECODE_THREADS` override keeps precedence in all three shapes.

### Docs

`docs/runbooks/quiet-hw-verification.md` gains the `gate-bw-curve` row
(output file → verdict destination), matching the existing gate table.

### Invariants (all inherited, none loosened)

1. **The cap never changes output bits.** `shard_table` computes each output
   row entirely within one lane, so the cap only regroups rows into shards.
   M4b.5 proved this and left the `par_gemv` cap-invariance test (sweeping
   `1..=capacity`) as the standing guard. This is what makes even a
   machine-measured, run-to-run-varying cap numerically inert — the usual
   objection to auto-tuning does not apply here.
2. **No tolerance loosening** — compiled-vs-interpreter (`inferno-codegen`)
   and artifact (`inferno-core`) differentials green with existing bounds.
3. **The t=1 nightly is unaffected by construction** — at `active = 1` every
   candidate formula resolves to 1.

## Testing plan

- **`decode_cap` unit tests** rewritten for the shipped formula (they
  currently assert `active/3`), including the `INFERNO_DECODE_THREADS`
  override, rejection of garbage/zero, and `active == 1`.
- **Cap-invariance** (existing `par_gemv` sweep `1..=capacity`) stays green —
  the guard that the cap is bit-invisible.
- **If the probe ships:** the probe result is clamped to `[1, capacity]`; a
  probe failure falls back to `active` rather than panicking, with a unit test
  asserting it.
- **Existing gates:** `cargo test -p inferno-codegen --test differential` and
  `cargo test -p inferno-core --test artifact` green with **no tolerance
  edits** (AGENTS.md standing rule).
- **`mise run bench-compiled`** (pinned `--threads 1`) stays green.
- No new kernel math, so no new scalar↔SIMD bit-identity rigs.

## Verification protocol and verdict gate

1. **Three quiet-hw sessions** via `mise run metal`, each running
   `gate-decode-cap` **and** `gate-bw-curve`: `d2.c1.medium` (16c),
   `s2.c2.medium` (8c), `d2.c5.large` socket-pinned (32c). Record every sweep
   verbatim; never edit a recorded data point.
2. **Apply the pre-registered rule** to select the formula. Ship it.
3. **Exit criterion — performance:** the shipped default's **worst-case
   regret ≤ 5% across all three machine classes**. Recorded as amendments in
   the [M4b.5 spec](2026-07-08-m4b5-phase-aware-decode-threading-design.md)
   §Amendments (the leg-2 ledger, where the three recorded misses live) and
   cross-linked here. **This closes M4b.5's exit-criterion leg 2, open since
   2026-07-08.**
4. **Exit criterion — correctness:** cap-invariance and both differentials
   green, no tolerance touched.
5. **Re-record the M4a headline** (tg ratio) in the same session as the new
   default. The expected move is 0.84x → ~0.94x; if it is still below 1x it
   is recorded **NOT MET**. No silent gate-loosening — the v1 win criterion
   stays owned by the M4a spec.
6. **Record the model verdict explicitly:** did the bandwidth curve predict
   the decode knee? If it did not, the physical model is **refuted** and
   approach P is retired permanently. That is a real finding, not a null
   result, and it must be written down either way.

## Risks

- **All three machines say "no knee", and the deliverable is a deletion.**
  This is the likeliest outcome given the flat 8–16 plateau already
  recorded. The bandwidth-curve work would then have bought only the
  confidence to delete — which is still the right trade against shipping a
  third unfounded constant, and the curve remains as diagnostic surface.
- **The knee is HT contention, not bandwidth.** These boxes report 2x
  logical/physical, and `active` defaults to *physical* cores, so the sweep
  never oversubscribes. But if the knee tracks logical-core pressure rather
  than DRAM saturation, the bandwidth model will mispredict. Detector: the
  bandwidth curve saturating at a different lane count than the decode knee —
  exactly what rule 2's validation test measures. This is the risk the
  falsifiable rule exists to catch.
- **The probe adds startup latency** if rule 2 fires. Mitigation: the probe
  is bounded (target ~10ms) and `INFERNO_DECODE_THREADS` skips it entirely.
  Perf *reproducibility* across runs would need the env override pinned in
  the bench protocol; output determinism is unaffected (invariant 1).
- **Session time on the 32-core pinned box.** Caps 1..32 × 3 reps × 128
  tokens is ~100 runs; the coarse grid above 16 bounds it.
- **Bimodal single-thread decode** on these boxes (turbo/frequency behavior,
  already in the M4b.5 ledger) adds noise to absolute numbers. Regret is
  computed within-round, so it cancels.

## Amendments

*(none yet — sweeps pending)*
