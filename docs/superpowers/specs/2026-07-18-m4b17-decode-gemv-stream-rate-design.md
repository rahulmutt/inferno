# M4b.17 — Decode GEMV Stream-Rate Attribution + Gated Bandwidth Levers Design

**Date:** 2026-07-18
**Status:** Approved design, pre-implementation
**Milestone:** M4b.17 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.16](2026-07-18-m4b16-emitted-decode-attention-design.md))

This milestone follows M4b.16's closing scoping verdict verbatim: "target
the GEMV/bandwidth axis or accept the t-scaling ceiling; further
decode-attention micro-levers are not supported by this data." It is
attribution-first (M4b.12/M4b.15 precedent): Task 1 is a stream-rate gap
decomposition instrument with a pre-registered gate; lever tasks exist
only behind the gate, and at most one lever family is ever authorized.

## Motivation

At M4b.16 close, e2e decode tg vs llama.cpp best-of-builds stands at
**0.96x (16c 6336Y d2.c1.medium)** / **0.86x (8c E-2388G s2.c2.medium)**.
The decode wall is matmul-shaped — M4b.15's residual attribution: ffn
matmuls ~40% (16c) / 47.4% (8c) of the decode wall, lm_head 16.7% /
20.5%, attention 29.1% / 20.6% — and every dispatch-side and
attention-side lever is closed with recorded STOPs (M4b.10, M4b.12,
M4b.15, M4b.16).

The one unexplained number is the decode GEMV stream rate: **~41 GB/s
(16c) / ~40 GB/s (8c)**, against measured pure-stream ceilings of
**54.39 GB/s (16c) / 45.95 GB/s (8c)** (gate-bw-curve, recorded in the
M4b.11 spec). On the same 16c box, prefill matmul reaches **47.6 GB/s**
(M4b.13 gate-session profile). Arithmetic per box:

- **16c:** ~33% stream-rate headroom on the ~57% of the wall that is
  GEMV (ffn + lm_head), against a **4% tg gap**. Closing even a third of
  the headroom closes the gap. This is the deciding box.
- **8c:** ~15% headroom against a **16% gap** — llama is effectively at
  the stream ceiling there. Perfect streaming on the GEMV fraction
  projects roughly +10% e2e; the honest deliverable on this box is the
  measured ceiling, not a win claim.

Candidate causes for the gap, none yet attributed (that is Task 1's
job): page-walk/TLB cost — `weights.bin` is a plain `PROT_READ`/
`MAP_PRIVATE` 4 KiB-page mmap with no madvise (`inferno-core`
`artifact.rs`), so every decode token streams the full weight set
through 4 KiB TLB entries; per-thread kernel quality below the
shape-specific roofline (the GEMV kernels are the AVX2
`maddubs`+`madd` set — the AVX-512 VNNI path planned in M2 and gated
out of M4b.13 was never built, and `Isa::X86_64v4` still falls back to
AVX2); non-GEMV wall time diluting the effective rate; or a structural
shape tax (GEMV's read-once pattern may simply roofline below the
sequential-stream curve). Prefetch distance is NOT a candidate: the
M4b.7 quiet-hw sweep already fixed `INFERNO_PF_DIST` and found the
surface flat.

## Scope Decisions (M4b.17)

| Decision | Choice |
|---|---|
| Structure | **Attribution-first** (M4b.12/15 precedent): Task 1 = decomposition instrument + pre-registered gate; lever tasks exist only behind the gate; at most one lever family authorized |
| Exit criterion | **Split.** Hard target: e2e decode tg vs llama best-of-builds **≥ 1.0x on the 16c box**. On the 8c box the deliverable is the measured GEMV-shaped roofline and a recorded ceiling statement (whether any streaming lever can reach 1.0x). Sanctioned STOP-out: rule-3 all-STOP with both findings recorded (M4b.12 precedent) |
| Phase | **Decode GEMV only** (`m == 1` matmul path: ffn gate/up/down, lm_head, attn q/k/v/o projections). Prefill, attention (both phases), sampling, KV handling untouched |
| Dtype | **Q8_0 only** (the criterion model). Q4_K keeps its kernels and invariants |
| Instrument honesty | Every arm measures the **shipping kernel through the shipping dispatch** (registry-resolved `gemv_q8_0_rs8`, `inferno_par_gemv`, real packed weight buffers) — never a bench-local copy. A bench-local copy compiles const-geometry-specialized and its numbers do not transfer (M4b.15 inadmissibility finding, reproduced on both boxes) |
| Machines | 16c d2.c1.medium (6336Y) + 8c s2.c2.medium (E-2388G), M4a protocol, fresh llama best-of baselines for any e2e claim |
| Metal budget | Round 1 (attribution, both boxes) always; Round 2 (closing e2e A/B, both boxes) only if a lever is authorized and passes its dev-box local gate. STOP case: 2 provisions; worst case: 4 |

**Explicitly out of scope:** decode-attention anything (M4b.16
forecloses further micro-levers); prefill levers (M4b.13/14 findings
stand); F16 KV (M4b.11 STOP stands); quant-format or model changes;
thread-count, dispatch, or shard-granularity levers (M4b.10/12 findings
stand); NEON/AOT (v2).

## Task 1 — The Attribution Instrument

The instrument decomposes `ceiling − achieved` into named, separately
measured causes. Four arms, run on both boxes in one quiet-hw round:

1. **GEMV-shaped roofline arm** (bench crate, criterion): the shipping
   `gemv_q8_0_rs8` kernels dispatched through the shipping pool
   (`inferno_par_gemv`) at protocol thread counts, over the real packed
   weight buffers at the protocol shapes (896×896, 4864×896, 896×4864,
   151936×896) — measured as effective GB/s — next to a pure
   byte-stream loop over the same packed buffers at the same thread
   counts and shard boundaries (the gate-bw-curve anchor extended to
   the GEMV access pattern). The gap between the two is
   kernel/dispatch quality; the gap between the stream loop and the
   sequential-stream ceiling is the shape tax. Both gaps are recorded
   per shape.
2. **Page/TLB arm**: the same shipping kernel over the same bytes, A/B
   between (a) the artifact's 4 KiB-page mmap and (b) a one-time copy
   of the packed buffers into THP-backed anonymous memory
   (`madvise(MADV_HUGEPAGE)`). Any recovered bandwidth is directly
   attributed to page-walk cost. This arm doubles as the feasibility
   probe for Lever H. Both arms are warmed identically (one full pass
   before timing) — first-touch and page-cache state must not differ
   between arms.
3. **Counter lane** (script-level, `perf stat`, bare metal): DRAM read
   bandwidth (uncore), dTLB-load-misses, and top-level stall
   distribution for arms 1–2. Corroborating evidence only: rule 2's
   pressure test and rule 1's dTLB corroboration (§Risks) consult
   them, but no counter value is ever a gate quantity on its own.
4. **Idle-gap check**: per-token wall at best-t minus the sum of the
   matmul brackets (existing profiler brackets, M4b.9 splits) — bounds
   how much non-GEMV time dilutes the effective stream rate. M4b.12's
   sub-0.2% dispatch finding predicts this is small; verify, don't
   assume.

The instrument lives in the bench crate and `scripts/quiet-hw/` (a
`gate-gemv-stream.sh` session script following the existing gate-script
conventions: machine header, verbatim tables, human verdict line). Dev-box
runs are for development and smoke only; recorded numbers come from quiet
hardware.

## Pre-Registered Gate (arithmetic recorded once; 16c is the deciding box)

Let `G = roofline_arm − achieved` on the 16c box (per-shape,
profile-weighted the way M4b.6 weighted its projection).

- **Rule 1:** THP arm recovers **≥ G/2** → authorize **Lever H**
  (hugepage weight residency). Lever V stays unbuilt.
- **Rule 2** (else): the shipping kernel sits **≥ 5% below its own
  GEMV-shaped roofline** and the counter lane shows port/compute
  pressure rather than DRAM saturation → authorize **Lever V**
  (AVX-512 VNNI GEMV). Lever H stays unbuilt.
- **Rule 3** (else): achieved is within 5% of the GEMV-shaped roofline
  → **STOP.** The kernel is at its structural ceiling; record the
  finding, record the 8c ceiling statement, close as a diagnostic. No
  lever is built.

Exactly one lever family can be authorized. The gate verdict, with the
arithmetic shown, is recorded once as a spec amendment before any lever
task starts.

## Lever H — Hugepage Weight Residency (behind rule 1)

At artifact load, copy the weight payload from the 4 KiB mmap into
THP-backed anonymous memory (`madvise(MADV_HUGEPAGE)` on an anonymous
region; graceful fallback to the mmap path if THP is unavailable or the
copy fails). Controlled by `INFERNO_HUGEPAGE_WEIGHTS` with the default
set by the ship-gate outcome.

- **Bit-neutral by construction:** same bytes, same kernels, same
  dispatch, same shard ownership — residency is not a numeric change.
  No rig, tolerance, or differential implications.
- **Cache key untouched:** residency is a runtime property, not a
  codegen input.
- **Costs, stated:** one-time load copy (~0.6 GB for the criterion
  model) and the weights become anonymous RSS instead of evictable
  page cache. Documented in the CLI help if shipped.

## Lever V — AVX-512 VNNI GEMV (behind rule 2)

The never-built `Isa::X86_64v4` kernel set, GEMV-scoped: `vpdpbusd`
u8×i8→i32 dot at 512-bit width for `gemv_q8_0_rs8`. The registry
dispatch seam already exists (`Isa::X86_64v4` currently falls back to
AVX2); `inferno-target` already detects `Feature::Vnni`.

- **Bit-identity:** the integer dot is exact in both ISAs and the f32
  per-block accumulation chain keeps today's order, so scalar-vs-VNNI
  bit-identity is provable and joins the rig as a standing suite.
- **CI:** SDE lane for VNNI correctness tests (the M4b.13 Lever-2
  machinery, GEMV-scoped); native rig runs on quiet hw.
- **Local gate before metal spend** (M4b.13 precedent): criterion
  µbench on the four protocol shapes must beat the AVX2 kernel on the
  dev box, and a same-session before/after `bench-compiled` pair must
  improve, before Round 2 is provisioned.

## Ship Gate (pre-registered; fresh llama best-of baselines, M4a protocol, e2e decode tg at best-t)

- 16c reaches **tg ≥ 1.0x vs llama best-of-builds** → **ship**
  (lever default-on).
- 16c < 1.0x but the lever shows **≥ +3%** with non-overlapping CIs →
  **judgment rung**, decision recorded as an amendment with the
  arithmetic.
- Lever **< +3%** on 16c → **STOP**; lever stays default-off, shipped
  as an opt-in diagnostic (M4b.16 precedent).
- 8c constraint on any ship: no tg regression beyond CI overlap.

## Structure

1. **Task 1:** instrument (bench arms + session script + counter lane)
   — dev-box smoke, then Round 1 quiet-hw sessions (both boxes, serial
   provisions per the PNAP rule).
2. **Gate verdict amendment** (arithmetic shown once). Rule 3 → skip to
   task 4 with Round 1 as closing data.
3. **Lever task** (H or V, never both): implementation + local gate +
   Round 2 closing sessions + ship-gate verdict amendment.
4. **Close:** exit-criteria walk; 8c ceiling statement; M4a spec
   §Amendments protocol tables for every session; AGENTS.md decode
   paragraph updated; v1 context ratios recorded (never the gate).

## Testing & Standing Invariants (all unchanged)

- Scalar-vs-SIMD bit-identity per ISA (VNNI joins the rig if Lever V is
  built); `gemm(m=1)` bit-equals `gemv`; cross-thread bit-identity
  (`shard_table` row ownership untouched by every arm and lever).
- Codegen differential (`cargo test -p inferno-codegen --test
  differential`) and core artifact differential green; **zero tolerance
  edits** (`git diff main -- crates/inferno-graph/src/tolerance.rs`
  empty at close).
- `mise run test` and `mise run lint` green at every task boundary.
- Kernel perf numbers only from quiet hardware; every session's
  protocol tables recorded verbatim in the M4a spec §Amendments; no
  recorded data point is ever edited.
- Every STOP recorded as a finding; the gate and ship-gate arithmetic
  each recorded exactly once.

## Risks

- **Most likely outcome on current evidence:** rule-3 STOP on 8c and a
  genuine decision on 16c. If 16c also STOPs, the v1 tg criterion is
  formally at its measured ceiling — the milestone's deliverable is
  that statement, which forces the v1-reckoning conversation
  (criterion redefinition or acceptance) as its own next item. That is
  a valid close, not a failure.
- **THP A/B confounding:** page-cache and first-touch state can differ
  between arms on a warm box. The instrument warms both arms
  identically and the counter lane (dTLB misses) must corroborate any
  claimed THP recovery before rule 1 fires.
- **Apples-to-apples:** llama.cpp also runs from a 4 KiB mmap. A Lever
  H win is still a legitimate e2e win — the gate is tg vs llama's
  shipping default — but the finding must state the mechanism honestly
  (we bought the win with resident hugepages, not kernel quality).
- **Roofline-arm honesty:** the pure-stream loop must respect the same
  shard boundaries and thread counts as the shipping dispatch, or the
  roofline is fiction and the gate misfires. The session script
  cross-checks it against the recorded gate-bw-curve ceilings.

## Amendments

(recorded during execution; none yet)
