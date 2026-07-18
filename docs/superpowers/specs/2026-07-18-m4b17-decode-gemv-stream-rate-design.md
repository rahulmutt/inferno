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

### 2026-07-18 — Round 1 Session A — d2.c1.medium (16c 6336Y, CHI, server 6a5b77b7, HEAD 315af9a)

Provision history: attempt 1 aborted mid-workload (gate script lacked
`mkdir -p "$OUT"` on the fresh clone — fixed in 315af9a; its partial
best-t table is superseded by this session and not a recorded data
point); attempt 2 PHX 406 no-stock (no server); attempt 3 (this one)
CHI, PREFLIGHT: FIT (psi_some_avg10=0.18, quota=unquota'd, tsc=ok).

```
```

#### gate-gemv-stream arm tables (verbatim)

```
gemv_stream: 24 layers + lm_head, 169 matrices, 530.0 MiB packed, Avx2, lanes=16, reps=5
thp arm: region 557842432 B, AnonHugePages 544768 kB (from /proc/self/smaps)

| arm | kernel | attn GB/s | ffn GB/s | lm_head GB/s | total GB/s | ms/token |
|---|---|---|---|---|---|---|
| heap | gemv | 30.65 | 45.85 | 40.60 | 42.46 | 13.09 |
| heap | stream | 31.60 | 46.76 | 42.31 | 43.63 | 12.74 |
| mmap4k | gemv | 31.14 | 40.10 | 41.25 | 39.39 | 14.11 |
| mmap4k | stream | 32.35 | 40.92 | 42.97 | 40.49 | 13.72 |
| thp | gemv | 29.41 | 40.10 | 41.25 | 39.13 | 14.20 |
| thp | stream | 30.52 | 41.10 | 43.97 | 40.57 | 13.70 |

gate input (human, to the M4b.17 spec §Amendments): roofline = stream
rows; page/TLB recovery = thp vs mmap4k on the gemv rows; the heap row
is the recorded-ceiling condition (bw_curve used heap buffers).

gemv_stream: 24 layers + lm_head, 169 matrices, 530.0 MiB packed, Avx2, lanes=1, reps=5
thp arm: region 557842432 B, AnonHugePages 544768 kB (from /proc/self/smaps)

| arm | kernel | attn GB/s | ffn GB/s | lm_head GB/s | total GB/s | ms/token |
|---|---|---|---|---|---|---|
| heap | gemv | 13.46 | 13.55 | 13.69 | 13.58 | 40.92 |
| heap | stream | 14.46 | 14.49 | 14.53 | 14.50 | 38.33 |
| mmap4k | gemv | 14.03 | 14.11 | 14.13 | 14.11 | 39.40 |
| mmap4k | stream | 14.45 | 14.48 | 14.56 | 14.50 | 38.33 |
| thp | gemv | 13.81 | 13.87 | 13.68 | 13.81 | 40.23 |
| thp | stream | 14.46 | 14.54 | 14.43 | 14.50 | 38.33 |

gate input (human, to the M4b.17 spec §Amendments): roofline = stream
rows; page/TLB recovery = thp vs mmap4k on the gemv rows; the heap row
is the recorded-ceiling condition (bw_curve used heap buffers).
```

#### Counter lane

DEVIATION (recorded per script): `perf` unavailable on the box
(`apt-get install linux-perf` → "Unable to locate package"; Debian
bookworm image, kernel 6.9.10+bpo). Counter lane skipped; dTLB
corroboration for any rule-1 claim is therefore UNAVAILABLE this
session. AnonHugePages lines above stand in as THP-backing evidence
(544768 kB ≈ full 532 MiB region on both prints).

#### gate-decode-attr profiles (verbatim)

```
profile [prefill] 33.485s wall, 103799490602 cyc total
  op                                   cycles   share        GB/s
  attention                       24749777312   23.8%           -
  matmul:lm_head.weight           19809879010   19.1%        48.6
  matmul:layers.*.ffn.gate_proj.weight    15170851742   14.6%        48.8
  matmul:layers.*.ffn.up_proj.weight    15156345206   14.6%        48.8
  matmul:layers.*.ffn.down_proj.weight    15077634818   14.5%        49.1
  swiglu                           4270513364    4.1%           -
  matmul:layers.*.attn.q_proj.weight     2788990486    2.7%        48.9
  matmul:layers.*.attn.o_proj.weight     2787297778    2.7%        48.9
  rmsnorm                          1191908922    1.1%           -
  rope                              863607742    0.8%           -
  quantize                          530197490    0.5%           -
  matmul:layers.*.attn.v_proj.weight      401238574    0.4%        48.5
  matmul:layers.*.attn.k_proj.weight      401220408    0.4%        48.5
  add                               330611324    0.3%           -
  bias                              136156962    0.1%           -
  kv_append                         114618190    0.1%           -
  embed                              18641274    0.0%           -
profile [decode] 3.989s wall, 12267378900 cyc total
  op                                   cycles   share        GB/s
  attention                        4182726052   34.1%           -
  matmul:lm_head.weight            2145973644   17.5%        14.0
  matmul:layers.*.ffn.down_proj.weight     1664335614   13.6%        13.9
  matmul:layers.*.ffn.gate_proj.weight     1660712500   13.5%        13.9
  matmul:layers.*.ffn.up_proj.weight     1655898434   13.5%        14.0
  matmul:layers.*.attn.o_proj.weight      312484428    2.5%        13.7
  matmul:layers.*.attn.q_proj.weight      309334502    2.5%        13.8
  swiglu                            136336992    1.1%           -
  matmul:layers.*.attn.v_proj.weight       45285424    0.4%        13.5
  matmul:layers.*.attn.k_proj.weight       44830436    0.4%        13.6
  rope                               39679188    0.3%           -
  rmsnorm                            39303576    0.3%           -
  add                                17951366    0.1%           -
  bias                               11673132    0.1%           -
  embed                                449552    0.0%           -
  quantize                             404060    0.0%           -
  kv_append                                 0    0.0%           -

profile [prefill] 2.952s wall, 9147747668 cyc total
  op                                   cycles   share        GB/s
  attention                        2298586842   25.1%           -
  matmul:lm_head.weight            1497737146   16.4%       643.0
  matmul:layers.*.ffn.up_proj.weight     1271582200   13.9%       581.9
  matmul:layers.*.ffn.gate_proj.weight     1271161036   13.9%       582.1
  matmul:layers.*.ffn.down_proj.weight     1263327942   13.8%       585.7
  swiglu                            358566700    3.9%           -
  matmul:layers.*.attn.o_proj.weight      290397760    3.2%       469.3
  matmul:layers.*.attn.q_proj.weight      272665196    3.0%       499.9
  quantize                          159274300    1.7%           -
  rmsnorm                           109719228    1.2%           -
  rope                               88087528    1.0%           -
  matmul:layers.*.attn.k_proj.weight       62712606    0.7%       310.5
  add                                62110802    0.7%           -
  matmul:layers.*.attn.v_proj.weight       59501456    0.7%       327.2
  bias                               53092570    0.6%           -
  kv_append                          19649326    0.2%           -
  embed                               9575030    0.1%           -
profile [decode] 1.419s wall, 4298246334 cyc total
  op                                   cycles   share        GB/s
  attention                        1388604342   32.3%           -
  matmul:lm_head.weight             678395804   15.8%        43.8
  matmul:layers.*.ffn.gate_proj.weight      546999134   12.7%        41.7
  matmul:layers.*.ffn.down_proj.weight      546566994   12.7%        41.7
  matmul:layers.*.ffn.up_proj.weight      546554320   12.7%        41.7
  swiglu                            140859220    3.3%           -
  matmul:layers.*.attn.q_proj.weight      137392212    3.2%        30.6
  matmul:layers.*.attn.o_proj.weight      124993502    2.9%        33.6
  rope                               45634558    1.1%           -
  rmsnorm                            42774558    1.0%           -
  matmul:layers.*.attn.v_proj.weight       32623316    0.8%        18.4
  matmul:layers.*.attn.k_proj.weight       32176524    0.7%        18.7
  add                                19028820    0.4%           -
  bias                               14807376    0.3%           -
  embed                                478572    0.0%           -
  quantize                             357082    0.0%           -
  kv_append                                 0    0.0%           -
```

#### Session A gate-input quantities (human-computed; rule walk deferred to the gate-verdict amendment)

- **Achieved in-loop per-class GB/s** (decode profile, t=16): lm_head
  **43.8**; ffn gate/down/up **41.7 / 41.7 / 41.7**; attn projections
  q **30.6** / o **33.6** / v **18.4** / k **18.7** (attn-projection
  classes are 8.8% of streamed bytes; the low kv rates are 128-row
  matrices, dispatch-overhead-bound, share 0.8%).
- **Profile-weighted achieved GEMV rate**: decode t=16 wall = 1.419 s /
  64 tokens = **22.17 ms/token**; matmul bracket cycles 2,645,701,806 of
  4,298,246,334 total = **61.55%** → **13.65 ms/token** in GEMV. 530.0
  MiB packed per token / 13.65 ms = **40.7 GB/s achieved**.
- **GEMV-shaped roofline** (stream rows, best-t): shipping condition
  (mmap4k) **40.49 GB/s** total (attn 32.35 / ffn 40.92 / lm_head
  42.97); heap **43.63**; thp **40.57**.
- **Per-shape-class gaps (roofline − achieved)**: ffn 40.92 − 41.7 =
  **−0.8**; lm_head 42.97 − 43.8 = **−0.8**; i.e. the shipping loop's
  big GEMV classes run AT the GEMV-shaped stream roofline (negative gap
  = within measurement noise of zero). Total-rate gap
  G = 40.49 − 40.7 = **−0.2 GB/s ≈ 0**.
- **THP recovery** (gemv rows, thp − mmap4k): 39.13 − 39.39 =
  **−0.26 GB/s** (nil), with AnonHugePages 544768 kB confirming the thp
  arm really was hugepage-backed. Page/TLB cost is NOT where the
  headroom is.
- **Shape tax** (recorded-ceiling condition vs GEMV shape): heap stream
  43.63 at lanes=16 vs the M4b.10-recorded sequential-stream ceiling
  54.39 (lanes=8, same box class) → **10.76 GB/s (19.8%)** is the
  GEMV access-pattern tax; file-backed mmap costs a further 43.63 −
  40.49 = **3.14 GB/s (7.2%)** vs heap on the stream rows (present in
  the thp arm too — it is not a 4KiB-page effect).
- **Idle-gap (arm 4)**: 22.17 ms wall − 13.65 ms matmul = **8.52 ms**,
  fully accounted by bracketed non-GEMV ops (attention 32.3% = 7.16 ms;
  swiglu/rope/rmsnorm/add/bias ≈ 6.1% = 1.35 ms); unbracketed residual
  ≈ 0.05% — M4b.12's sub-0.2% dispatch finding re-confirmed. Non-GEMV
  dilution is real but attributed; it does not contaminate the per-class
  achieved rates above.
- **Instrument-vs-loop cross-check**: instrument mmap4k gemv row 14.11
  ms/token vs in-loop matmul 13.65 ms/token (3.4% apart) — the
  instrument reproduces the shipping decode matmul wall.
- **dTLB corroboration**: unavailable (counter-lane deviation above).
  Note rule 1 cannot fire without it per §Risks; moot if THP recovery
  stays nil.
