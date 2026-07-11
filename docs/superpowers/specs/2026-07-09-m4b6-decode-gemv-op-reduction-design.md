# M4b.6 — Decode GEMV Reduce/Combine Op-Reduction Design

**Date:** 2026-07-09
**Status:** Approved design, pre-implementation
**Milestone:** M4b.6 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.5](2026-07-08-m4b5-phase-aware-decode-threading-design.md))

This milestone takes up the gated follow-up recorded in
[M4b.4](2026-07-08-m4b4-decode-gemv-mlp-design.md) §Gated Follow-Ups:
**approach B — op-reduction of the per-block `hsum8_i32` cross-lane
reduction** in the Q8_0 GEMV kernel. M4b.4's Task 1 classified the AVX2
GEMV compute-bound on the one trustworthy DRAM-bound shape (baseline at
~90% of the box's sequential-read bandwidth ceiling) and attributed the
residual gap to the per-block reduce/combine, not unhidden memory
latency. That attribution has never been measured directly — it was
inferred from what the streaming ceiling *ruled out*. M4b.6 measures it
first and only then acts.

## Motivation — and an honest headroom warning

Decode is single-row GEMV over every weight matrix, once per token. In
the AVX2 full-strip fast path (`q8_0.rs`), each 8-row × 32-element weight
group costs roughly: 40 ops of int8 dot (`load`/`sign`×2/`maddubs`/`madd`
per lane), **9 ops of `hsum8_i32` transpose-reduce** (6× `vphaddd` + 2×
`vperm2i128` + add), and **~5 ops of f32 combine** (`cvt`, `dw` load,
`dx` broadcast, `mul`, `fmadd`). The reduce+combine is therefore ~26% of
the op count on ~0% of the data movement — the classic op-reduction
target.

The warning, stated plainly so no ledger entry over-claims:

- The recorded kernel benches (M2 §Amendments, M4b.2 §Amendments) show
  this kernel **already beats ggml's Q8_0 dot at every measured shape** —
  1.48–1.76× on the FFN shapes, 1.16× on the DRAM-bound lm_head shape.
- M4b.4 Task 1 put the baseline at **~90% of the streaming ceiling** on
  lm_head (151936×896), the shape that dominates decode bytes. In real
  decode every GEMV streams its weights from DRAM (model ≫ L3), so on
  DRAM-bound shapes the op-reduction lever is bounded by that residual
  ~10% regardless of how many ops it deletes.
- End-to-end tg vs llama.cpp t=1 is already 0.88×–1.09× (M4b.3), and
  M4b.5's directional sweep suggests capped multi-thread decode moves
  well past t=1 bandwidth. The tg win may not need this lever at all.

So M4b.6 is **diagnostic-gated**: Task 1 measures the reduce/combine's
true share of decode-shape wall time with a ceiling arm, projects the
maximum end-to-end decode win, and a STOP gate decides whether the
restructure happens. A clean STOP with a recorded finding is a
successful outcome (M4b.4 Task 1 precedent), not a failed milestone.

## Scope Decisions (M4b.6)

| Decision | Choice |
|---|---|
| Lever | Op-reduction of the per-block `hsum8_i32` + f32 combine in `inferno_gemv_q8_0_rs8_avx2`'s full-strip fast path (and its GEMM twin, in lockstep) |
| Structure | **Diagnostic-first.** Task 1 = reduction-ceiling bench arm + profile-weighted projection + STOP gate. Restructure tasks exist only behind the gate |
| STOP gate | Projected decode-wall reduction **≥5% → proceed**; **3–5% → controller judgment call**, recorded as a spec amendment either way; **<3% → STOP**, record the finding, close the milestone as a diagnostic |
| Numeric contract | **Contract change is in scope** (user decision, this design). The "f32 combine runs in block order" contract may be redefined if the winning candidate needs it, under the lockstep discipline below |
| Scalar reference | Redefined **in lockstep** with any contract change: scalar↔AVX2 **exact bit-identity is preserved under the new contract**, never relaxed to a tolerance |
| GEMM parity | `inferno_gemm_q8_0_rs8_{scalar,avx2}` change identically so `gemm(m=1) == gemv` stays **bitwise** — prefill numerics move with decode's, and one tolerance re-derivation covers both paths |
| Interpreter | **Untouched.** It remains the reference the compiled path is measured against |
| Tolerances | `gemv_rel_tol` / `logits_abs_tol` re-derived from the `observed_error_*` diagnostic sweeps (M4b.3 discipline) if the contract changes; **never loosened to green a red test**. Bit-neutral candidate → no tolerance touched |
| ABI | `HOST_ABI_VERSION` bump if kernel numerics change (stale cached `model.so` + new tolerances must not mix; M4b.3 precedent). Bit-neutral candidate → no bump |
| Activation layout | q8a layout may change (e.g. contiguous per-block `d`s) if the winning candidate needs it; `pack_q8_0_rs8` and the quantize path move together, fuzz targets updated with them |
| Docs | AGENTS.md's "f32 combine runs in block order" invariant text updated in the same change as any contract change |
| Measurement discipline | Same-box **baseline-vs-ceiling / baseline-vs-new ratios only**, interleaved A/B reps; absolute GB/s untrusted on the shared devpod; formal perf verdict deferred to quiet hardware (standing M4b discipline) |

**Explicitly out of scope:**

- **F16 KV cache** — unchanged gate; a different bytes lever, small at the
  pinned geometry.
- **Register-blocked GEMM** — the prefill lever, its own follow-up
  (gated in M4b.3's verdict).
- **AVX-512 / VNNI (`vpdpbusd`) paths** — a different ISA target (M2
  follow-up); noted only as a beneficiary of a simplified contract.
- **The quiet-hardware verification pass** (M4b.1 prefill-scaling gate,
  M4b.4 PF_DIST keep/revert, M4b.5 cap default, official bench protocol
  run) — the natural M4b.7; the STOP branch of this milestone points at
  it explicitly.
- **Other dtypes (Q4_K, F32 GEMV)** — mirroring a winning restructure is
  a follow-up, not silent scope growth here.

## Design

### Task 1 — Reduction-ceiling diagnostic (measurement, decides the milestone)

Modeled on M4b.4's weight-streaming ceiling arm (`b12264d`). Add a
bench-only **reduction-ceiling arm** to `benches/gemv.rs`: a copy of the
AVX2 full-strip kernel that keeps every weight/activation load and the
full int8 dot (`sign`×2 / `maddubs` / `madd` per lane) intact, but
replaces the per-block `hsum8_i32` + cvt/mul/fmadd combine with the
cheapest bit-sink the optimizer cannot delete — accumulate the eight
`p[]` vectors into a running i32 vector, fold it into a `black_box`'d
checksum once per strip. Wrong numbers by design; the right cost model
for "reduce+combine cost → 0". It lives in the bench crate next to the
ggml-compare arm and never ships in the library.

Measurement:

- **Shapes:** the pinned Qwen2.5-0.5B decode set — 896×896 (attn
  q/o_proj), 4864×896 / 896×4864 (FFN), 151936×896 (lm_head) — plus the
  two non-Qwen shapes for continuity with M4b.4's tables.
- **Signal:** per-shape baseline-vs-ceiling **time ratio**, interleaved
  A/B criterion runs on the same box. Absolute GB/s recorded but
  untrusted.
- **Projection:** weight each shape's ceiling headroom by its decode-wall
  share from `inferno run --profile` on the pinned model to project the
  **maximum end-to-end decode-wall reduction** if reduce+combine cost
  went to zero.

**Gate (recorded as an amendment either way):**

- Projected decode-wall reduction **≥5%** → proceed to Task 2.
- **3–5%** → controller judgment call, weighing which candidate the
  attribution points at (bit-neutral candidate 1 deserves a lower
  effective bar than contract-changing candidates 2–3, which carry the
  tolerance re-derivation cost).
- **<3%** → STOP. Record the per-shape table, the projection, and a
  ledger note that the decode GEMV inner loop is exhausted as a lever;
  the tg win effort moves to the quiet-hardware verification pass.

### Task 2+ — Restructure (gated; candidate picked by Task 1's attribution)

Task 1's per-shape data also attributes the headroom between the reduce
tree and the f32 combine chain (a second arm variant that keeps
`hsum8_i32` but stubs the combine separates the two if the first arm
shows real headroom). Candidates, ranked:

1. **Unpack/add transpose-reduce tree** (bit-neutral — integer reduction
   structure is explicitly unconstrained by the numeric contract, per the
   `hsum8_i32` doc comment). Replaces the 6× `vphaddd` tree with a
   `vpunpck`/`vpaddd` tree. Recorded caveat: on Zen 2 the µop count is a
   near-wash (~14 shuffle + 7 add either way); this candidate only wins
   if the measured bottleneck is `vphaddd`'s port placement, and it helps
   Intel more than AMD. Cheap to try, trivially safe.
2. **Block-pair combine merge** (contract change, small): reduce two
   blocks' dots, then one paired f32 combine — halves the
   cvt/mul/fmadd chain depth per byte. Changes the f32 accumulation
   order (pairwise instead of strictly per-block).
3. **Lane-deferred f32 accumulation** (contract change, ggml-shape):
   per-lane f32 partial accumulators, one cross-lane f32 reduce per row
   at the end of the k-loop, eliminating the per-block transpose-reduce
   entirely. Biggest ceiling, biggest numeric change. Recorded caveat:
   the strip layout already amortizes the reduce to ~9 ops per 8 rows'
   block dots, and a naive lane-deferred form *adds* per-lane scale
   broadcasts — this candidate must prove itself in the plan's µop
   analysis, not be assumed because llama.cpp uses the shape.

Whichever candidate ships follows the lockstep discipline from Scope
Decisions: scalar reference redefined identically (bit-identity under
the new contract), GEMM twins updated for bitwise `gemm(m=1) == gemv`,
interpreter untouched, tolerances re-derived from observed error,
`HOST_ABI_VERSION` bumped, AGENTS.md contract text updated.

## Correctness

- **Bit-identity lock set stays green under whichever contract ships:**
  `q8_0_isa_variants_bitwise_equal`, `q8_0_gemm_m1_equals_gemv`,
  `q8_0_gemv_matches_oracle` (oracle updated with the scalar reference
  if the contract changes — one edit, both variants must match it),
  `q8_0_range_partition_bitwise`, the pool's
  `*_thread_count_is_bit_invisible` / `q8_0_decode_cap_is_bit_invisible`
  locks.
- **Compiled-vs-interpreter:** differential + artifact suites green. Bit-
  neutral branch: no tolerance touched, provable by
  `git diff -- crates/inferno-graph/src/tolerance.rs` empty. Contract-
  change branch: `observed_error_*` sweeps re-run, new tolerances derived
  from the observed distributions and recorded (value + derivation) in
  this spec's Amendments — never widened ad hoc.
- **Fuzzing:** if `pack_q8_0_rs8` or the q8a layout changes, the
  corresponding fuzz targets are updated in the same task.
- Whole-workspace `mise run test` and `mise run lint` green before any
  task is called done.

## Measurement & Exit Criterion

Two-legged, with an explicit STOP branch:

- **Leg 1 — correctness (load-bearing, provable on the devpod):** the
  Correctness section above, in full.
- **Leg 2 — performance:** same-box interleaved baseline-vs-new kernel
  ratio on the decode shape set consistent with Task 1's projection,
  plus one directional end-to-end tg data point (`mise run bench`,
  recorded per protocol in the M4a spec's Amendments). The **formal perf
  verdict defers to quiet hardware**, same as every M4b verdict since
  M4b.1 — no absolute-number claim from the shared devpod.
- **STOP branch (Task 1 gate fails):** exit = the recorded diagnostic
  amendment. The milestone is complete when the finding (per-shape
  table + projection + "inner loop exhausted" ledger note) is committed;
  no code ships beyond the bench arm.

## Risks

- **Contention-confounded ratios on the shared devpod.** Mitigation:
  interleaved A/B reps, ratio-only reading, and the gate compares a
  projection against a bar rather than trusting any absolute number; a
  borderline reading lands in the 3–5% judgment band, recorded as an
  amendment.
- **The ceiling arm gets optimized away or de-optimized.** A deleted dot
  loop overstates headroom; a spilled checksum understates it.
  Mitigation: `black_box` checksum folded once per strip, and a sanity
  check that the arm's time is *below* baseline but *above* the M4b.4
  stream-read ceiling on the DRAM-bound shape.
- **Zen 2 shuffle-port parity kills candidate 1.** Anticipated in-spec;
  the plan's µop table decides before code is written.
- **Tolerance re-derivation has prefill blast radius** (GEMM moves with
  GEMV). Mitigation: the lockstep-scalar rule keeps compiled-vs-scalar
  exact; the differential suite already covers both prefill and decode
  paths against the untouched interpreter; tolerances derive from
  observed distributions with the derivation recorded.
- **Reading a STOP as failure.** It is the design working as intended:
  the milestone's product is either a faster kernel or a recorded proof
  that this lever is spent — both advance the v1 win criterion by
  telling M4b.7 where to spend effort.

## Gated Follow-Ups (not tasks in this milestone)

- **Quiet-hardware verification pass (natural M4b.7):** M4b.1
  prefill-scaling gate, M4b.4 PF_DIST keep/revert, M4b.5 cap default,
  and the official `inferno bench` protocol run — the v1 win criterion
  can only be judged there.
- **Mirroring a winning restructure into Q4_K / F32 GEMV.**
- **AVX-512 / VNNI dot path**, which collapses `maddubs`+`madd` into
  `vpdpbusd` and reduces the same way a simplified contract does.

## Amendments

*(Recorded protocol data points and scoped amendments land here; never
edit a recorded data point.)*

### 2026-07-09 — Task 1 diagnostic: reduce/combine ceiling — PROCEED

- **Commit:** 48da93c (`bench(kernels): reduce/combine ceiling arms for M4b.6 decode
  GEMV diagnostic`).
- **Command:** `devenv shell -- cargo bench -p inferno-kernels --bench gemv -- 'gemv/Q8_0'`,
  6 back-to-back reps — the plan's 3, plus 3 more after the ordering sanity gate fired and
  the arms were cleared by disassembly (below). Full outputs in scratch (`m4b6-rep{1..6}.out`).
- **Environment caveat:** shared 24-core devpod (AMD Ryzen 9 3900, 12C/24T); ratio-only
  reading per standing protocol; absolute times/GiB/s recorded but untrusted.

#### Per-shape baseline vs arms (medians across 6 reps)

Time columns are per-arm medians (context only); the ratio columns are **medians of the
per-rep ratios** (the signal — not ratios of the median times), with headroom_A's per-rep
min–max range in parens.

| shape | t_base | t_combine-stub | t_reduce-ceiling | t_stream-read | headroom_A | combine share | reduce share |
|---|---|---|---|---|---|---|---|
| 896×896 | 22.7 µs | 21.6 µs | 18.2 µs | 13.0 µs | **20.2%** (15.2–27.0) | 4.4% | 15.6% |
| 4864×896 | 129.2 µs | 124.4 µs | 102.9 µs | 77.5 µs | **20.8%** (18.9–22.8) | 6.6% | 14.4% |
| 896×4864 | 131.8 µs | 122.3 µs | 104.8 µs | 78.6 µs | **22.5%** (12.0–29.9) | 5.8% | 15.6% |
| 151936×896 | 16.93 ms | 17.88 ms | 17.68 ms | 10.52 ms | **−0.4% ≈ 0** (−12.0–2.9) | 1.5% | −0.8% |
| 4096×4096 | 2.09 ms | 1.94 ms | 1.96 ms | 1.31 ms | 3.8% (−9.3–21.2) | 5.0% | −2.5% |
| 14336×4096 | 6.61 ms | 7.06 ms | 6.32 ms | 4.10 ms | 5.5% (−50.4–8.2) | −1.0% | 2.3% |

Sanity orderings (plan Task 2 Step 4): the strict `A ≤ B ≤ base` ordering was violated on
≥1 shape in every rep, so the prescribed disassembly gate ran (`objdump -d` on the Task 1
bench binary, `gemv::reduce_ceiling::*` symbols): **both arms are structurally sound** —
full 8-lane dot present (8× `vpmaddubsw`/`vpmaddwd`, 16× `vpsignb`), `prefetcht0` exactly
PF_DIST=4 groups ahead, arm B's `hsum8_i32` intact (6× `vphaddd` + permute/insert), sink
register-resident with one store per strip, zero stack traffic. The violations are
noise-band sign flips where the true delta is ≈0 (lm_head A≈B≈base: the kernel is
memory-bound there, so stubbing the reduce/combine changes nothing) plus global noise
bursts in reps 3–4. The dot-deleted failure mode (`arm < stream-read`) never occurred on
any rep. `A < base` held 6/6 reps on all three mid Qwen shapes — the gate inputs are
noise-robust. Data accepted on that basis; dataset extended 3→6 reps for tighter medians.

#### Profile weighting (t=1 decode, pinned qwen2.5-0.5b Q8_0, 64 steps)

`inferno run --profile --threads 1`: decode 4.627 s wall, 14 243 765 734 cyc
(`m4b6-profile-t1.out`).

| slot | decode share | shape | headroom_A | product |
|---|---|---|---|---|
| matmul:lm_head.weight | 27.0% | 151936×896 | −0.35% | −0.09 pp |
| matmul:\*.ffn.down_proj.weight | 21.1% | 896×4864 | 22.52% | 4.75 pp |
| matmul:\*.ffn.up_proj.weight | 20.4% | 4864×896 | 20.76% | 4.24 pp |
| matmul:\*.ffn.gate_proj.weight | 20.3% | 4864×896 | 20.76% | 4.21 pp |
| matmul:\*.attn.q_proj.weight | 3.8% | 896×896 | 20.21% | 0.77 pp |
| matmul:\*.attn.o_proj.weight | 3.8% | 896×896 | 20.21% | 0.77 pp |
| matmul:\*.attn.v_proj.weight | 0.6% | 896×896 (**approx**; real 128×896, not in bench set) | 20.21% | 0.12 pp |
| matmul:\*.attn.k_proj.weight | 0.5% | 896×896 (**approx**; real 128×896, not in bench set) | 20.21% | 0.10 pp |
| non-matmul (swiglu, attention, rope, rmsnorm, add, bias, embed, quantize) | 2.6% | — | 0 | 0 |

**projected_decode_win = 14.9%** (combine-only: 4.7%, reduce-only: 10.3%).

#### Gate decision

≥5% → **PROCEED to the restructure plan.** The attribution is reduce-dominated (the
per-block `hsum8_i32` transpose-reduce costs ~2.2× the f32 combine on the Qwen mid
shapes), which per §Task 2+ puts **bit-neutral candidate 1** in play first — subject to
the recorded Zen 2 shuffle-port caveat, which the restructure plan's µop analysis must
resolve before code is written; candidates 2–3 remain fallbacks under the lockstep
contract-change discipline. Standing caveat: the headroom lives entirely in the
L2/L3-resident mid shapes; lm_head (27% of decode cycles) is memory-bound with ≈0
op-reduction headroom, so 14.9% is a same-box projection ceiling, not a promised win —
the formal perf verdict stays deferred to quiet hardware per §Measurement & Exit
Criterion.

### 2026-07-09 — Task 2+ restructure: candidate 1 measured, not shipped

- **Commits:** 092b191 (candidate arm, bitwise-checked), 4311ac0 (removal).
- **Candidate selection:** the restructure plan's µop table
  (docs/superpowers/plans/2026-07-09-m4b6-reduce-unpack-restructure.md
  §Decision Record, uops.info Zen 2 data) resolved the recorded Zen 2
  shuffle-port caveat FOR candidate 1: µop count is the anticipated wash
  (21 vs 21) but `vphaddd` is 3 µops at rt 2.0 vs the dword unpacks' 1 µop
  at rt 0.5, and the hadd tree's third µop class contends with the FP0-bound
  int8 dot. Candidates 2–3 rejected by the same table (combine share caps C2
  under the 3% bar; C3 is µop- and port-negative on Zen 2), not deferred.
- **Command:** 3 reps of `cargo bench -p inferno-kernels --bench gemv
  -- 'gemv/Q8_0/(inferno-avx2|reduce-unpack)/'` (devenv shell, shared devpod,
  ratio-only; criterion mid estimates; `w_r = 1 − t_unpack,r/t_base,r`,
  medians of per-rep ratios, never ratios of medians). Outputs in scratch
  (`m4b6r-rep{1..3}.out`, `m4b6r-table.md`).

| shape | t_base med | t_unpack med | median w | w range | all-reps w>0? |
|---|---|---|---|---|---|
| 896x896 | 21.39 µs | 21.44 µs | −0.26% | −8.26…+0.84% | no |
| 4864x896 | 116.80 µs | 116.45 µs | +1.22% | +0.14…+1.41% | YES |
| 896x4864 | 118.71 µs | 114.67 µs | +2.50% | +2.24…+3.40% | YES |
| 151936x896 | 15.228 ms | 17.625 ms | −16.47% | −18.06…−15.74% | no (all <0) |
| 4096x4096 | 1.7013 ms | 1.7507 ms | −5.93% | −6.01…−1.11% | no (all <0) |
| 14336x4096 | 6.3235 ms | 6.7529 ms | −6.05% | −6.79…−3.69% | no (all <0) |

- **projected_decode_win = −3.45%** (Task 1 amendment's decode shares — a
  projected decode-wall LOSS).
- **Ship gate:** NOT met. Condition 1 held (4864x896 and 896x4864 kept
  `w_r > 0` in 3/3 reps — the port model called the L2-resident mid shapes
  right, though at +1.2%/+2.5%, far under the ~15% reduce-share ceiling;
  note the arm carries a small favorable outer-loop/dispatch delta vs the
  registry-dispatched baseline — direct call, bare whole-strip loop — so
  treat those mid-shape figures as upper bounds if re-reading them for a
  future ship decision).
  Condition 2 failed decisively: median regressions of −16.47% on
  151936x896 (range −18.06…−15.74%, no 0-straddle → no rep extension per
  the gate's own rule), −6.05% on 14336x4096, −5.93% on 4096x4096 —
  consistent across all reps, far beyond the noise band. lm_head's 27%
  decode share alone swamps the mid-shape wins. Attribution (hypothesis,
  not measured): the µop table modeled issue-port pressure but not code
  footprint — the unpack tree is 21 instructions where the hadd tree is 9,
  and the fatter loop consistently loses exactly on the DRAM-/latency-bound
  streaming shapes, where issue ports were never the constraint. So the
  bottleneck candidate 1 attacks is real but small on this part, and the
  candidate carries a streaming-shape cost the model missed. With
  candidates 2–3 already µop-negative on Zen 2 (plan §Decision Record), the
  decode GEMV inner loop is exhausted as an op-reduction lever on this
  hardware. The tg win effort moves to the quiet-hardware verification pass
  (M4b.7), which should re-run this A/B (restore the arm by cherry-picking
  092b191) once on an Intel box before declaring the lever dead
  cross-vendor (SKL model says wash, not loss) —
  and should weight any future candidate by code size on streaming shapes,
  not just port placement.
- No library change shipped; tolerances/ABI untouched by construction
  (`git diff -- crates/inferno-graph/src/tolerance.rs` empty at every
  commit; the branch touches only `benches/gemv.rs`, net-zero, and this
  spec).

### 2026-07-11 — quiet-hw Intel A/B (M4b.7 gate-intel-ab, bare metal): SHIP — the op-reduction lever is REOPENED cross-vendor

Quiet bare metal via `mise run metal` (d2.c1.medium, Xeon Gold 6336Y, 16
physical / 32 logical, PREFLIGHT FIT), candidate = PR #11's reduce-unpack
("arm") kernel A/B'd against the library kernel at 6b0df49, reps=3,
bitwise pre-check green. Script verdict inputs: **condition 1 (w_r > 0
every rep on ≥2 of 3 mid shapes): 2 of 3 → MET; condition 2 (no shape
median w < −3%): PASS; projected_decode_win = 3.14%** (weights
.270/.211/.407/.087 per the earlier amendment). Per the recorded rule —
SHIP iff condition 1 MET and condition 2 PASS — **the verdict is SHIP:
the candidate that was NO-SHIP on the original measurement box wins on
Intel, so the lever is reopened cross-vendor.**

The script's straddling-rep WARNING (896x4864 has one negative rep, and
4096x4096 one at −4.08) is noted and did not force a --reps 6 re-run:
both straddlers already count against condition 1 — a re-run could only
move MET 2-of-3 to MET 3-of-3, never flip the verdict; condition 2
medians (1.35, 0.79) are clear of the −3% line.

Shipping is a scoped follow-up, not part of this pass: PR #11
deliberately touches only `benches/gemv.rs`, so SHIP here authorizes
porting the reduce-unpack kernel into the library GEMV (bit-identity
gates and the standing tolerances untouched, per this spec's rules) —
with the cross-vendor caveat that the win is Intel-measured; re-run the
A/B on the original vendor before making it the unconditional kernel.

```
# gate-intel-ab (M4b.6 reduce-unpack cross-vendor A/B) — 2026-07-11T13:15:57Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-11
From https://github.com/rahulmutt/inferno
 * branch            refs/pull/11/head -> FETCH_HEAD
Preparing worktree (detached HEAD 6b0df49)
bitwise pre-check (arm vs library kernel, --test mode)…

| shape | w per rep (%) | median w (%) | (w = 1 − t_unpack/t_base; positive = arm wins) |
|---|---|---|---|
| 896x896 | 2.60 6.93 0.96 | 2.6 | |
| 4864x896 | 3.37 1.33 3.13 | 3.13 | |
| 896x4864 | 1.35 -0.50 1.38 | 1.35 | |
| 151936x896 | 4.35 23.43 5.02 | 5.02 | |
| 4096x4096 | 0.86 -4.08 0.79 | 0.79 | |
| 14336x4096 | 3.69 32.22 4.75 | 4.75 | |

condition 1 (w_r>0 every rep on >=2 of 3 mid shapes): 2 of 3 -> MET
condition 2 (no shape median w < -3%): PASS
projected_decode_win = 3.14% (weights .270/.211/.407/.087 per M4b.6 amendment)
WARNING: a shape's w_r straddles 0 — if it is a deciding shape, re-run with --reps 6 before recording.
verdict (human, to M4b.6 Amendments): SHIP iff condition 1 MET and condition 2 PASS.
```

### 2026-07-11 — second quiet-hw session: SHIP does NOT reproduce — superseded to NO-SHIP

Same box type, same gate, same candidate (PR #11 reduce-unpack), library
baseline @ 1804d9f, hours after the SHIP session: **condition 1 now
FAILS (1 of 3 mid shapes, vs MET 2-of-3 this morning), projected win
2.01% (vs 3.14%), straddle warning again** — this time with negative
reps on 4864x896 (−3.97) and 896x4864 (−0.94) and every median under
the morning's. Two same-type quiet boxes disagreeing on the conditions
means the effect is within session-to-session noise, and the recorded
rule (SHIP iff condition 1 MET) fails on the repeat. **The morning's
SHIP verdict is withdrawn; the standing verdict is NO-SHIP — the
cross-vendor reopening did not survive replication.** A future
`--reps 6` run may re-open the question if it shows a stable
every-rep-positive win on the mid shapes; until then the op-reduction
lever stays closed on Intel as well, consistent with the original
NO-SHIP on the measurement box.

```
# gate-intel-ab (M4b.6 reduce-unpack cross-vendor A/B) — 2026-07-11T21:12:18Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-11
From https://github.com/rahulmutt/inferno
 * branch            refs/pull/11/head -> FETCH_HEAD
Preparing worktree (detached HEAD 1804d9f)
bitwise pre-check (arm vs library kernel, --test mode)…

| shape | w per rep (%) | median w (%) | (w = 1 − t_unpack/t_base; positive = arm wins) |
|---|---|---|---|
| 896x896 | 4.70 2.64 2.68 | 2.68 | |
| 4864x896 | 1.25 -3.97 3.44 | 1.25 | |
| 896x4864 | -0.94 1.45 1.53 | 1.45 | |
| 151936x896 | 3.72 3.55 3.22 | 3.55 | |
| 4096x4096 | 1.18 3.01 -0.29 | 1.18 | |
| 14336x4096 | 4.67 4.01 0.29 | 4.01 | |

condition 1 (w_r>0 every rep on >=2 of 3 mid shapes): 1 of 3 -> FAILED
condition 2 (no shape median w < -3%): PASS
projected_decode_win = 2.01% (weights .270/.211/.407/.087 per M4b.6 amendment)
WARNING: a shape's w_r straddles 0 — if it is a deciding shape, re-run with --reps 6 before recording.
verdict (human, to M4b.6 Amendments): SHIP iff condition 1 MET and condition 2 PASS.
```
