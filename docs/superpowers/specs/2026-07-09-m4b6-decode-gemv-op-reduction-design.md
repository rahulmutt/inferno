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
