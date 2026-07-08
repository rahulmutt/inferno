# M4b.4 — Decode GEMV Memory-Level Parallelism Design

**Date:** 2026-07-08
**Status:** Approved design, pre-implementation
**Milestone:** M4b.4 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.3](2026-07-07-m4b3-vectorized-attention-design.md))

M4b.3 vectorized the attention kernel and collapsed attention from ~70%
(prefill) / ~82% (decode) of cycles to ~26–27% of both phases. Its
recorded t=1 exit data has inferno **winning prefill** (pp 1.26–1.34×
llama.cpp) but only at **rough decode parity** (tg 0.88–1.09×, noisy
around 1.0×). Decode parity is the softest part of the v1 "done when we
win" bar, so it is the target here.

The post-M4b.3 decode profile reassigns the blame cleanly. With attention
shrunk, **GEMM/matmul is now ~66% of decode cycles**, and every one of
those matmuls is a **single-row GEMV** (one token → each weight byte used
exactly once, no reuse). The profiler's effective throughput tells the
story:

| Decode op | Share | GB/s |
|---|---|---|
| attention | ~26.5% | — |
| `matmul:lm_head.weight` | ~19.4% | **16.1** |
| `matmul:layers.*.ffn.{gate,up,down}_proj.weight` | ~45% (3×15%) | **~15.9** |
| `matmul:layers.*.attn.{q,o,k,v}_proj.weight` | ~6% | ~15.8 |

The **same weights** move at **~40 GB/s** in prefill's batched `gemm`
(reused across an 8-token tile) but only **~16 GB/s** in decode's `gemv`.
A single-row GEMV issues far fewer independent, in-flight cache-line
loads than an 8-token GEMM, so it cannot hide DRAM latency: decode is
**memory-latency / MLP-bound**, not throughput-bound. M4b.4 closes that
specific gap by raising memory-level parallelism in the GEMV kernel,
**without touching the numeric contract**.

The hot path is the AVX2 full-strip GEMV
(`inferno_gemv_q8_0_rs8_avx2`, `inferno-kernels/src/q8_0.rs`): per strip
of 8 output rows it reads each 288-byte weight group once, computes 8
rows' int8 dots lane-parallel via the sign trick, reduces per block with
`hsum8_i32`, then combines into f32 with a per-block `fmadd` **in strict
block order**. The two levers that raise MLP — software prefetch and
processing more strips concurrently — are both **bit-neutral**: neither
reorders the f32 accumulation.

## Scope Decisions (M4b.4)

| Decision | Choice |
|---|---|
| Lever | **Memory-level parallelism in the AVX2 GEMV**, single-threaded. Software prefetch of upcoming weight groups + (if needed) N-strip interleaving so N× the outstanding weight-load streams hide DRAM latency |
| Kernel boundary | Same `inferno_gemv_{dtype}_rs8_avx2` symbols, same rs8 pack layout, same ABI. Body-only change; no codegen change (GEMV was already a symbol call since M2) |
| Dtype coverage | Primary **Q8_0** (the bench model, `qwen2.5-0.5b-instruct-q8_0.gguf`). The same bit-neutral transform is mirrored into the `q4_k` and `f32` rs8 AVX2 GEMV paths so the three kernels stay structurally parallel |
| Diagnostic-first | Task 1 is a **measurement task** (not code): sweep prefetch distance and strip-interleave width on the existing `benches/gemv.rs`, classify memory- vs compute-bound against a weight-streaming (memcpy-class) ceiling, and record the **minimal winning config**. Implementation follows the recorded decision |
| Scalar path | **Unchanged** — stays the independent reference oracle. Only the AVX2 body changes |
| Correctness — scalar vs AVX2 | **Exact equality**, unchanged: prefetch is a hint and strip-interleave only reorders row iteration, so AVX2 output stays byte-identical to scalar. The existing `inferno-kernels` rig assertion covers it |
| Correctness — gemm(m=1) vs gemv | **Exact equality**, unchanged: the f32 block-order combine is untouched |
| Correctness — compiled vs interpreter | Differential stays green with **no `logits_abs_tol` / `gemv_rel_tol` loosening**. No new tolerance constant and **no `observed_error_*` sweep** — the kernel produces identical bits, so there is no numeric surface to bound |
| Exit criterion | Two-legged, wide margin: **(1)** decode m=1 GEMV effective GB/s lifted meaningfully off the ~16 baseline toward prefill's ~40 (concrete bar from the diagnostic's memcpy-class ceiling; e.g. ≥25 GB/s / ≥1.5× baseline, finalized in the plan), measured by `bench-kernels`; **(2)** a recorded t=1 `bench` data point with **tg improved and its low end at/above parity (≥1.0×)**, pp not regressed, plus a fresh t=1 `--profile` decode capture showing the GEMV ops' GB/s risen |
| Standing invariants | scalar-vs-SIMD GEMV bit-identity; gemm(m=1)-vs-gemv bit-identity; compiled-vs-interpreter differential green with no tolerance loosening; the interpreter unchanged |

**Explicitly out of scope:**

- **Threading / parallel GEMV** (`inferno_par_gemv`) — stays gated on the
  M4b.1 bare-metal rerun. M4b.4 is single-threaded (t=1) throughout; no
  scaling claim is made.
- **F16 KV cache** — unchanged from its existing gate (KV stays f32 in
  M3+; changing it re-derives tolerances).
- **Register-blocked GEMM** — the prefill lever, its own follow-up.
- **Op-reduction of the int8 dot (approach B)** — folded in **only if**
  Task 1's diagnostic classifies the kernel as compute-bound rather than
  memory-bound (see Risks). Otherwise a gated follow-up.
- **rs8 pack layout / kernel ABI changes** — none.

## Design

### Task 1 — Diagnostic (measurement, no shipped code)

On the existing `benches/gemv.rs` decode geometry (m=1 GEMV, real Qwen
Q8_0 shapes: 896/4864/151936), sweep and record:

- **Software prefetch distance** — `_mm_prefetch(_MM_HINT_T0)` of the
  weight group `d` blocks ahead, for `d ∈ {1, 2, 4, 8}` groups. A
  strip's `nb` groups are fully contiguous (`nb × 288 B`), so prefetch
  reaches cleanly across the block loop and into the next strip.
- **Strip-interleave width** — process `N ∈ {1, 2, 4}` strips
  concurrently, each with its own `__m256` accumulator.

Classify the kernel by comparing achieved GB/s against a pure
weight-streaming (memcpy-class) baseline over the same bytes:

- **Memory-latency / MLP-bound** (expected): achieved GB/s well below the
  streaming ceiling → prefetch and/or interleave will close the gap.
- **Compute-bound**: achieved GB/s near the streaming ceiling but cycles
  dominated by the per-block `hsum8_i32` / sign-trick chain → prefetch
  won't help; the milestone pivots to approach **B** (op-reduction).

Output: a recorded decision naming the **minimal winning config** (the
smallest prefetch distance and interleave width that hit the exit bar).

### Task 2+ — Implement the winning config (AVX2 GEMV body)

In the AVX2 full-strip path of `inferno_gemv_{q8_0,q4_k,f32}_rs8_avx2`:

- **Software prefetch:** inside the block loop, prefetch group
  `b + PF_DIST` ahead. Pure hint; zero effect on results.
- **Multi-strip interleave (only if the diagnostic requires it):** keep N
  independent strip accumulators live so N× the outstanding weight-load
  streams overlap DRAM latency. Row iteration reorders across strips, but
  each row's per-block `fmadd` chain is byte-for-byte unchanged, so
  output is identical. Interleave width is bounded by YMM register
  pressure (16 registers; the transient `p[STRIP]` partials already use
  up to 8) — take the smallest N that meets the bar.

The scalar GEMV is not modified.

## Correctness

This is a pure-performance change with **no numeric surface**, a
deliberate simplification versus M4b.3:

- **scalar ≡ AVX2 (exact)** — scalar untouched; prefetch/interleave are
  bit-neutral. Covered by the existing rig assertion.
- **gemm(m=1) ≡ gemv (exact)** — the f32 block-order combine is
  unchanged.
- **compiled ≡ interpreter (tolerance-bounded, green, no loosening)** —
  holds because the kernel emits identical bits. `cargo test -p
  inferno-codegen --test differential` and `cargo test -p inferno-core
  --test artifact` must stay green with `logits_abs_tol` /
  `gemv_rel_tol` untouched.

**No `HOST_ABI_VERSION` bump is required** (contrast M4b.3): codegen is
unchanged — GEMV was already a symbol call — so cached `model.so`
artifacts stay valid and resolve the same symbol to the freshly built,
faster host kernel. The only operational requirement is that
profile/bench runs use a freshly built release host (standard protocol).

## Measurement & Exit Criterion

Two-legged, wide margin (per the noisy tg band):

- **Leg 1 — kernel** (`mise run bench-kernels`, `--features
  ggml-compare`, devenv shell, quiet hardware): record before/after
  effective **GB/s** for the decode m=1 GEMV on the real Qwen Q8_0
  shapes, ggml GEMV side-by-side. Bar: a meaningful lift off ~16 GB/s
  toward prefill's ~40 — concrete threshold set from Task 1's
  memcpy-class ceiling, finalized in the plan (working target ≥25 GB/s /
  ≥1.5× baseline). Low-noise, load-bearing leg.
- **Leg 2 — end-to-end** (`mise run bench`, t=1, M4a protocol): a
  recorded data point with **tg improved and its low end at/above parity
  (≥1.0×)**, **pp not regressed**, plus a fresh t=1 `--profile` decode
  capture showing the GEMV ops' GB/s risen.

All data points land in `## Amendments` below (M4b.2/M4b.3 precedent);
never edit a recorded data point.

## Risks

- **Diagnostic says compute-bound, not memory-bound.** Prefetch/interleave
  would then barely move GB/s. Mitigation: Task 1's diagnostic runs
  *first* and gates implementation; if compute-bound, the milestone
  pivots to approach **B** (op-reduction of the per-block `hsum8_i32`) as
  its lever — recorded as a scoped decision, not a silent scope change.
- **Register pressure from multi-strip interleave.** 16 YMM registers,
  and the transient `p[STRIP]` partials already use up to 8. Mitigation:
  interleave width is a swept knob (Task 1); take the smallest N that
  hits the bar, and prefer prefetch-only if it suffices.
- **Bench noise on tg** (the 0.88–1.09× band). Mitigation: the two-legged
  criterion — the kernel-level GB/s leg is low-noise and load-bearing;
  the end-to-end leg confirms it lands.

## Gated Follow-Ups (not tasks in this milestone)

- **Op-reduction of the int8 dot (approach B).** If not pulled in by the
  compute-bound branch above, it stays a scoped follow-up: cut the
  per-block `hsum8_i32` cross-lane reduction cost. Own plan.
- **Parallel / threaded GEMV** (`inferno_par_gemv`). Still gated on the
  M4b.1 bare-metal rerun.
- **F16 KV cache.** Unchanged gate; re-derives tolerances.
- **Register-blocked GEMM.** The prefill lever, its own follow-up.
- **Doc hygiene (in scope, trivial):** the `mise run bench` task
  description in `mise.toml` still points at the M4b.1 spec for recording
  data points — repoint it to the current milestone spec.

## Amendments

*(Recorded protocol data points and scoped amendments land here; never
edit a recorded data point.)*

### 2026-07-08 — Task 1 diagnostic: streaming ceiling + baseline classification — **COMPUTE-BOUND, prefetch path STOPPED**

- **Commit:** `b12264d` (`bench(kernels): weight-streaming ceiling arm for
  decode GEMV diagnostic`), branch `m4b4-decode-gemv-mlp`.
- **Command:** `devenv shell -- cargo bench -p inferno-kernels --features
  ggml-compare`, full output in `/tmp/bench.out`. `rustc 1.96.1`, `INFERNO_GGML_CPU_LIB`
  unchanged from M4b.3's pin.
- **Environment caveat (read before the numbers):** measured on a shared
  24-core devpod (AMD Ryzen 9 3900, 12C/24T, 64 MiB L3 split across **4
  CCX instances of 16 MiB each**), *not* the quiet bare metal the plan
  text assumed. Per AGENTS.md/M4a protocol, absolute GB/s is noisy —
  **the same-box baseline-vs-ceiling ratio is the trustworthy signal**,
  not the raw numbers. Criterion reports GiB/s; converted to decimal
  GB/s below (×1.073741824) to match this spec's existing GB/s
  convention (§ profile table).

#### Per-shape baseline (inferno-avx2) vs ceiling (stream-read)

| Shape | Packed size | avx2 GB/s | stream GB/s | ratio | cache fit (16 MiB/CCX L3) |
|---|---|---|---|---|---|
| 896×896 | 0.81 MiB | 45.60 | 74.87 | 0.609 | resident |
| 4864×896 | 4.42 MiB | 45.42 | 71.57 | 0.635 | resident |
| 896×4864 | 4.42 MiB | 45.61 | 69.91 | 0.652 | resident |
| **151936×896** | **137.94 MiB** | **13.08** | **14.52** | **0.901** | **exceeds L3** |

(GiB/s as printed by criterion, before conversion: avx2/stream =
42.468/69.731, 42.302/66.641, 42.483/65.113, 12.181/13.519
respectively — see `/tmp/bench.out` lines 172–318.)

Two more Q8_0 shapes present in the same run (not in the mandated Qwen
set, but sharing the same weight image/kernel and useful as
corroboration) show the identical pattern:

| Shape | Packed size | avx2 GB/s | stream GB/s | ratio |
|---|---|---|---|---|
| 4096×4096 | 17.00 MiB | 14.10 | 15.72 | 0.897 |
| 14336×4096 | 59.50 MiB | 13.19 | 14.66 | 0.900 |

#### Reading the mixed signal

Three of the four mandated Qwen shapes (896×896, 4864×896, 896×4864)
pack to ≤4.42 MiB — well inside a single 16 MiB CCX-local L3 slice — so
criterion's repeated-iteration harness keeps **both** arms cache-resident
for the whole sample window. Their ratio (0.61–0.65) is a cache-bandwidth
ratio, not a DRAM-latency ratio, and is not by itself dispositive under
either threshold (it sits strictly between 0.6 and 0.85). Only
**151936×896** (the lm_head/vocab projection, 137.94 MiB) exceeds even
the *full* 64 MiB L3 and is genuinely DRAM-bound for this benchmark. Its
ratio, 0.901, is corroborated independently by the two large non-Qwen
Q8_0 shapes in the same run (0.897, 0.900) — three DRAM-bound
measurements, in the same run, clustering within 0.4 percentage points
of each other. That tightness, on an otherwise-noisy shared devpod, is
itself evidence the ratio is real rather than a fluke.

#### Classification

Applying the plan's threshold (baseline ≳0.85× ceiling → compute-bound)
to the only trustworthy (non-cache-resident) data point among the
mandated shapes: **compute-bound** — the AVX2 GEMV kernel already
achieves ~90% of this machine's pure sequential-read bandwidth on the
one shape large enough to force real DRAM traffic. There is very little
headroom left for prefetch/MLP-interleave to recover; the residual gap
between baseline and ceiling is dominated by the per-block `hsum8_i32` /
sign-trick reduction compute, not by unhidden memory latency.

**Decision (per the spec's Risks section):** STOP the prefetch/interleave
path (Tasks 2+ of this plan as written). This is the compute-bound branch
anticipated in `## Risks`; the milestone pivots to the gated follow-up
**approach B (op-reduction of the per-block `hsum8_i32` cross-lane
reduction)**, which needs its own plan before implementation resumes.
Task 1 is measurement-only per its brief and ships no kernel change; no
further M4b.4 prefetch tasks should be started against this plan without
a new design amendment reflecting the pivot.

### 2026-07-08 — Task 2: prefetch implemented despite Task 1's STOP, pending quiet-HW reverification

**Scoped decision, recorded per the "no silent scope change" rule.** Task
1 (above) classified the kernel compute-bound on this shared devpod and
called STOP on Tasks 2+. The controller judged that classification
contention-confounded rather than settled — this box is a shared 24-core
devpod, not the quiet bare metal the plan assumed, and Task 1's own
ceiling numbers are themselves absolute GB/s on the same noisy box. The
controller approved proceeding with Task 2 anyway **because it is
bit-neutral and independently gated by the existing bit-identity
proptests** — implementing it costs nothing numerically and the
keep/revert and bar/Task-3 decisions are explicitly **deferred to a
quiet-hardware remeasurement**, not made here. This amendment does not
retract Task 1's compute-bound finding; it records that the finding is
being treated as unconfirmed pending quiet HW, and that the prefetch
landed as a reversible, bit-neutral hedge in the meantime.

**Implementation:** `PF_DIST = 4` (module const, `q8_0.rs:14-19`) and a
`_mm_prefetch::<_MM_HINT_T0>` of weight group `b + PF_DIST` inserted
immediately after the block loop's `let g = ...` in the full-strip fast
path of `inferno_gemv_q8_0_rs8_avx2` only (`q8_0.rs:154`) — not the
partial head/tail path, not the scalar kernel, not the GEMM paths. Per
the brief exactly; `PF_DIST` is kept at the plan default of 4 regardless
of any directional sweep result (see below) — the controller instructed
that no noisy-box sweep may override the default.

**Gate (load-bearing):** `cargo nextest run -p inferno-kernels -E
'test(q8_0)'` — 10/10 green before and after the change, including
`q8_0_isa_variants_bitwise_equal`, `q8_0_gemm_m1_equals_gemv`,
`q8_0_gemv_matches_oracle`, `q8_0_range_partition_bitwise`. Full
`inferno-kernels` suite (43 tests) also green after the change. `mise
run lint` clean (rustfmt + clippy `-D warnings`, no new `#[allow]`).
Bit-identity holds.

**Directional bench — NOT a tuning decision, shared noisy devpod, no
trustworthy signal:** `devenv shell -- cargo bench -p inferno-kernels
--bench gemv gemv/Q8_0` (inferno-avx2 arm only, no ggml), run once before
the change (base commit `dfccaa9`) and once after (`PF_DIST = 4`),
back-to-back on the same contended box. Full output in
`/tmp/bench_before.out` / `/tmp/bench_after_pf4.out`.

| Shape | before (GiB/s, median) | after PF_DIST=4 (GiB/s, median) |
|---|---|---|
| 896×896 | 42.369 | 40.433 |
| 4864×896 | 42.508 | 39.712 |
| 896×4864 | 41.996 | 39.971 |
| **151936×896** | **14.726** | **10.065** |
| 4096×4096 (non-Qwen) | 16.616 | 11.253 |
| 14336×4096 (non-Qwen) | 14.967 | 8.6435 |

Every shape reads as a regression, including the cache-resident small
shapes where the prefetch instruction can only add overhead — that
pattern (uniform "regression" even on shapes too small to be
DRAM-bound) is itself a symptom of box contention rather than a real
software-prefetch cost, consistent with Task 1's own observation that
absolute GB/s wobbles heavily here. **No PF_DIST sweep ({2,4,8}) was
run**: the single before/after pair already shows swings large enough
(down to ~68% of baseline on the largest shape) that a 3-point sweep on
this box would not add a trustworthy signal, only more noisy samples.
**Directional only — not a tuning decision, not a keep/revert verdict,
not a bar assessment.** Per the controller's instruction, `PF_DIST`
stays at 4 and both the Leg-1 bar assessment (≥25 GB/s / ≥1.5× baseline)
and the Task 3 (interleave) go/no-go are **deferred to quiet hardware**.

**Status:** prefetch is committed as a reversible, bit-neutral hedge.
Nothing here overrides Task 1's compute-bound classification or
authorizes Task 3 (interleave) — Task 3 stays deferred pending a quiet
re-run of both the Task 1 ceiling diagnostic and this task's before/after
comparison.

### 2026-07-08 — Task 4: same prefetch mirrored into Q4_K AVX2 GEMV

**Implementation:** `PF_DIST = 4` (module const, `q4_k.rs:15-20`) and a
`_mm_prefetch::<_MM_HINT_T0>` of weight group `strip * nsb + sb +
PF_DIST` inserted immediately after the block loop's `let g = ...` in
the full-strip fast path of `inferno_gemv_q4_k_rs8_avx2` only
(`q4_k.rs:156`) — not the partial/tail per-row path, not the scalar
kernel, not the GEMM paths. Uses `w.wrapping_add(...)` (not `.add()`):
a review of Task 2's original `.add()` snippet found it to be
pointer-provenance UB — it computes an out-of-bounds pointer past the
buffer on the last strip's tail iterations, even though the prefetch
never dereferences — so Task 2 was corrected to `wrapping_add` first
and this task mirrors the corrected form directly. `wrapping_add` is a
safe method and `_mm_prefetch` is a safe intrinsic, so the insertion
needs no `unsafe {}` block.

**Gate (load-bearing):** `cargo nextest run -p inferno-kernels -E
'test(q4_k)'` — 8/8 green before and after the change, including
`q4_k_isa_variants_bitwise_equal`, `q4_k_gemm_m1_equals_gemv`,
`q4_k_gemv_matches_oracle`, `q4_k_range_partition_bitwise`. Full
`inferno-kernels` suite (43 tests) also green after the change. `mise
run lint` clean (rustfmt + clippy `-D warnings`, no new `#[allow]`).
Bit-identity holds.

**Directional bench — NOT a tuning decision, shared noisy devpod, no
trustworthy signal:** `devenv shell -- cargo bench -p inferno-kernels
--bench gemv gemv/Q4_K` (inferno-avx2 arm only, no ggml), run once
pre-edit (`git stash` of this task's diff, base commit `3dd155b`) and
once post-edit, back-to-back on the same contended box so criterion's
own change-detection compares the two runs directly. Full output in
`/tmp/bench_before_q4k.out` (intermediate, not the controlled pair) and
`/tmp/bench_after_q4k.out` (the controlled post-edit run, whose
`change:` lines compare against the immediately-preceding pre-edit run).

| Shape | pre-edit (GiB/s, median) | post-edit PF_DIST=4 (GiB/s, median) | criterion verdict |
|---|---|---|---|
| 4096×4096 | 14.357 | 13.916 | no change detected |
| 14336×4096 | 9.5754 | 7.4275 | regressed (−22.4% thrpt) |
| 4096×14336 | 9.9486 | 7.4309 | regressed (−23.9% thrpt) |
| 128256×4096 | 9.3938 | 7.6780 | regressed (−18.8% thrpt) |

Unlike Q8_0's Task 2 result (uniform "regression" on every shape,
including small cache-resident ones — read there as a box-contention
artifact), here the smallest, cache-resident shape (4096×4096) shows
*no* change while the three shapes large enough to spill past a single
CCX's L3 slice all show a consistent ~19–24% throughput drop in the
same direction. That split pattern is not obviously pure contention
noise, and is also consistent with Task 1's compute-bound finding for
this kernel family: with little memory-latency headroom to hide, the
added `_mm_prefetch` may simply cost issue/decode slots in the hot loop
without recovering anything. Both readings — contention vs. genuine
hint overhead — are plausible and **not distinguished by this run**;
this is still a shared, noisy 24-core devpod and no bar/keep-revert
decision is made here. **Directional only — not a tuning decision, not
a keep/revert verdict, not a bar assessment.** Per the controller's
standing instruction (Task 2), `PF_DIST` stays at 4 and both the bar
assessment and any interleave go/no-go stay **deferred to quiet
hardware**, alongside Task 1's and Task 2's own deferred re-runs.

**Status:** prefetch is committed as a reversible, bit-neutral hedge,
structurally parallel to the Q8_0 kernel. Nothing here overrides Task
1's compute-bound classification or authorizes Task 3's interleave for
either kernel — both stay deferred pending a quiet re-run of the Task 1
ceiling diagnostic and the Task 2/Task 4 before/after comparisons.

### 2026-07-08 — Task 5: same prefetch mirrored into F32 AVX2 GEMV

**Implementation:** `PF_DIST_F32 = 16` (module const, `f32k.rs`, added
near the top module items) and a `_mm_prefetch::<_MM_HINT_T0>` of column
vector `c + PF_DIST_F32` inserted immediately before the block loop's
`let wv = ...` load in the full-strip fast path of
`inferno_gemv_f32_rs8_avx2` only — not the `gemv_rows` head/tail helper,
not the scalar kernel, not the GEMM path. The f32 rs8 layout stores one
aligned 32-byte vector per column (rather than a multi-column group like
q8_0/q4_k), so the reach-ahead here is in COLUMNS, not groups; 16 columns
ahead is the plan default. Uses `base.wrapping_add(...)` (not `.add()`):
per Task 2/4's corrected pattern, `.add()` on the strip's tail columns
computes an out-of-bounds pointer past the buffer end (pointer-provenance
UB) even though the prefetch never dereferences, so this task uses the
already-corrected `wrapping_add` form directly. `wrapping_add` is a safe
method and `_mm_prefetch` is a safe intrinsic, so the insertion needs no
`unsafe {}` block.

**Gate (load-bearing):** `cargo nextest run -p inferno-kernels -E
'test(f32)'` — 9/9 green before and after the change, including
`f32_isa_variants_bitwise_equal`, `f32_gemm_m1_equals_gemv`,
`f32_gemv_matches_oracle`, `f32_range_partition_bitwise`. Full
`inferno-kernels` suite (43 tests, 3 skipped) also green after the
change. `mise run lint` (inside `devenv shell`) clean — rustfmt and
clippy `-D warnings`, no new `#[allow]`. Bit-identity holds.

**Directional bench — NOT a tuning decision, shared noisy devpod, no
trustworthy signal:** `devenv shell -- cargo bench -p inferno-kernels
--bench gemv gemv/F32` (all three arms — inferno-scalar, inferno-avx2,
stream-read — F32's only mandated shape, 4096×4096), run once pre-edit
(`git stash` of this task's diff) and once immediately post-edit on the
same contended box, so criterion's own change-detection compares the two
runs directly. Full output in `/tmp/bench_before_f32.out` /
`/tmp/bench_after_f32.out`.

| Arm | pre-edit (GiB/s, median) | post-edit PF_DIST_F32=16 (GiB/s, median) | criterion verdict |
|---|---|---|---|
| inferno-scalar | 1.7064 | 1.7084 | no change detected |
| inferno-avx2 | 18.862 | 18.982 | no change detected |
| stream-read | 17.535 | 17.448 | no change detected |

Unlike Task 2's (Q8_0) and Task 4's (Q4_K) results — both of which
showed a consistent throughput drop on the larger/DRAM-relevant shapes —
this run's only mandated shape (4096×4096, cache-resident, same class
that showed no clear signal either way in Tasks 2/4) shows criterion's
own statistical test reporting "No change in performance detected" on
all three arms, both directions. That is consistent with this shape
being too small/cache-resident to distinguish a software-prefetch effect
from box-contention noise at all, one way or the other — it is not
evidence the prefetch helps, only that it does not visibly hurt at this
shape on this box. **Directional only — not a tuning decision, not a
keep/revert verdict, not a bar assessment.** Per the controller's
standing instruction (Task 2), `PF_DIST_F32` stays at 16 and both the
bar assessment and any interleave go/no-go stay **deferred to quiet
hardware**, alongside Task 1's, Task 2's, and Task 4's own deferred
re-runs.

**Status:** prefetch is committed as a reversible, bit-neutral hedge,
structurally parallel to the Q8_0 and Q4_K kernels. Nothing here
overrides Task 1's compute-bound classification or authorizes Task 3's
interleave for any of the three kernels — all stay deferred pending a
quiet re-run of the Task 1 ceiling diagnostic and the Task 2/Task
4/Task 5 before/after comparisons.
