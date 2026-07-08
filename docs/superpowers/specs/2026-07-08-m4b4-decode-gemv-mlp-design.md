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
