# M4b.3 — Vectorized Attention Kernel Design

**Date:** 2026-07-07
**Status:** Approved design, pre-implementation
**Milestone:** M4b.3 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.2](2026-07-07-m4b2-per-thread-gap-design.md))

M4b.2 tiled prefill into batched GEMM and moved the matmuls into their
compute-bound regime (µbench 2.5–3.0× ggml; effective GB/s 12.5→~14.5),
but its t=1 prefill exit criterion **missed at 0.55×** (target ≥0.70×).
The recorded post-tiling profile reassigned the blame: with the matmuls
shrunk, **attention is 70.6% of prefill cycles** (up from 68.5%
pre-tiling — its *share* grew precisely because the GEMM tiling worked)
and **81.5% of decode cycles**. GEMM matmuls are only 28.3% of prefill
combined; the M4b.2 arithmetic ceiling shows that zeroing the *entire*
matmul cost still caps t=1 prefill at 0.78×. **The dominant lever for
both phases is the attention path, not the GEMM.**

The cause is structural and confirmed in the source. `lower_attention`
(`inferno-codegen/src/llvm/ops.rs`) emits attention as **fully scalar
inline LLVM IR** — a per-head scalar QK dot over `head_dim`, a scalar
max-subtraction softmax with a per-element `exp`, and a scalar AV
accumulate — run **once per query token** in prefill. It is the one hot
math path in the engine that never became a kernel: every matmul goes
through a tuned AVX2 `inferno-kernels` C-ABI symbol; attention is raw
scalar IR with no SIMD and no `KernelSet`. llama.cpp runs a vectorized
attention with a polynomial-exp softmax. M4b.3 closes that specific gap.

What this milestone rules in and out:

- **Vectorization is the lever; it is not gated.** The specs gate
  *parallel* (multi-thread) attention on the M4b.1 bare-metal rerun.
  Single-threaded SIMD vectorization is a different axis and is **not**
  gated — it lives entirely inside the t=1 per-thread criterion M4b.2
  already validated as meaningful on the CPU-quota'd devpod. M4b.3 stays
  single-threaded on purpose.
- **Attention is f32-only.** q/k/v activations and the KV cache are all
  f32 (KV stays f32 in M3+; see AGENTS.md). Unlike the matmul kernels,
  attention has **no dtype axis** — it is a single kernel family
  dispatched purely by ISA (`scalar` / `avx2`). This is simpler than the
  gemv/gemm path.
- **Prefill stays per-token.** The query-panel/flash-style blocking that
  would rewrite the causal-horizon loop is explicitly out of scope
  (below); it is a larger, riskier lever left for a later milestone.
  Vectorizing the per-token kernel already attacks the 70.6% / 81.5%
  slice and helps **both** phases with one piece of work.

## Scope Decisions (M4b.3)

| Decision | Choice |
|---|---|
| Measurement basis | t=1-vs-t=1 under the standard protocol (devenv shell, release, pinned Q8_0 model). Single-threaded throughout; no threading/scaling claim is made, so nothing here is gated on the M4b.1 bare-metal rerun |
| Kernel boundary | Attention becomes a C-ABI kernel family in `inferno-kernels` (`inferno_attention_f32_{scalar,avx2}`), f32-only, ISA-dispatched. `lower_attention` stops emitting the scalar triple-loop and instead declares + calls the symbol, the path `lower_gemv` already uses |
| `exp` implementation | **Vectorized polynomial `exp`** (minimax poly, ggml/SLEEF-style), evaluated **identically** in the scalar and AVX2 kernels. This is the source of the softmax speedup and is what llama.cpp does |
| Correctness — kernel vs kernel | **scalar-kernel vs AVX2-kernel bit-identical** (exact equality), asserted by the `inferno-kernels` rig — achievable because both share the poly `exp` and the same lane-partitioned reduction order. Extends the existing scalar-vs-SIMD invariant |
| Correctness — kernel vs interpreter | **Tolerance-bounded** by a new `attn_rel_tol` constant in `inferno-graph/src/tolerance.rs`, **derived from an observed error-distribution sweep** (an ignored `observed_error_*` diagnostic, same discipline as `gemv_rel_tol` / `LOGIT_TIE_EPSILON`) — never hand-set to make a test green |
| Exit criterion | **No hard 0.70× on attention alone** (M4b.2 showed attention + the deferred GEMM register-blocking are jointly necessary). Instead: (1) a fresh t=1 `--profile` capture showing attention is **no longer the majority** of prefill (target <45%, from 70.6%); (2) a recorded t=1 bench data point with **both** pp and tg improved. The 0.70× decision moves to a follow-up that re-evaluates the GEMM lever on top |
| Standing invariants | scalar-vs-SIMD kernel bit-identity (now extended to attention); compiled-vs-interpreter differential green with **no `logits_abs_tol` loosening**; the interpreter stays the independent std-`exp` oracle, unchanged |

**Explicitly out of scope:** parallel / multi-thread attention (gated on
the M4b.1 bare-metal rerun); prefill query-panel / flash-style blocking
(the causal-horizon rewrite — a later milestone); F16 KV cache (its own
decode-fork amendment, re-derives tolerances); AVX-512 attention bodies
(the `Isa::X86_64v4` detection stays ready); the register-blocked GEMM
escalation (M4b.2's separate follow-up); any change to the decode-fork
tolerances; any CI perf gate (never — AGENTS.md).

## What M4b.3 Adds

- **`inferno-kernels`** — new `inferno_attention_f32_scalar` /
  `inferno_attention_f32_avx2` C-ABI kernels (`#[unsafe(no_mangle)]
  extern "C"`); a shared vectorized polynomial-`exp` helper used by both;
  the `KernelSet`/registry selector extended so `reference_kernels` /
  `kernels_for` surface the attention fn pointer per ISA; rig coverage
  (scalar-vs-AVX2 bit-identity + kernel-vs-interpreter tolerance sweep).
- **`inferno-codegen`** — `lower_attention` rewritten to declare + call
  the ISA-selected symbol instead of emitting the scalar triple-loop; the
  existing `scores` `entry_alloca` passed through as the kernel's scratch
  pointer; the profiler's `attention` slot now times the call.
- **`inferno-graph`** — new `attn_rel_tol` tolerance constant (with its
  derivation note); the reference `ops::attention` is **unchanged**.

## Kernel Boundary & ABI

Attention today is inline IR that reads q/out via arena pointers, appends
this token's k/v into the f32 KV cache at `pos`, and reads a `scores`
stack buffer. As a kernel it takes those as explicit parameters. The
proposed C-ABI signature (final arg order firmed at plan time):

```
unsafe extern "C" fn inferno_attention_f32_{isa}(
    out:        *mut f32,    // this token's output rows [n_heads * head_dim]
    q:          *const f32,  // this token's query rows  [n_heads * head_dim]
    kv:         *mut f32,    // KV cache base pointer
    scores:     *mut f32,    // caller scratch, >= seq_len f32
    kv_base:    usize,       // per-layer K region offset (elements)
    v_off:      usize,       // V region offset from kv_base (= seq_len*kv_dim)
    pos:        usize,       // current position (causal horizon = pos+1)
    kv_dim:     usize,       // n_kv_heads * head_dim
    n_heads:    usize,
    n_kv_heads: usize,
    head_dim:   usize,
)
```

The kernel does the KV append (write this token's k/v at `pos`) and the
read (per-head scored softmax-weighted V sum), matching
`inferno_graph::ops::attention` op-for-op except for the vectorized `exp`
and lane-partitioned reductions. `scale = 1/sqrt(head_dim)` is derived
inside the kernel from `head_dim` (kept out of the ABI). `k`/`v` for this
token are read from the arena rows the codegen currently uses; the plan
step resolves whether they arrive as one fused `q|k|v` pointer or three.
Codegen selects the symbol from the same `Isa` it already uses for
gemv/gemm and calls it via `get_function(sym).build_call(...)`.

## Kernel Internals (AVX2)

Per query row, per head (`head_dim=64` on the pinned Qwen2.5-0.5B = eight
`__m256` lanes-of-8), the two-pass structure mirrors the interpreter so
the kernel is a faithful drop-in:

1. **QK-dot** — vectorized FMA accumulate of `q_head · kcache[t,g]` over
   `head_dim`, horizontal-reduce per visible position `t in 0..=pos`,
   scale → `scores[t]`.
2. **softmax** — vectorized max over `scores[..visible]`; then vectorized
   `(scores[t] − max)` fed to the **vectorized polynomial `exp`**;
   vectorized sum → `denom`.
3. **AV** — vectorized FMA accumulate of `(scores[t]/denom) · vcache[t,g]`
   into `out_head` over `head_dim`.

GQA mapping (`g = h / (n_heads/n_kv_heads)`) is unchanged. The **scalar
reference kernel** runs the identical polynomial `exp` and the identical
lane-partitioned reduction order, so the two kernels are bit-identical.

## Correctness Contract

Three layers, mirroring the gemm precedent:

1. **scalar-kernel vs AVX2-kernel → exact bit-identity**, asserted in the
   `inferno-kernels` rig over the property shape distribution. This is the
   flagship invariant (AGENTS.md); it holds here because both kernels
   share the poly `exp` and reduction tree.
2. **kernel vs interpreter (std-`exp` oracle) → `attn_rel_tol`**, a new
   constant in `inferno-graph/src/tolerance.rs` bounded by an ignored
   `observed_error_*` sweep across the shape distribution, exactly like
   `gemv_rel_tol`. The interpreter (`ops::attention`) is the ground truth
   and stays on `std::f32::exp` — it is **not** modified.
3. **End-to-end → `logits_abs_tol`** (unchanged: 1e-2 Q8_0). The
   compiled-vs-interpreter differential (`inferno-codegen --test
   differential`) and the artifact differential (`inferno-core --test
   artifact`) must stay green **with no loosening**. A degree-≈6 minimax
   `exp` is ~1 ULP (~1e-7 relative); propagated through a *normalizing*
   softmax and summed into logits, that sits orders of magnitude under
   the 1e-2 budget. If any differential goes red, the poly or reduction
   is wrong — the tolerance is never the fix.

## Measurement Protocol

All at t=1, devenv shell, release build, pinned
`qwen2.5-0.5b-instruct-q8_0.gguf`, results recorded in `## Amendments`
(never edited once recorded):

- **Kernel µbench** — an attention criterion group in
  `inferno-kernels/benches` over the pinned model's head geometry
  (`head_dim=64`, `n_heads=14`, `n_kv_heads=2`) at representative
  horizons, scalar vs AVX2, so the SIMD win is visible per-position.
- **`--profile` capture** — t=1 prefill + decode profile after the swap;
  exit-criterion check is attention share < 45% of prefill.
- **Bench data point** — `mise run bench -- <model> --threads 1 --json`;
  record the table + json and the pp/tg ratios vs llama.cpp t=1. Both pp
  and tg must improve over M4b.2's recorded 0.55× / 0.43×.

## Gated Follow-Ups (not tasks in this milestone)

- **Register-blocked GEMM escalation** (inherited from M4b.2). After
  M4b.3's profile exists, a follow-up decides whether register-blocked
  GEMM tiles on top of vectorized attention are needed to reach 0.70×
  prefill. Its own spec amendment / plan.
- **Prefill query-panel (flash-style) attention.** The causal-horizon
  rewrite that batches the query panel against shared K/V with online
  softmax — the bigger prefill lever, deferred here for risk. Own plan.
- **Parallel attention.** Multi-threading the per-head loop stays gated
  on the M4b.1 bare-metal rerun.
- **AVX-512 / F16 KV.** Unchanged from their existing gates.

## Amendments

*(Recorded protocol data points and scoped amendments land here; never
edit a recorded data point.)*
