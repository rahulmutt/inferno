# M4b.3 Vectorized Attention Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scalar inline attention codegen (70.6% of prefill / 81.5% of decode cycles) with a SIMD-vectorized, single-threaded C-ABI kernel that helps both phases.

**Architecture:** Attention becomes an f32-only, ISA-dispatched C-ABI kernel family in `inferno-kernels` (`inferno_attention_f32_{scalar,avx2}`), authored the same way as the gemv/gemm kernels (scalar `mul_add` reference + `#[target_feature(avx2,fma)]` SIMD sibling that matches the scalar reduction order bitwise). Codegen's `lower_attention` stops emitting the scalar triple-loop and instead declares + calls the ISA-selected symbol, passing the query/output arena pointers, the KV-cache pointer, and a `scores` scratch pointer. Correctness is guarded in three layers exactly like gemm: scalar-vs-AVX2 kernel bit-identity, kernel-vs-interpreter within a newly-derived `attn_rel_tol`, and the unchanged end-to-end `logits_abs_tol` differential.

**Tech Stack:** Rust, `std::arch` x86_64 AVX2/FMA intrinsics, LLVM via `inkwell` (codegen), `proptest` (kernel rig), `criterion` (µbench), `insta` (snapshots).

## Global Constraints

- **`inferno-kernels` is one of only two crates allowed `unsafe`** (intrinsics + the C ABI); it opts out of the workspace `unsafe_code = "deny"` with its own `[lints.rust]`. Do not add `unsafe` anywhere else.
- **Scalar and SIMD kernel variants must stay bit-identical** — the rig asserts exact equality (`to_bits()`). The AVX2 attention kernel must match the scalar reference's accumulation order and use the *identical* polynomial `exp`.
- **Attention is f32-only, ISA-dispatched** (no dtype axis): symbols are `inferno_attention_f32_scalar` / `inferno_attention_f32_avx2`.
- **Tolerances are derived from observed data, never set to make a red test green.** `attn_rel_tol` is derived by a throwaway `#[ignore]` sweep printing observed max error, then armed at ~4× that (the `gemv_rel_tol` discipline).
- **The interpreter `inferno_graph::ops::attention` is the independent std-`exp` oracle and is NOT modified.**
- **Never loosen `logits_abs_tol`** (1e-2 for Q8_0/Q4_K) — the compiled-vs-interpreter differential (`inferno-codegen --test differential`) and artifact differential (`inferno-core --test artifact`) must stay green on their own.
- **Kernel perf numbers come only from `mise run bench-kernels`** inside the devenv shell on quiet hardware; end-to-end `inferno bench` is a manual protocol, never a CI gate. Record data points in this spec's Amendments; never edit a recorded data point.
- **Workflows are mise tasks:** `mise run test` / `lint` / `bench-kernels` / `differential`. Don't hand-roll cargo invocations in docs/CI.
- **Tensor shapes are row-major, outermost first.** GQA group mapping is `g = h / (n_heads / n_kv_heads)` (contiguous, not interleaved).

---

## File Structure

- `crates/inferno-kernels/src/expf.rs` — **new**: vectorized polynomial `expf` (scalar-lane + AVX2 `__m256`), the shared softmax primitive. One responsibility: `exp`.
- `crates/inferno-kernels/src/attention.rs` — **new**: the two attention kernels + a private scalar core they share, plus the KV-append. One responsibility: attention.
- `crates/inferno-kernels/src/lib.rs` — **modify**: `mod expf; mod attention;` + re-export the two attention symbols and the `AttnFn` type.
- `crates/inferno-kernels/src/registry.rs` — **modify**: add an `attention` selector (`attention_kernel(isa) -> AttnFn`) + a safe `KernelSet`-style wrapper for the rig.
- `crates/inferno-kernels/tests/rig.rs` — **modify**: attention oracle, bit-identity proptest, tolerance proptest, and the `#[ignore]` observed-error sweep.
- `crates/inferno-graph/src/tolerance.rs` — **modify**: add `attn_rel_tol`.
- `crates/inferno-codegen/src/loopir.rs` — **modify**: add `attention_symbol(isa)` helper next to `gemv_symbol`/`gemm_symbol`.
- `crates/inferno-codegen/src/llvm/mod.rs` — **modify**: declare the attention symbols in `declare_kernels`.
- `crates/inferno-codegen/src/llvm/ops.rs` — **modify**: rewrite `lower_attention` to call the symbol.
- `crates/inferno-codegen/tests/differential.rs` — **modify**: extend `retain_kernel_symbols` with the two attention symbols.
- `crates/inferno-core/src/artifact.rs` — **modify**: extend `ensure_kernels_linked` with the two attention symbols.
- `crates/inferno-kernels/benches/attention.rs` — **new**: criterion µbench group (registered in `Cargo.toml`).

---

## Task 1: Vectorized polynomial `expf` primitive

**Files:**
- Create: `crates/inferno-kernels/src/expf.rs`
- Modify: `crates/inferno-kernels/src/lib.rs` (add `mod expf;`)
- Test: inline `#[cfg(test)]` in `expf.rs`

**Interfaces:**
- Produces:
  - `pub(crate) fn expf_scalar(x: f32) -> f32` — one lane of the polynomial.
  - `#[target_feature(enable = "avx2,fma")] pub(crate) unsafe fn expf_avx2(x: __m256) -> __m256` — 8 lanes, **bit-identical** to `expf_scalar` applied per lane (same constants, same FMA order).

The polynomial: range-reduce `x = n·ln2 + r` with `n = round(x·log2e)`, `r` via two-stage Cody-Waite subtraction (`r = x − n·ln2_hi − n·ln2_lo`), evaluate a degree-6 minimax poly of `exp(r)` in Horner form with FMA, then scale by `2^n` via integer exponent bit insertion. Clamp `x` to `[-88.0, 88.0]` first so `2^n` cannot overflow/underflow f32.

- [ ] **Step 1: Write the failing test**

In `crates/inferno-kernels/src/expf.rs`:

```rust
//! Vectorized polynomial expf shared by the attention softmax. The scalar
//! lane and the AVX2 lane evaluate the *identical* constants and FMA order,
//! so a softmax built on them is bit-identical across ISAs (rig invariant).
//! Accuracy target: << 1 ULP-ish; the interpreter's std::exp stays the
//! ground truth, bounded by attn_rel_tol (see inferno-graph tolerance.rs).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const LOG2E: f32 = 1.442_695_04;
const LN2_HI: f32 = 0.693_359_38;
const LN2_LO: f32 = -2.121_944_4e-4;
const C: [f32; 6] = [
    1.0,
    1.0,
    0.5,
    0.166_666_67,
    0.041_666_67,
    0.008_333_34,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_matches_std_within_relative_1e_6() {
        for i in -880..=880 {
            let x = i as f32 * 0.1;
            let got = expf_scalar(x);
            let want = x.exp();
            let rel = (got - want).abs() / want.max(1e-30);
            assert!(rel <= 1e-6, "x={x}: got {got}, want {want}, rel {rel}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_lane_is_bitwise_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let xs = [-88.0f32, -3.3, -0.7, 0.0, 0.2, 1.5, 11.0, 87.9];
        // SAFETY: avx2 detected above.
        let out = unsafe {
            let v = _mm256_loadu_ps(xs.as_ptr());
            let r = expf_avx2(v);
            let mut o = [0f32; 8];
            _mm256_storeu_ps(o.as_mut_ptr(), r);
            o
        };
        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(out[i].to_bits(), expf_scalar(x).to_bits(), "lane {i} x={x}");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p inferno-kernels --lib expf`
Expected: FAIL — `expf_scalar`/`expf_avx2` not found (won't compile).

- [ ] **Step 3: Implement the scalar lane and AVX2 lane**

Add above the `#[cfg(test)]` block in `expf.rs`:

```rust
#[inline]
pub(crate) fn expf_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * LOG2E).round();
    let r = n.mul_add(-LN2_LO, n.mul_add(-LN2_HI, x));
    // Horner: p = C0 + r*(C1 + r*(C2 + ... )). C0==C1==1 => exp(r) series.
    let mut p = C[5];
    p = p.mul_add(r, C[4]);
    p = p.mul_add(r, C[3]);
    p = p.mul_add(r, C[2]);
    p = p.mul_add(r, C[1]);
    p = p.mul_add(r, C[0]);
    // scale by 2^n via exponent bits: (n as i32 + 127) << 23.
    let pow2n = f32::from_bits((((n as i32) + 127) << 23) as u32);
    p * pow2n
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn expf_avx2(x: __m256) -> __m256 {
    let x = _mm256_min_ps(_mm256_set1_ps(88.0), _mm256_max_ps(_mm256_set1_ps(-88.0), x));
    // round-to-nearest-even matches f32::round for the |x*log2e| range here
    // (n is an integer well under 2^23); use the AVX2 round intrinsic.
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm256_mul_ps(x, _mm256_set1_ps(LOG2E)),
    );
    let r = _mm256_fmadd_ps(
        n,
        _mm256_set1_ps(-LN2_LO),
        _mm256_fmadd_ps(n, _mm256_set1_ps(-LN2_HI), x),
    );
    let mut p = _mm256_set1_ps(C[5]);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[4]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[3]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[2]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[1]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[0]));
    // pow2n = ((n_i32 + 127) << 23) reinterpreted as f32, per lane.
    let ni = _mm256_cvtps_epi32(n);
    let bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(ni, _mm256_set1_epi32(127)));
    _mm256_mul_ps(p, _mm256_castsi256_ps(bits))
}
```

Then in `crates/inferno-kernels/src/lib.rs`, add `mod expf;` near the other `mod` lines.

**Note on bit-identity:** `f32::round` is round-half-away-from-zero while `_MM_FROUND_TO_NEAREST_INT` is round-half-to-even. They differ only at exact `x·log2e == k+0.5`, which does not occur for any test input; if the tolerance sweep in Task 6 ever surfaces a lane mismatch, replace the scalar `.round()` with a round-half-to-even (`(x*LOG2E).round_ties_even()`) to match the intrinsic exactly. Prefer `round_ties_even()` from the start if the toolchain (mise-pinned) has it stabilized.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p inferno-kernels --lib expf`
Expected: PASS (both tests). If `avx2_lane_is_bitwise_scalar` fails on a half-even boundary, apply the round_ties_even fix from Step 3's note and re-run.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/src/expf.rs crates/inferno-kernels/src/lib.rs
git commit -m "feat(kernels): vectorized polynomial expf (scalar + avx2, bit-identical lanes)"
```

---

## Task 2: Scalar attention kernel + shared core

**Files:**
- Create: `crates/inferno-kernels/src/attention.rs`
- Modify: `crates/inferno-kernels/src/lib.rs` (add `mod attention;` + re-export symbol)
- Test: `crates/inferno-kernels/tests/rig.rs` (oracle + one matches-interpreter proptest)

**Interfaces:**
- Consumes: `crate::expf::expf_scalar` (Task 1).
- Produces:
  - ```rust
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn inferno_attention_f32_scalar(
        out: *mut f32, q: *const f32, kv: *mut f32, scores: *mut f32,
        kv_base: usize, v_off: usize, pos: usize, kv_dim: usize,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
    )
    ```
  - For each head it computes scaled QK scores into `scores[0..=pos]`, max-subtraction softmax with `expf_scalar`, and the softmax-weighted V sum into `out`. It is **read-only over the KV cache** (the caller has already appended this token's k/v at `pos`).

**Contract (matches codegen's arena layout).** The kernel is called once per query token and is **read-only**: `q` points at this token's `[n_heads*head_dim]` query row, and `kv` **already contains** this token's k and v at position `pos` (codegen appends them with its existing IR before the call; the rig driver and oracle append identically). `kv_base` is this layer's K-region element offset and `v_off = seq_len*kv_dim` is the K→V region gap (so V-region base is `kv_base + v_off`). `scale = 1/sqrt(head_dim)`. This mirrors the *read* half of `inferno_graph::ops::attention` op-for-op (dot → scale → max → exp → normalize → AV) except `exp` is the poly. Keeping the append in the caller means the ABI never has to carry k/v pointers and the existing, already-correct append IR is reused unchanged.

- [ ] **Step 1: Write the failing test** (in `crates/inferno-kernels/tests/rig.rs`, appended)

```rust
// ---------- Attention ----------

/// Reference: the interpreter attention over a single query row at `pos`,
/// with the KV cache pre-populated for positions 0..pos and this token's
/// k/v appended. Returns the [n_heads*head_dim] output row.
fn attn_oracle(
    q: &[f32], k: &[f32], v: &[f32],
    kcache: &mut Vec<f32>, vcache: &mut Vec<f32>,
    pos: usize, kv_dim: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
) -> Vec<f32> {
    // Append this token's k/v at position `pos`.
    kcache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k);
    vcache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(v);
    let qt = inferno_graph::Tensor { shape: vec![1, n_heads * head_dim], data: q.to_vec() };
    inferno_graph::ops::attention(
        &qt, kcache, vcache, pos + 1, n_heads, n_kv_heads, head_dim, pos,
    ).data
}

/// Drive the scalar attention kernel for one token; returns [n_heads*head_dim].
/// Appends this token's k/v into `kv` at `pos` first (the caller's job — the
/// kernel is read-only), matching what codegen does before the call.
fn attn_kernel_scalar(
    q: &[f32], k: &[f32], v: &[f32],
    kv: &mut [f32], seq_len: usize, pos: usize,
    kv_dim: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
) -> Vec<f32> {
    // Append k/v at pos: K region [0..seq*kv_dim), V region after it.
    kv[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k);
    let vreg = seq_len * kv_dim;
    kv[vreg + pos * kv_dim..vreg + (pos + 1) * kv_dim].copy_from_slice(v);
    let mut out = vec![f32::NAN; n_heads * head_dim];
    let mut scores = vec![0f32; seq_len];
    // Single-layer cache: kv_base=0, v_off=seq_len*kv_dim.
    // SAFETY: buffers sized to the documented contract; pos < seq_len.
    unsafe {
        inferno_kernels::inferno_attention_f32_scalar(
            out.as_mut_ptr(), q.as_ptr(), kv.as_mut_ptr(), scores.as_mut_ptr(),
            0, seq_len * kv_dim, pos, kv_dim, n_heads, n_kv_heads, head_dim,
        );
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn attention_scalar_matches_interpreter(
        seed in any::<u64>(), pos in 0usize..12, hd in prop::sample::select(vec![8usize, 16, 64]),
    ) {
        let (n_heads, n_kv_heads, head_dim) = (4usize, 2usize, hd);
        let kv_dim = n_kv_heads * head_dim;
        let seq_len = 16usize;
        // Random pre-populated cache for positions < pos, plus this token's k/v.
        let mut kv = pseudo(seed, 2 * seq_len * kv_dim); // scalar-kernel KV (K then V regions)
        let q = pseudo(seed ^ 1, n_heads * head_dim);
        let k = pseudo(seed ^ 2, kv_dim);
        let v = pseudo(seed ^ 3, kv_dim);
        // Oracle uses separate k/v caches; seed them from the same bytes as
        // the kernel's single kv buffer (K region = kv[..seq*kv_dim], V region after).
        let mut kc = kv[..seq_len * kv_dim].to_vec();
        let mut vc = kv[seq_len * kv_dim..].to_vec();
        let want = attn_oracle(&q, &k, &v, &mut kc, &mut vc, pos, kv_dim, n_heads, n_kv_heads, head_dim);
        let got = attn_kernel_scalar(&q, &k, &v, &mut kv, seq_len, pos, kv_dim, n_heads, n_kv_heads, head_dim);
        // Poly exp vs std exp: bounded, not bitwise. attn_rel_tol derived in Task 6;
        // until then assert a loose 1e-4 to prove structural correctness.
        let scale = want.iter().fold(1f32, |m, x| m.max(x.abs()));
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            prop_assert!((g - w).abs() <= 1e-4 * scale, "elem {i}: got {g} want {w}");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p inferno-kernels --test rig attention_scalar_matches_interpreter`
Expected: FAIL — `inferno_attention_f32_scalar` not found (link error).

- [ ] **Step 3: Implement the scalar kernel**

Create `crates/inferno-kernels/src/attention.rs`:

```rust
//! Causal GQA attention as a C-ABI kernel (f32-only, ISA-dispatched).
//! Mirrors `inferno_graph::ops::attention` op-for-op, except the softmax
//! `exp` is the shared polynomial (`crate::expf`), so the compiled path is
//! bounded against the std-exp interpreter by `attn_rel_tol`, and the
//! scalar and AVX2 kernels are bit-identical to each other (shared poly +
//! reduction order). One call = one query token.

use crate::expf::expf_scalar;

/// # Safety
/// - `out`, `q` valid for `n_heads*head_dim` f32.
/// - `kv` valid for the K region `[kv_base .. kv_base + seq_len*kv_dim]`
///   and V region `[kv_base + v_off ..][.. seq_len*kv_dim]`, and already
///   contains this token's k/v at `pos`; `pos < seq_len`.
/// - `scores` valid for `pos+1` f32. Read-only over `kv`.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_attention_f32_scalar(
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    scores: *mut f32,
    kv_base: usize,
    v_off: usize,
    pos: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    // SAFETY: contract above. Delegate to a safe-slice core for clarity.
    unsafe {
        let q = std::slice::from_raw_parts(q, n_heads * head_dim);
        let out = std::slice::from_raw_parts_mut(out, n_heads * head_dim);
        let scores = std::slice::from_raw_parts_mut(scores, pos + 1);
        // KV regions (single flat buffer; kv_base/v_off pick this layer).
        let kv = std::slice::from_raw_parts(kv, kv_base + 2 * v_off);
        attn_core_scalar(out, q, kv, scores, kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim);
    }
}

#[allow(clippy::too_many_arguments)]
fn attn_core_scalar(
    out: &mut [f32],
    q: &[f32],
    kv: &[f32],
    scores: &mut [f32],
    kv_base: usize,
    v_off: usize,
    pos: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let group = n_heads / n_kv_heads;
    let kreg = kv_base;
    let vreg = kv_base + v_off;
    let visible = pos + 1;
    for h in 0..n_heads {
        let g = h / group;
        let qh = &q[h * head_dim..][..head_dim];
        // scores[t] = dot(qh, kcache[t,g]) * scale
        for (t, sc) in scores.iter_mut().enumerate().take(visible) {
            let kbase = kreg + t * kv_dim + g * head_dim;
            let mut acc = 0f32;
            for d in 0..head_dim {
                acc = qh[d].mul_add(kv[kbase + d], acc);
            }
            *sc = acc * scale;
        }
        let max = scores[..visible].iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
        let mut denom = 0f32;
        for sc in scores[..visible].iter_mut() {
            *sc = expf_scalar(*sc - max);
            denom += *sc;
        }
        let oh = &mut out[h * head_dim..][..head_dim];
        oh.fill(0.0);
        for (t, &w) in scores[..visible].iter().enumerate() {
            let vbase = vreg + t * kv_dim + g * head_dim;
            let wn = w / denom;
            for d in 0..head_dim {
                oh[d] = wn.mul_add(kv[vbase + d], oh[d]);
            }
        }
    }
}
```

In `crates/inferno-kernels/src/lib.rs`: add `mod attention;` and re-export, mirroring the gemv line:

```rust
pub use attention::inferno_attention_f32_scalar;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p inferno-kernels --test rig attention_scalar_matches_interpreter`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/src/attention.rs crates/inferno-kernels/src/lib.rs crates/inferno-kernels/tests/rig.rs
git commit -m "feat(kernels): scalar attention kernel matching the interpreter (poly-exp softmax)"
```

---

## Task 3: AVX2 attention kernel (bit-identical to scalar)

**Files:**
- Modify: `crates/inferno-kernels/src/attention.rs`
- Modify: `crates/inferno-kernels/src/lib.rs` (re-export `inferno_attention_f32_avx2`)
- Test: `crates/inferno-kernels/tests/rig.rs` (bit-identity proptest)

**Interfaces:**
- Consumes: `crate::expf::expf_avx2` (Task 1), the scalar kernel (Task 2).
- Produces: `#[unsafe(no_mangle)] pub unsafe extern "C" fn inferno_attention_f32_avx2(..same ABI as scalar..)`.

**Bit-identity strategy.** `head_dim` on the target models is a multiple of 8 (64, 128), so the QK-dot and AV inner loops vectorize cleanly into `head_dim/8` `__m256` FMA steps. The reduction order must match the scalar core exactly: the scalar dot accumulates `d = 0,1,2,...` sequentially, but the AVX2 lanes accumulate 8 partial sums that are horizontally reduced at the end — these are **different** orders and will NOT be bitwise equal. To keep bit-identity, the **scalar reference must accumulate in the same 8-lane-partitioned order.** Implement a shared `dot8`/`av8` accumulation shape used by BOTH kernels (scalar simulates 8 lanes with an `[f32; 8]` array reduced by the identical tree). Refactor Task 2's scalar core to this lane-partitioned shape in this task, then the AVX2 kernel mirrors it. When `head_dim % 8 != 0` (not on target models, but the rig samples `hd=8/16/64` so it is always a multiple of 8 here) the kernels fall back to the sequential scalar path in both — identical by construction.

- [ ] **Step 1: Write the failing bit-identity test** (append to `rig.rs`)

```rust
fn attn_kernel_avx2(
    q: &[f32], k: &[f32], v: &[f32],
    kv: &mut [f32], seq_len: usize, pos: usize,
    kv_dim: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
) -> Vec<f32> {
    // Append k/v at pos (caller's job; kernel is read-only) — same as scalar.
    kv[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k);
    let vreg = seq_len * kv_dim;
    kv[vreg + pos * kv_dim..vreg + (pos + 1) * kv_dim].copy_from_slice(v);
    let mut out = vec![f32::NAN; n_heads * head_dim];
    let mut scores = vec![0f32; seq_len];
    // SAFETY: same contract as the scalar driver; avx2 checked by caller.
    unsafe {
        inferno_kernels::inferno_attention_f32_avx2(
            out.as_mut_ptr(), q.as_ptr(), kv.as_mut_ptr(), scores.as_mut_ptr(),
            0, seq_len * kv_dim, pos, kv_dim, n_heads, n_kv_heads, head_dim,
        );
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn attention_isa_variants_bitwise_equal(
        seed in any::<u64>(), pos in 0usize..12, hd in prop::sample::select(vec![8usize, 16, 64]),
    ) {
        if !std::is_x86_feature_detected!("avx2") { return Ok(()); }
        let (n_heads, n_kv_heads, head_dim) = (4usize, 2usize, hd);
        let kv_dim = n_kv_heads * head_dim;
        let seq_len = 16usize;
        let base_kv = pseudo(seed, 2 * seq_len * kv_dim);
        let q = pseudo(seed ^ 1, n_heads * head_dim);
        let k = pseudo(seed ^ 2, kv_dim);
        let v = pseudo(seed ^ 3, kv_dim);
        let mut kv_s = base_kv.clone();
        let mut kv_a = base_kv;
        let a = attn_kernel_scalar(&q, &k, &v, &mut kv_s, seq_len, pos, kv_dim, n_heads, n_kv_heads, head_dim);
        let b = attn_kernel_avx2(&q, &k, &v, &mut kv_a, seq_len, pos, kv_dim, n_heads, n_kv_heads, head_dim);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            prop_assert_eq!(x.to_bits(), y.to_bits(), "elem {i}: scalar {x} avx2 {y}");
        }
        // KV appends must also be bit-identical.
        prop_assert!(kv_s.iter().zip(&kv_a).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p inferno-kernels --test rig attention_isa_variants_bitwise_equal`
Expected: FAIL — `inferno_attention_f32_avx2` not found.

- [ ] **Step 3: Refactor the scalar core to lane-partitioned accumulation, then add the AVX2 kernel**

In `attention.rs`, replace the scalar dot/AV inner loops with an 8-lane-partitioned reduction and add the matching AVX2 kernel. Add helpers used by both:

```rust
/// Dot product over `head_dim` in 8-lane-partitioned order (an [f32; 8] of
/// partial sums, then a fixed reduction tree), so the AVX2 kernel's
/// horizontal reduce is bitwise-identical. `head_dim` here is a multiple of 8.
#[inline]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    for chunk in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let (ca, cb) = chunk;
        for l in 0..8 {
            lanes[l] = ca[l].mul_add(cb[l], lanes[l]);
        }
    }
    reduce8(lanes)
}

/// The horizontal reduction tree AVX2 uses: (0+4)(1+5)(2+6)(3+7) then pairwise.
#[inline]
fn reduce8(v: [f32; 8]) -> f32 {
    let a = [v[0] + v[4], v[1] + v[5], v[2] + v[6], v[3] + v[7]];
    let b = [a[0] + a[2], a[1] + a[3]];
    b[0] + b[1]
}
```

Rewrite `attn_core_scalar`'s score loop to `*sc = dot8(qh, &kv[kbase..kbase+head_dim]) * scale;` and the AV loop to accumulate per-`d` (AV has no cross-`d` reduction, so its order is already lane-parallel — keep the `mul_add` per `d`, which the AVX2 kernel matches lane-for-lane). Add the AVX2 kernel:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// # Safety: as `inferno_attention_f32_scalar`, plus the CPU has AVX2+FMA.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_attention_f32_avx2(
    out: *mut f32, q: *const f32, kv: *mut f32, scores: *mut f32,
    kv_base: usize, v_off: usize, pos: usize, kv_dim: usize,
    n_heads: usize, n_kv_heads: usize, head_dim: usize,
) {
    // SAFETY: contract as scalar; head_dim on target models is a mult of 8.
    unsafe {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group = n_heads / n_kv_heads;
        let kreg = kv_base;
        let vreg = kv_base + v_off;
        // Read-only: caller already appended this token's k/v at `pos`.
        let visible = pos + 1;
        let scale_v = _mm256_set1_ps(scale);
        for h in 0..n_heads {
            let g = h / group;
            let qh = q.add(h * head_dim);
            // scores[t] = reduce8(sum_d qh[d]*kcache) * scale
            for t in 0..visible {
                let kb = kv.add(kreg + t * kv_dim + g * head_dim);
                let mut acc = _mm256_setzero_ps();
                let mut d = 0;
                while d < head_dim {
                    let qv = _mm256_loadu_ps(qh.add(d));
                    let kvv = _mm256_loadu_ps(kb.add(d));
                    acc = _mm256_fmadd_ps(qv, kvv, acc);
                    d += 8;
                }
                *scores.add(t) = hsum8(acc) * scale;
            }
            // max
            let mut max = f32::NEG_INFINITY;
            for t in 0..visible { max = max.max(*scores.add(t)); }
            // exp + denom (8 lanes at a time, tail scalar via expf_scalar)
            let maxv = _mm256_set1_ps(max);
            let mut denom = 0f32;
            let mut t = 0;
            while t + 8 <= visible {
                let s = _mm256_loadu_ps(scores.add(t));
                let e = crate::expf::expf_avx2(_mm256_sub_ps(s, maxv));
                _mm256_storeu_ps(scores.add(t), e);
                denom += hsum8(e);
                t += 8;
            }
            while t < visible {
                let e = crate::expf::expf_scalar(*scores.add(t) - max);
                *scores.add(t) = e;
                denom += e;
                t += 1;
            }
            // AV: oh[d] += (scores[t]/denom) * vcache
            let oh = out.add(h * head_dim);
            for d in (0..head_dim).step_by(8) {
                _mm256_storeu_ps(oh.add(d), _mm256_setzero_ps());
            }
            let inv = _mm256_set1_ps(1.0); // denom applied per-weight below to match scalar
            let _ = inv;
            for t in 0..visible {
                let wn = _mm256_set1_ps(*scores.add(t) / denom);
                let vb = kv.add(vreg + t * kv_dim + g * head_dim);
                for d in (0..head_dim).step_by(8) {
                    let cur = _mm256_loadu_ps(oh.add(d));
                    let vv = _mm256_loadu_ps(vb.add(d));
                    _mm256_storeu_ps(oh.add(d), _mm256_fmadd_ps(wn, vv, cur));
                }
            }
        }
        let _ = (scale_v, scores);
    }
}

/// Horizontal sum matching `reduce8`'s tree: (lo+hi) halves then pairwise.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum8(v: __m256) -> f32 {
    // SAFETY: avx2 enabled by the caller's target_feature.
    unsafe {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);              // [0+4,1+5,2+6,3+7]
        let sh = _mm_movehl_ps(s, s);            // [2+6,3+7,..]
        let s2 = _mm_add_ps(s, sh);              // [(0+4)+(2+6),(1+5)+(3+7),..]
        let s3 = _mm_add_ss(s2, _mm_shuffle_ps::<0x55>(s2, s2));
        _mm_cvtss_f32(s3)
    }
}
```

Re-export in `lib.rs`:

```rust
#[cfg(target_arch = "x86_64")]
pub use attention::inferno_attention_f32_avx2;
```

**Bit-identity note:** `hsum8` must reduce in the exact order `reduce8` does. `_mm_movehl_ps`+`_mm_add_ps` gives `(0+4)+(2+6)` and `(1+5)+(3+7)`; the final `_mm_add_ss` adds those two — matching `reduce8`'s `b[0]+b[1]`. The scalar `dot8` uses the same tree. The AV loop uses `mul_add` per `d` in both (no reduction), and the exp tail uses `expf_scalar` for the same lanes, so tail lanes match too. Only the `denom`-summation order across the 8-lane blocks differs from the scalar `denom += *sc` sequential sum — **make the scalar `denom` accumulate in the same block-of-8 + `hsum8`-tree order** (mirror the AVX2 loop structure in `attn_core_scalar`) or the two denoms diverge. This is the one subtle spot; the bit-identity proptest is what catches it.

- [ ] **Step 4: Run the bit-identity test + the scalar-vs-interpreter test**

Run: `cargo test -p inferno-kernels --test rig attention_`
Expected: PASS (both `attention_scalar_matches_interpreter` and `attention_isa_variants_bitwise_equal`). If bit-identity fails, align the scalar `denom`/`dot`/`reduce8` order to the AVX2 structure per the note above — never widen a tolerance to pass this (it is exact-equality).

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/src/attention.rs crates/inferno-kernels/src/lib.rs crates/inferno-kernels/tests/rig.rs
git commit -m "feat(kernels): avx2 attention kernel, bit-identical to scalar reference"
```

---

## Task 4: Registry selector + safe wrapper

**Files:**
- Modify: `crates/inferno-kernels/src/registry.rs`
- Test: `crates/inferno-kernels/src/registry.rs` (`#[cfg(test)]` selection test)

**Interfaces:**
- Consumes: the two symbols (Tasks 2–3), `KernelIsa`, `inferno_target::Isa`.
- Produces:
  - `type AttnFn = unsafe extern "C" fn(*mut f32, *const f32, *mut f32, *mut f32, usize, usize, usize, usize, usize, usize, usize);`
  - `pub fn attention_kernel(isa: Isa) -> Option<AttnFn>` — returns the AVX2 symbol for v3/v4 when the CPU supports it, else the scalar symbol; `None` never (scalar always available) — return `AttnFn` directly, not `Option`, if simpler. Keep signature consistent with `kernels_for`'s ISA handling.

- [ ] **Step 1: Write the failing test** (in `registry.rs` `#[cfg(test)] mod tests`)

```rust
#[test]
fn attention_selector_picks_isa() {
    use inferno_target::Isa;
    // Scalar is always available.
    let s = attention_reference();
    assert!(std::ptr::fn_addr_eq(s, crate::inferno_attention_f32_scalar as AttnFn));
    // v3 on an AVX2 host resolves to the avx2 symbol; on a non-AVX2 host it
    // must fall back to scalar (never hand out an unrunnable symbol).
    let k = attention_kernel(Isa::X86_64v3);
    if KernelIsa::Avx2.available() {
        assert!(std::ptr::fn_addr_eq(k, crate::inferno_attention_f32_avx2 as AttnFn));
    } else {
        assert!(std::ptr::fn_addr_eq(k, crate::inferno_attention_f32_scalar as AttnFn));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p inferno-kernels --lib attention_selector_picks_isa`
Expected: FAIL — `attention_kernel`/`attention_reference` not found.

- [ ] **Step 3: Implement the selector**

Add to `registry.rs` (near `kernels_for`/`reference_kernels`):

```rust
/// Attention is f32-only and has no dtype axis; a single fn-pointer type.
pub type AttnFn = unsafe extern "C" fn(
    *mut f32, *const f32, *mut f32, *mut f32,
    usize, usize, usize, usize, usize, usize, usize,
);

/// The portable scalar attention kernel — always runnable.
pub fn attention_reference() -> AttnFn {
    crate::inferno_attention_f32_scalar
}

/// The attention kernel for a target ISA, falling back to scalar when the
/// running CPU can't execute the SIMD variant (mirrors `kernels_for`).
pub fn attention_kernel(isa: Isa) -> AttnFn {
    let kisa = match isa {
        Isa::X86_64v3 | Isa::X86_64v4 => KernelIsa::Avx2,
    };
    #[cfg(target_arch = "x86_64")]
    if kisa.available() {
        return crate::inferno_attention_f32_avx2;
    }
    let _ = kisa;
    crate::inferno_attention_f32_scalar
}
```

Re-export from `lib.rs` if `registry`'s items are surfaced there (match how `kernels_for` is re-exported): add `attention_kernel, attention_reference, AttnFn` to the existing `pub use registry::{...}` line.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p inferno-kernels --lib attention_selector_picks_isa`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/src/registry.rs crates/inferno-kernels/src/lib.rs
git commit -m "feat(kernels): attention_kernel ISA selector + AttnFn type"
```

---

## Task 5: `attn_rel_tol` derivation + tolerance constant

**Files:**
- Modify: `crates/inferno-graph/src/tolerance.rs`
- Test: `crates/inferno-kernels/tests/rig.rs` (`#[ignore]` observed-error sweep; then tighten the Task-2/3 tests to use `attn_rel_tol`)

**Interfaces:**
- Produces: `pub fn attn_rel_tol() -> f32` in `inferno_graph::tolerance` (no dtype arg — attention is f32-only).

- [ ] **Step 1: Add the observed-error sweep (ignored)** in `rig.rs`

```rust
/// Throwaway diagnostic (run manually): print the max relative error of the
/// attention kernel vs the std-exp interpreter across the shape distribution,
/// so `attn_rel_tol` is armed from data, not guessed. Run:
///   cargo test -p inferno-kernels --test rig observed_error_attention -- --ignored --nocapture
#[test]
#[ignore]
fn observed_error_attention() {
    let mut worst = 0f32;
    for seed in 0..20_000u64 {
        for &hd in &[8usize, 16, 64] {
            let (n_heads, n_kv_heads, head_dim) = (4usize, 2usize, hd);
            let kv_dim = n_kv_heads * head_dim;
            let seq_len = 16usize;
            let pos = (seed as usize) % 12;
            let mut kv = pseudo(seed, 2 * seq_len * kv_dim);
            let q = pseudo(seed ^ 1, n_heads * head_dim);
            let k = pseudo(seed ^ 2, kv_dim);
            let v = pseudo(seed ^ 3, kv_dim);
            let mut kc = kv[..seq_len * kv_dim].to_vec();
            let mut vc = kv[seq_len * kv_dim..].to_vec();
            let want = attn_oracle(&q, &k, &v, &mut kc, &mut vc, pos, kv_dim, n_heads, n_kv_heads, head_dim);
            let got = attn_kernel_scalar(&q, &k, &v, &mut kv, seq_len, pos, kv_dim, n_heads, n_kv_heads, head_dim);
            let scale = want.iter().fold(1f32, |m, x| m.max(x.abs())).max(1.0);
            for (g, w) in got.iter().zip(&want) {
                worst = worst.max((g - w).abs() / scale);
            }
        }
    }
    println!("observed_error_attention: max rel {worst:e}");
}
```

- [ ] **Step 2: Run the sweep and read the number**

Run: `cargo test -p inferno-kernels --test rig observed_error_attention -- --ignored --nocapture`
Expected: prints `observed_error_attention: max rel <N>e-<k>`. Record `<N>` — a degree-6 poly should land ~1e-6..1e-5 relative.

- [ ] **Step 3: Add `attn_rel_tol` armed at ~4× the observed max**

In `crates/inferno-graph/src/tolerance.rs`, append (fill the observed value into the doc comment and set the arm to ~4× it, rounded to a clean power-of-ten-ish constant — e.g. observed 2.3e-6 → `5e-6`):

```rust
/// Attention kernel (poly-exp softmax) vs the std-exp interpreter oracle,
/// relative to max(1, max|want|). The kernel's softmax uses a degree-6
/// minimax `exp` (crate `inferno-kernels::expf`) while the interpreter uses
/// `std::f32::exp`; this bounds that approximation plus lane-reduction
/// rounding. Scalar and AVX2 attention kernels are bit-identical to *each
/// other* (rig `attention_isa_variants_bitwise_equal`), so a single constant
/// covers both. Derived from the `observed_error_attention` sweep (rig,
/// #[ignore]); observed max <FILL: e.g. 2.3e-6> over 20000 seeds × hd∈{8,16,64}
/// on the dev Ryzen 9 3900 — armed at ~4×.
pub const fn attn_rel_tol() -> f32 {
    <FILL: e.g. 5e-6>
}
```

- [ ] **Step 4: Tighten the Task-2/3 correctness tests to use it**

In `rig.rs`, replace the placeholder `1e-4 * scale` in `attention_scalar_matches_interpreter` with:

```rust
        let tol = inferno_graph::tolerance::attn_rel_tol() * scale.max(1.0);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            prop_assert!((g - w).abs() <= tol, "elem {i}: got {g} want {w} (tol {tol})");
        }
```

- [ ] **Step 5: Run the tightened tests**

Run: `cargo test -p inferno-kernels --test rig attention_`
Expected: PASS with the derived tolerance. If it fails, the observed sweep undercounted — re-run Step 2 with more seeds and re-arm; never just widen the constant past ~4× without new data.

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-graph/src/tolerance.rs crates/inferno-kernels/tests/rig.rs
git commit -m "feat(graph): attn_rel_tol derived from observed kernel-vs-interpreter error"
```

---

## Task 6: Codegen — declare the attention symbols + `attention_symbol` helper

**Files:**
- Modify: `crates/inferno-codegen/src/llvm/mod.rs` (`declare_kernels`)
- Modify: `crates/inferno-codegen/src/loopir.rs` (`attention_symbol`)
- Test: `crates/inferno-codegen/src/llvm/mod.rs` (`#[cfg(test)]` IR-contains assertion)

**Interfaces:**
- Produces:
  - `pub fn attention_symbol(isa: inferno_kernels::KernelIsa) -> String` → `"inferno_attention_f32_<isa>"`.
  - The module declares `inferno_attention_f32_scalar` / `_avx2` with the 11-arg all-`ptr`/`i64` type.

- [ ] **Step 1: Write the failing IR test** (extend the existing `declare_kernels` test in `mod.rs`)

Find the test asserting `ir.contains("inferno_gemm_")` (around line 307) and add:

```rust
    assert!(ir.contains("inferno_attention_f32_scalar"));
    assert!(ir.contains("inferno_attention_f32_avx2"));
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p inferno-codegen --lib declare`
Expected: FAIL — IR lacks `inferno_attention_f32_*`.

- [ ] **Step 3: Declare the symbols + add the helper**

In `mod.rs` `declare_kernels`, after the quantize declarations, add:

```rust
        // void inferno_attention_f32_<isa>(ptr out, ptr q, ptr kv, ptr scores,
        //   i64 kv_base, i64 v_off, i64 pos, i64 kv_dim,
        //   i64 n_heads, i64 n_kv_heads, i64 head_dim)
        let attn_ty = void.fn_type(
            &[
                ptr.into(), ptr.into(), ptr.into(), ptr.into(),
                i64_t.into(), i64_t.into(), i64_t.into(), i64_t.into(),
                i64_t.into(), i64_t.into(), i64_t.into(),
            ],
            false,
        );
        for isa in ["scalar", "avx2"] {
            self.module.add_function(
                &format!("inferno_attention_f32_{isa}"),
                attn_ty,
                Some(Linkage::External),
            );
        }
```

In `loopir.rs`, next to `gemm_symbol`:

```rust
/// `inferno_attention_f32_{isa}`: the single f32 attention kernel (no dtype
/// axis). Selected by the same `KernelIsa` codegen uses for gemv/gemm.
pub fn attention_symbol(isa: inferno_kernels::KernelIsa) -> String {
    let isa = match isa {
        inferno_kernels::KernelIsa::Scalar => "scalar",
        inferno_kernels::KernelIsa::Avx2 => "avx2",
    };
    format!("inferno_attention_f32_{isa}")
}
```

(If `gemv_symbol` already maps `KernelIsa` to the `"scalar"`/`"avx2"` string, reuse that mapping rather than duplicating it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p inferno-codegen --lib declare`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-codegen/src/llvm/mod.rs crates/inferno-codegen/src/loopir.rs
git commit -m "feat(codegen): declare inferno_attention_f32 symbols + attention_symbol helper"
```

---

## Task 7: Codegen — `lower_attention` calls the kernel

**Files:**
- Modify: `crates/inferno-codegen/src/llvm/ops.rs` (`lower_attention`, ~1220–1368)
- Modify: `crates/inferno-codegen/src/snapshots/*loopir*` only if the loopir dump changes (it should NOT — `Step::Attention` is unchanged)
- Test: existing `crates/inferno-codegen/tests/differential.rs` (gated end-to-end; wired in Task 8)

**Interfaces:**
- Consumes: `attention_symbol` (Task 6); the declared symbols; the `Step::Attention` fields (`q,k,v,layer,n_heads,n_kv_heads,head_dim,out`); `self.plan.kv.kv_dim`, `self.plan.max_seq_len`, `self.isa` (the `KernelIsa` codegen already uses to pick gemv/gemm — confirm the field name at the `lower_gemv` call site).
- Produces: `lower_attention` emits a single `build_call` to the ISA-selected attention symbol instead of the scalar triple-loop.

**The call lowering.** The kernel is read-only over the KV cache (settled in Task 2's contract): codegen keeps its **existing** KV-append IR block (which reads the `k`/`v` arena rows and writes them into the cache at `pos` — already correct and cheap), then calls the kernel with the query-row pointer and the cache that now contains this token's k/v. The 11-arg ABI carries no k/v pointers. This is why `lower_attention` retains its append block and only replaces the read half.

- [ ] **Step 1: Rewrite `lower_attention`**

Replace the body from the `// --- Attention read ...` comment through the end of the per-head loop with a kernel call. Keep the existing KV-append IR block (lines ~1243–1257). After it:

```rust
        // Single-token attention read via the kernel (M4b.3). q is this
        // token's query row; k/v for this token were appended above, so the
        // kernel only reads. scores scratch is a per-call entry alloca.
        let scores = self.entry_alloca(self.f32_t.array_type(seq_len as u32), "scores");
        let q_ptr = self.arena_ptr(frame, self.row_base(frame, q), self.const_i64(0));
        let out_ptr = self.arena_ptr(frame, self.row_base(frame, out), self.const_i64(0));
        let sym = crate::loopir::attention_symbol(self.isa);
        let f = self.module.get_function(&sym).expect("attention kernel declared (Task 6)");
        let v_off = self.const_i64((seq_len * kv_dim) as i64);
        let kv_base_c = self.const_i64(kv_base as i64);
        self.builder
            .build_call(
                f,
                &[
                    out_ptr.into(),
                    q_ptr.into(),
                    frame.kv.into(),
                    scores.into(),
                    kv_base_c.into(),
                    v_off.into(),
                    frame.pos.into(),
                    kv_dim_c.into(),
                    self.const_i64(n_heads as i64).into(),
                    self.const_i64(n_kv_heads as i64).into(),
                    self.const_i64(head_dim as i64).into(),
                ],
                "attention",
            )
            .unwrap();
```

Delete the old scalar per-head loop (the `for h in 0..n_heads` block and its `scores`/`max`/`denom`/AV IR). Confirm `self.isa` is the right accessor (grep the `lower_gemv` call site: it derives the symbol from `pw.isa`; attention has no `pw`, so use the codegen's global ISA — likely `self.isa` or `self.target_isa()`; match whatever `declare`/gemv selection uses).

- [ ] **Step 2: Build the crate**

Run: `cargo build -p inferno-codegen`
Expected: compiles. Fix any borrow/type errors (e.g. `frame.kv` is already a `PointerValue`; `frame.pos` is an `IntValue`).

- [ ] **Step 3: Run the codegen unit + loopir snapshot tests**

Run: `cargo test -p inferno-codegen --lib`
Expected: PASS. The loopir snapshot is unchanged (`Step::Attention` is untouched). If `cargo insta` flags a diff, it means something structural changed unexpectedly — investigate, do not blind-accept (`cargo insta review`).

- [ ] **Step 4: Commit**

```bash
git add crates/inferno-codegen/src/llvm/ops.rs
git commit -m "feat(codegen): lower attention as a kernel call, not inline scalar IR"
```

---

## Task 8: Wire symbol retention + end-to-end differential

**Files:**
- Modify: `crates/inferno-codegen/tests/differential.rs` (`retain_kernel_symbols`)
- Modify: `crates/inferno-core/src/artifact.rs` (`ensure_kernels_linked`)
- Test: existing differential + artifact tests (the milestone's correctness gate)

**Interfaces:**
- Consumes: `inferno_kernels::inferno_attention_f32_scalar` / `_avx2`.

- [ ] **Step 1: Extend both retention lists**

In `crates/inferno-codegen/tests/differential.rs` `retain_kernel_symbols`, add before the `inferno_pool` lines:

```rust
    p(inferno_kernels::inferno_attention_f32_scalar as *const ());
    p(inferno_kernels::inferno_attention_f32_avx2 as *const ());
```

In `crates/inferno-core/src/artifact.rs` `ensure_kernels_linked`, add the same two lines before the `inferno_pool` lines.

- [ ] **Step 2: Run the codegen differential (the gate)**

Run: `cargo test -p inferno-codegen --test differential`
Expected: PASS — `differential_tiny_gguf`, `differential_tiny_mlx`, `differential_tiny_bias`, `prefill_tiling_is_bit_invariant_to_tile_size`, `profiling_does_not_change_logits` all green. Last-token logits match the interpreter within `logits_abs_tol` (1e-2 Q8_0). **If red:** the kernel diverges from the interpreter beyond budget — debug the kernel (Task 2–3), never loosen `logits_abs_tol`.

- [ ] **Step 3: Run the artifact differential**

Run: `cargo test -p inferno-core --test artifact`
Expected: PASS.

- [ ] **Step 4: Full workspace test + lint**

Run: `mise run test && mise run lint`
Expected: PASS. (`mise run differential` if defined runs the nightly diff harness — run it too if present.)

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-codegen/tests/differential.rs crates/inferno-core/src/artifact.rs
git commit -m "feat(core,codegen): retain attention kernel symbols for dlopen resolution"
```

---

## Task 9: Attention µbench (criterion)

**Files:**
- Create: `crates/inferno-kernels/benches/attention.rs`
- Modify: `crates/inferno-kernels/Cargo.toml` (`[[bench]]`)

**Interfaces:**
- Consumes: the two attention symbols; the pinned model geometry (`head_dim=64`, `n_heads=14`, `n_kv_heads=2`).

- [ ] **Step 1: Add the bench + registration**

Create `crates/inferno-kernels/benches/attention.rs` (follow `benches/gemv.rs`'s `criterion_group!` structure):

```rust
//! Attention kernel µbench (M4b.3): scalar vs avx2 over the pinned model's
//! head geometry at representative causal horizons, so the SIMD win is
//! visible per-position. Throughput unit: elements (n_heads * head_dim per
//! call). Numbers only meaningful from `mise run bench-kernels` on quiet HW.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use inferno_kernels::{attention_kernel, attention_reference};
use inferno_target::Isa;

fn bench_attention(c: &mut Criterion) {
    let (n_heads, n_kv_heads, head_dim) = (14usize, 2usize, 64usize);
    let kv_dim = n_kv_heads * head_dim;
    let seq_len = 512usize;
    let mut group = c.benchmark_group("attention");
    for &pos in &[15usize, 127, 511] {
        let mut kv = vec![0.1f32; 2 * seq_len * kv_dim];
        let q = vec![0.05f32; n_heads * head_dim];
        let mut out = vec![0f32; n_heads * head_dim];
        let mut scores = vec![0f32; seq_len];
        group.throughput(Throughput::Elements((n_heads * head_dim) as u64));
        for (name, f) in [
            ("scalar", attention_reference()),
            ("avx2", attention_kernel(Isa::X86_64v3)),
        ] {
            group.bench_with_input(BenchmarkId::new(name, pos), &pos, |b, &pos| {
                b.iter(|| unsafe {
                    f(
                        out.as_mut_ptr(), q.as_ptr(), kv.as_mut_ptr(), scores.as_mut_ptr(),
                        0, seq_len * kv_dim, pos, kv_dim, n_heads, n_kv_heads, head_dim,
                    )
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_attention);
criterion_main!(benches);
```

In `crates/inferno-kernels/Cargo.toml`, add:

```toml
[[bench]]
name = "attention"
harness = false
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo bench -p inferno-kernels --no-run`
Expected: compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/inferno-kernels/benches/attention.rs crates/inferno-kernels/Cargo.toml
git commit -m "bench(kernels): attention criterion group (scalar vs avx2, pinned geometry)"
```

---

## Task 10: Manual measurement + recorded amendment (exit criterion)

**Files:**
- Modify: `docs/superpowers/specs/2026-07-07-m4b3-vectorized-attention-design.md` (Amendments)

This task is the exit criterion. All runs inside `devenv shell`, release build, on the quiet machine, pinned Q8_0 model, threads=1.

- [ ] **Step 1: Kernel µbench numbers**

Run: `mise run bench-kernels` (attention group). Record scalar vs avx2 `Melem/s` per `pos` in Amendments (same rules as M2/M4b.2 — the ratio is the meaningful figure on the quota'd box).

- [ ] **Step 2: `--profile` capture (the share check)**

Run:
```bash
mise run bench -- /home/dev/.cache/inferno-tests/qwen2.5-0.5b-instruct-q8_0.gguf --threads 1 --json
```
and a `--profile` capture:
```bash
inferno run --profile --threads 1 --max-tokens 64 <model>   # via the mise/devenv path used in M4b.2
```
Record both the t=1 bench table+json and the prefill+decode profile tables verbatim in Amendments.

- [ ] **Step 3: Evaluate the exit criterion**

Compute and record:
- Attention's **prefill cycle share** (target: **< 45%**, from 70.6%).
- pp and tg ratios vs llama.cpp t=1 (target: **both improve** over M4b.2's 0.55× / 0.43×).

- **If met:** note it; the residual 0.70× prefill decision passes to the register-blocked GEMM follow-up (a separate plan) which now re-profiles on top of vectorized attention.
- **If attention share is still the majority:** record it and open a scoped amendment — the vectorization underperformed and the next lever (prefill panel/flash blocking) is indicated. Do **not** silently start that work.

- [ ] **Step 4: Commit the amendment**

```bash
git add docs/superpowers/specs/2026-07-07-m4b3-vectorized-attention-design.md
git commit -m "bench: record M4b.3 vectorized-attention data point + profile share"
```

---

## Self-Review

**Spec coverage:**
- Kernel family `inferno_attention_f32_{scalar,avx2}`, f32-only, ISA-dispatched → Tasks 2, 3, 4. ✓
- Vectorized polynomial `exp`, identical scalar/AVX2 → Task 1. ✓
- `lower_attention` declares + calls the symbol instead of scalar IR → Tasks 6, 7. ✓
- Single-threaded (no `par_*`) → the ABI/call has no dispatcher; Task 7. ✓
- scalar-vs-AVX2 bit-identity → Task 3 (`attention_isa_variants_bitwise_equal`). ✓
- kernel-vs-interpreter `attn_rel_tol`, derived from observed sweep → Task 5. ✓
- End-to-end `logits_abs_tol` unchanged + symbol retention → Task 8. ✓
- Interpreter `ops::attention` unchanged → used only as oracle in Tasks 2/5. ✓
- µbench + recorded t=1 data point + <45% share exit criterion → Tasks 9, 10. ✓
- Out-of-scope (parallel attention, prefill panel blocking, F16 KV, AVX-512, GEMM escalation) → not touched; deferred in Task 10 Step 3. ✓

**Placeholder scan:** The only intentional fill-ins are the observed error value and the derived `attn_rel_tol` constant in Task 5 Step 3 (`<FILL: ...>`) — these are *required* to come from the Step-2 measurement, not guessable, and the plan says exactly how to obtain and arm them. No "TBD"/"handle edge cases"/"similar to Task N": every code step shows the code.

**Type consistency:** `AttnFn` (11-arg: `out:*mut f32, q:*const f32, kv:*mut f32, scores:*mut f32` + 7×`usize`) is identical across the `attention.rs` symbols (Tasks 2–3), the registry type (Task 4), the LLVM `attn_ty` (4 ptr + 7 i64, Task 6), the codegen `build_call` args (Task 7), and the µbench call (Task 9). `attention_symbol`/`attention_kernel`/`attention_reference`/`attn_rel_tol` names are used consistently. The kernel is **read-only over the KV cache** in every layer — the caller (rig driver, µbench, and codegen's existing append IR) writes this token's k/v at `pos` before the call, so `q` is the query row only and no ABI carries k/v pointers. This contract is stated once in Task 2 and unchanged through Tasks 3, 7, 9.
