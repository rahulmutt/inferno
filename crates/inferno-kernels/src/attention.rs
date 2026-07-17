//! Causal GQA attention as a C-ABI kernel (f32-only, ISA-dispatched).
//! Mirrors `inferno_graph::ops::attention` op-for-op, except the softmax
//! `exp` is the shared polynomial (`crate::expf`), so the compiled path is
//! bounded against the std-exp interpreter by `attn_rel_tol`, and the
//! scalar and AVX2 kernels are bit-identical to each other (shared poly +
//! reduction order). One call = one query token; the *_hspan variants
//! (M4b.11) run the same per-head math over a caller-chosen head range for
//! the pool's decode head-sharding.

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
        attn_core_scalar(
            out, q, kv, scores, kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim, 0,
            n_heads,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn attn_core_scalar(
    // `out`/`q` are span-local: extent `(h_end-h_start)*head_dim`, indexed
    // below via `h - h_start` so no lane's slice overlaps another's.
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
    h_start: usize,
    h_end: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let group = n_heads / n_kv_heads;
    let kreg = kv_base;
    let vreg = kv_base + v_off;
    let visible = pos + 1;
    for h in h_start..h_end {
        let g = h / group;
        let hl = h - h_start; // local index into the span-local out/q slices
        let qh = &q[hl * head_dim..][..head_dim];
        // scores[t] = dot(qh, kcache[t,g]) * scale, in the SAME 8-lane
        // partitioned order the AVX2 kernel reduces (see `dot8`/`reduce8`).
        for (t, sc) in scores.iter_mut().enumerate().take(visible) {
            let kbase = kreg + t * kv_dim + g * head_dim;
            *sc = dot8(qh, &kv[kbase..kbase + head_dim]) * scale;
        }
        let max = scores[..visible]
            .iter()
            .fold(f32::NEG_INFINITY, |m, v| m.max(*v));
        // exp + denom, mirroring the AVX2 loop EXACTLY: reduce each block of 8
        // exp values with the same `reduce8` tree, then a scalar tail. Summing
        // the denom sequentially here would diverge bitwise from AVX2.
        let mut denom = 0f32;
        let mut t = 0;
        while t + 8 <= visible {
            let mut lanes = [0f32; 8];
            for (l, lane) in lanes.iter_mut().enumerate() {
                let e = expf_scalar(scores[t + l] - max);
                scores[t + l] = e;
                *lane = e;
            }
            denom += reduce8(lanes);
            t += 8;
        }
        while t < visible {
            let e = expf_scalar(scores[t] - max);
            scores[t] = e;
            denom += e;
            t += 1;
        }
        let oh = &mut out[hl * head_dim..][..head_dim];
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

/// Query-blocked scalar attention (M4b.14). Computes query rows `[0,
/// m_block)` — row `r` at position `pos0 + r` — reusing each visible K and
/// V vector across the block's rows (streamed once per head per block
/// instead of once per token). Blocking only reorders the query axis, so
/// each row's arithmetic is bit-for-bit the per-token kernel's: same
/// `dot8`/`reduce8` order, same block-of-8 `expf_scalar` softmax + scalar
/// tail, same ascending-`t` `mul_add` V-accumulation.
///
/// # Safety
/// - `out`/`q` valid for `(m_block-1)*{out,q}_stride + n_heads*head_dim` f32.
/// - `scores` valid for `m_block * (pos0 + m_block)` f32 (scratch).
/// - `kv` valid for the K/V regions, holding every position `< pos0+m_block`.
/// - `pos0 + m_block <= seq_len`; `m_block >= 1`; `head_dim` a multiple of 8.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_attention_f32_scalar_qblock(
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    scores: *mut f32,
    kv_base: usize,
    v_off: usize,
    pos0: usize,
    m_block: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_stride: usize,
    out_stride: usize,
) {
    // SAFETY: contract above; delegate to a safe-slice core.
    unsafe {
        let s = pos0 + m_block;
        let q_extent = (m_block - 1) * q_stride + n_heads * head_dim;
        let out_extent = (m_block - 1) * out_stride + n_heads * head_dim;
        let q = std::slice::from_raw_parts(q, q_extent);
        let out = std::slice::from_raw_parts_mut(out, out_extent);
        let scores = std::slice::from_raw_parts_mut(scores, m_block * s);
        let kv = std::slice::from_raw_parts(kv, kv_base + 2 * v_off);
        attn_core_scalar_qblock(
            out, q, kv, scores, kv_base, v_off, pos0, m_block, kv_dim, n_heads, n_kv_heads,
            head_dim, q_stride, out_stride,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn attn_core_scalar_qblock(
    out: &mut [f32],
    q: &[f32],
    kv: &[f32],
    scores: &mut [f32],
    kv_base: usize,
    v_off: usize,
    pos0: usize,
    m_block: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_stride: usize,
    out_stride: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let group = n_heads / n_kv_heads;
    let kreg = kv_base;
    let vreg = kv_base + v_off;
    let s = pos0 + m_block; // per-row scores stride = max visible over the block
    for h in 0..n_heads {
        let g = h / group;
        // scores pass: each visible K vector loaded once, reused across rows.
        for t in 0..s {
            let kb = kreg + t * kv_dim + g * head_dim;
            let kt = &kv[kb..kb + head_dim];
            for r in 0..m_block {
                if t <= pos0 + r {
                    let qh = &q[r * q_stride + h * head_dim..][..head_dim];
                    // Same 8-lane partition order as the per-token kernel.
                    scores[r * s + t] = dot8(qh, kt) * scale;
                }
            }
        }
        // softmax + in-place normalize per row (denom is a scalar). Mirrors
        // the per-token loop EXACTLY: block-of-8 reduce8 denom + scalar tail.
        for r in 0..m_block {
            let visible = pos0 + r + 1;
            let row = &mut scores[r * s..r * s + s];
            let max = row[..visible]
                .iter()
                .fold(f32::NEG_INFINITY, |m, v| m.max(*v));
            let mut denom = 0f32;
            let mut t = 0;
            while t + 8 <= visible {
                let mut lanes = [0f32; 8];
                for (l, lane) in lanes.iter_mut().enumerate() {
                    let e = expf_scalar(row[t + l] - max);
                    row[t + l] = e;
                    *lane = e;
                }
                denom += reduce8(lanes);
                t += 8;
            }
            while t < visible {
                let e = expf_scalar(row[t] - max);
                row[t] = e;
                denom += e;
                t += 1;
            }
            // Normalize now; w/denom is the same value the per-token kernel
            // computes lazily in its output loop.
            for w in row[..visible].iter_mut() {
                *w /= denom;
            }
        }
        // output pass: each visible V vector loaded once, reused across rows.
        // Zero every row's head-span first, then accumulate in ascending t.
        for r in 0..m_block {
            let ob = r * out_stride + h * head_dim;
            out[ob..ob + head_dim].fill(0.0);
        }
        for t in 0..s {
            let vb = vreg + t * kv_dim + g * head_dim;
            for r in 0..m_block {
                if t <= pos0 + r {
                    let wn = scores[r * s + t];
                    let ob = r * out_stride + h * head_dim;
                    for d in 0..head_dim {
                        out[ob + d] = wn.mul_add(kv[vb + d], out[ob + d]);
                    }
                }
            }
        }
    }
}

/// Head-span variant (M4b.11): identical per-head math to
/// [`inferno_attention_f32_scalar`] restricted to heads `[h_start, h_end)`,
/// so any tiling of `0..n_heads` reproduces the whole call bit-for-bit.
/// `n_heads` stays the FULL head count (the GQA group divisor).
///
/// # Safety
/// As [`inferno_attention_f32_scalar`], plus `h_start <= h_end <= n_heads`.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_attention_f32_scalar_hspan(
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
    h_start: usize,
    h_end: usize,
) {
    // SAFETY: contract above, restricted to this lane's head span. `out`/`q`
    // are sliced to ONLY `[h_start*head_dim, h_end*head_dim)` (not the full
    // `n_heads*head_dim` extent), so concurrent lanes calling this function
    // with the same `out`/`q` base pointers over disjoint `[h_start, h_end)`
    // ranges never construct overlapping `&mut`/`&` slices — each lane's
    // slice is backed by disjoint memory. `kv` remains a shared read-only
    // slice over the full region (unchanged): the whole-call contract holds
    // KV read-only for the call's duration, so sharing it across lanes is
    // sound as long as no `&mut` to it is ever created, which we don't do.
    unsafe {
        let span = h_end - h_start;
        let q = std::slice::from_raw_parts(q.add(h_start * head_dim), span * head_dim);
        let out = std::slice::from_raw_parts_mut(out.add(h_start * head_dim), span * head_dim);
        let scores = std::slice::from_raw_parts_mut(scores, pos + 1);
        let kv = std::slice::from_raw_parts(kv, kv_base + 2 * v_off);
        attn_core_scalar(
            out, q, kv, scores, kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim,
            h_start, h_end,
        );
    }
}

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

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// # Safety
/// As [`inferno_attention_f32_scalar`], plus the running CPU has AVX2+FMA.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_attention_f32_avx2(
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
    // SAFETY: forwarding the contract for the full head range.
    unsafe {
        attn_core_avx2(
            out, q, kv, scores, kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim, 0,
            n_heads,
        );
    }
}

/// Head-span variant (M4b.11); see [`inferno_attention_f32_scalar_hspan`].
///
/// # Safety
/// As [`inferno_attention_f32_avx2`], plus `h_start <= h_end <= n_heads`.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_attention_f32_avx2_hspan(
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
    h_start: usize,
    h_end: usize,
) {
    // SAFETY: forwarding the contract for the caller's head range.
    unsafe {
        attn_core_avx2(
            out, q, kv, scores, kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim,
            h_start, h_end,
        );
    }
}

/// The AVX2 per-head loop, bounds-parameterized (M4b.11). Body is the
/// former `inferno_attention_f32_avx2` verbatim except `for h in
/// h_start..h_end`.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attn_core_avx2(
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
    h_start: usize,
    h_end: usize,
) {
    // SAFETY: contract as the public symbols; head_dim is a mult of 8.
    unsafe {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group = n_heads / n_kv_heads;
        let kreg = kv_base;
        let vreg = kv_base + v_off;
        // Read-only: caller already appended this token's k/v at `pos`.
        let visible = pos + 1;
        for h in h_start..h_end {
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
            for t in 0..visible {
                max = max.max(*scores.add(t));
            }
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
    }
}

/// Horizontal sum matching `reduce8`'s tree: (lo+hi) halves then pairwise.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum8(v: __m256) -> f32 {
    // SAFETY: avx2 enabled by the fn's target_feature; the intrinsics below
    // are safe to call in that context.
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(lo, hi); // [0+4,1+5,2+6,3+7]
    let sh = _mm_movehl_ps(s, s); // [2+6,3+7,..]
    let s2 = _mm_add_ps(s, sh); // [(0+4)+(2+6),(1+5)+(3+7),..]
    let s3 = _mm_add_ss(s2, _mm_shuffle_ps::<0x55>(s2, s2));
    _mm_cvtss_f32(s3)
}
