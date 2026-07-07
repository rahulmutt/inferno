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
        attn_core_scalar(
            out, q, kv, scores, kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim,
        );
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
        let max = scores[..visible]
            .iter()
            .fold(f32::NEG_INFINITY, |m, v| m.max(*v));
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
