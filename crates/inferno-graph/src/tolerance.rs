//! The single home for numeric comparison constants (spec §Scalar
//! interpreter). Every test layer — quant round-trips here, M2 kernel
//! properties, M3 compiled-vs-reference differentials — imports these.

use inferno_formats::DType;

/// pack→dequant max error, relative to the block's max |value|.
pub fn roundtrip_rel_tol(dtype: &DType) -> f32 {
    match dtype {
        DType::F32 => 0.0,
        DType::F16 => 1e-3,
        DType::BF16 => 8e-3,
        DType::Q8_0 => 8e-3,
        DType::Q4_K => 1.1e-1, // simple min/max reference quantizer, not ggml's optimizer
        DType::Unsupported(_) => 0.0,
    }
}

/// Absolute logit tolerance when comparing two implementations of a model
/// whose widest weight dtype is `dtype` (spec: ~1e-2 on quantized paths).
pub fn logits_abs_tol(dtype: &DType) -> f32 {
    match dtype {
        DType::Q8_0 | DType::Q4_K => 1e-2,
        _ => 1e-4,
    }
}

/// Teacher-forced differential: a position where our top-2 logit gap is
/// below this counts as a genuine tie, not a mismatch. Tuned against the
/// gap distributions the nightly diff reports (see AGENTS.md).
/// Observed so far (Qwen2.5-0.5B-Instruct Q8_0, first nightly, 2026-07-05):
/// 64 checked, 63 matched, 1 tie, min top-2 gap 0.0004.
pub const LOGIT_TIE_EPSILON: f32 = 0.05;

/// Kernel-GEMV vs dequant+reference-matmul, relative to max(1, max|y_ref|).
///
/// Comparison contract (`crates/inferno-kernels/tests/rig.rs`): for the
/// quantized dtypes, the oracle is fed the SAME activations the kernel
/// consumes — decoded back from the kernel's own q8a/q8k quantization
/// buffer (`x_hat`), not the raw f32 row. Both sides therefore see
/// identical quantized weights AND identical quantized activations; the
/// only remaining discrepancy is floating-point accumulation-order/fma
/// rounding, which is what these constants bound.
///
/// This replaces an earlier design where the oracle consumed raw f32
/// activations and the kernel quantized them on the fly, so the comparison
/// also measured activation-quantization noise. That noise has a heavy
/// tail dominated by near-cancelling, low-block-count shapes (`rows=1`,
/// few blocks) where `assert_close`'s `max(1, max|want|)` normalization
/// turns a small absolute quantization error into a large relative one;
/// a 2026-07-05 investigation on the dev Ryzen 9 3900 measured this
/// end-to-end noise at up to 3.37e-2 (Q8_0, 2k seeds) climbing to 6.25e-2
/// (Q8_0, 500k seeds) and up to ~0.142 (Q4_K tail, 3M seeds) — too large
/// for any single constant to both stay under a practical flake budget and
/// still catch a real ~5-20% layout/scale bug. Making the oracle consume
/// the kernel's own quantized activations eliminates that noise term
/// entirely rather than trying to bound it.
///
/// Observed max (release build, Ryzen 9 3900, 2026-07-05), oracle-on-x_hat,
/// swept over the property tests' shape distribution (rows 1..20, worst at
/// rows=1; `nb` 1..5 for Q8_0, `nsb` 1..3 for Q4_K; 20000 seeds per shape,
/// via a throwaway sweep deleted before commit):
/// - Q8_0: 2.384e-6 (rows=1, k=128) → arm set to ~4x → 1e-5.
/// - Q4_K: 9.239e-6 (rows=1, k=512) → arm set to ~4x → 4e-5.
///
/// Q4_K's max is ~3.9x Q8_0's (>2x), so the arms are split rather than
/// sharing one constant.
pub fn gemv_rel_tol(dtype: &DType) -> f32 {
    match dtype {
        DType::F32 => 1e-6, // fma-vs-mul+add rounding only
        DType::Q8_0 => 1e-5,
        DType::Q4_K => 4e-5,
        // No M2 kernels exist for these; the rig never asks.
        DType::F16 | DType::BF16 | DType::Unsupported(_) => 0.0,
    }
}
