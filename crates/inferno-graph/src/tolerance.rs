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
/// Quant paths are dominated by on-the-fly activation quantization (8-bit
/// blocks, ~0.4% per element); weight quantization error cancels because
/// both sides consume identical quantized weights. Initial values; tuned
/// against the observed error distributions printed by the rig's ignored
/// `observed_error_*` diagnostics (see AGENTS.md tolerance rule).
pub fn gemv_rel_tol(dtype: &DType) -> f32 {
    match dtype {
        DType::F32 => 1e-6, // fma-vs-mul+add rounding only
        DType::Q8_0 | DType::Q4_K => 2e-2,
        // No M2 kernels exist for these; the rig never asks.
        DType::F16 | DType::BF16 | DType::Unsupported(_) => 0.0,
    }
}
