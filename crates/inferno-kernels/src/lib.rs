//! Hand-tuned CPU microkernels behind a fixed C ABI (spec §inferno-kernels).
//! One `#[unsafe(no_mangle)] extern "C"` symbol per (op × dtype × ISA);
//! M3-generated code calls these by name. Weight packing is safe Rust.
//!
//! Numeric contract: integer block dots are exact and every f32 operation
//! happens in the same order with the same fusing in every ISA variant, so
//! variants are **bit-identical** — the rig asserts exact equality.
//!
//! Activation-side quantization formats (q8a/q8k) live here and never in
//! `inferno_formats::DType`: they are a kernel implementation detail, not a
//! weight-file dtype (spec boundary rule).

// The SIMD kernels are written directly against x86-64 AVX2/FMA intrinsics;
// portable NEON support is a v2 (NEON) milestone. Fail loudly on other targets
// rather than silently compiling a scalar-only, untested build.
#[cfg(not(target_arch = "x86_64"))]
compile_error!("inferno-kernels is x86-64-only until the v2 NEON milestone");

pub mod act;
mod attention;
mod buf;
mod error;
mod expf;
pub mod f32k;
mod pf;
pub mod q4_k;
pub mod q8_0;
pub mod registry;

#[cfg(target_arch = "x86_64")]
pub use attention::inferno_attention_f32_avx2;
#[cfg(target_arch = "x86_64")]
pub use attention::inferno_attention_f32_avx2_hspan;
pub use attention::inferno_attention_f32_scalar;
pub use attention::inferno_attention_f32_scalar_hspan;
pub use attention::inferno_attention_f32_scalar_qblock;
pub use buf::AlignedBuf;
pub use error::{KernelError, Result};
pub use f32k::{
    inferno_gemm_f32_rs8_avx2, inferno_gemm_f32_rs8_scalar, inferno_gemv_f32_rs8_avx2,
    inferno_gemv_f32_rs8_scalar,
};
pub use q4_k::{
    inferno_gemm_q4_k_rs8_avx2, inferno_gemm_q4_k_rs8_scalar, inferno_gemv_q4_k_rs8_avx2,
    inferno_gemv_q4_k_rs8_scalar,
};
pub use q8_0::{
    inferno_gemm_q8_0_rs8_avx2, inferno_gemm_q8_0_rs8_scalar, inferno_gemv_q8_0_rs8_avx2,
    inferno_gemv_q8_0_rs8_scalar,
};
pub use registry::{
    AttnFn, KernelSet, attention_kernel, attention_reference, kernels_for, reference_kernels,
};

/// Rows per packed strip: every rs8 layout interleaves 8 rows.
pub const STRIP: usize = 8;

/// Upper bound on `k` (and `rows`) accepted by the safe validation wrappers.
/// The public length helpers (`packed_len_*`, `q8a_len`, `q8k_len`) multiply
/// dimensions unchecked; for absurd inputs those products wrap, which would let
/// a tiny buffer pass a `KernelSet` equality check and cause an OOB read in the
/// kernel. Rejecting any dimension above this bound keeps every length
/// computation (worst case `rows_padded * k * 4` for F32) well under `2^60`, so
/// no product can overflow `usize`. `2^28` (~268M) is far above any real tensor
/// dimension yet leaves ~36 bits of headroom.
pub const MAX_K: usize = 1 << 28;

/// Which implementation of a kernel to run. Scalar is always available; SIMD
/// variants only where the CPU supports them. Runtime CPU-feature detection for
/// *kernel selection* lives in the registry (`kernels_for`); the safe execution
/// wrappers additionally guard each dispatch with `available()` so no safe call
/// can reach an AVX2 symbol on an unsupported CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelIsa {
    Scalar,
    Avx2,
}

impl KernelIsa {
    pub fn available(self) -> bool {
        match self {
            KernelIsa::Scalar => true,
            KernelIsa::Avx2 => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
        }
    }

    /// All variants runnable on this CPU (rig helpers iterate this).
    pub fn all_available() -> Vec<KernelIsa> {
        [KernelIsa::Scalar, KernelIsa::Avx2]
            .into_iter()
            .filter(|i| i.available())
            .collect()
    }
}
