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

pub mod act;
mod buf;
mod error;
pub mod f32k;

pub use buf::AlignedBuf;
pub use error::{KernelError, Result};
pub use f32k::{inferno_gemv_f32_rs8_avx2, inferno_gemv_f32_rs8_scalar};

/// Rows per packed strip: every rs8 layout interleaves 8 rows.
pub const STRIP: usize = 8;

/// Which implementation of a kernel to run. Scalar is always available; SIMD
/// variants only where the CPU supports them (the registry enforces this —
/// the single place runtime feature detection happens).
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
