//! Kernel selection + the validated safe wrappers every non-codegen caller
//! uses. The raw `extern "C"` symbols stay unchecked (M3 codegen guarantees
//! their contracts by construction); tests, benches, and the M3 planner go
//! through `KernelSet`, which validates lengths, block multiples, and row
//! ranges. This is also the single place runtime CPU-feature detection
//! happens: `kernels_for` refuses to hand out kernels the CPU can't run.

use inferno_formats::DType;
use inferno_target::Isa;

use crate::{AlignedBuf, KernelError, KernelIsa, Result, act, f32k, q4_k, q8_0};

type PackFn = fn(&[u8], usize, usize) -> Result<AlignedBuf>;
type PackedLenFn = fn(usize, usize) -> usize;
type ActLenFn = fn(usize) -> usize;
type QuantFn = unsafe extern "C" fn(*const f32, *mut u8, usize);
type GemvFn = unsafe extern "C" fn(*mut f32, *const u8, *const u8, usize, usize, usize);

pub struct KernelSet {
    pub dtype: DType,
    pub isa: KernelIsa,
    /// Packed-layout identifier; part of the symbol names ("rs8").
    pub layout: &'static str,
    wblock: usize,
    pack: PackFn,
    packed_len: PackedLenFn,
    act_len: ActLenFn,
    quantize: Option<QuantFn>, // None: activations are raw f32 LE bytes
    gemv: GemvFn,
}

impl KernelSet {
    pub fn packed_len(&self, rows: usize, k: usize) -> usize {
        (self.packed_len)(rows, k)
    }

    pub fn act_len(&self, k: usize) -> usize {
        (self.act_len)(k)
    }

    pub fn pack(&self, bytes: &[u8], rows: usize, k: usize) -> Result<AlignedBuf> {
        (self.pack)(bytes, rows, k)
    }

    pub fn quantize_row(&self, x: &[f32]) -> Result<Vec<u8>> {
        if x.is_empty() || !x.len().is_multiple_of(self.wblock) {
            return Err(KernelError::BadK {
                k: x.len(),
                block: self.wblock,
            });
        }
        match self.quantize {
            Some(f) => {
                let mut out = vec![0u8; (self.act_len)(x.len())];
                // SAFETY: x/out lengths validated against the symbol contract;
                // SIMD sets exist only when the CPU supports them.
                unsafe { f(x.as_ptr(), out.as_mut_ptr(), x.len()) };
                Ok(out)
            }
            None => Ok(x.iter().flat_map(|v| v.to_le_bytes()).collect()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemv(
        &self,
        y: &mut [f32],
        xq: &[u8],
        w: &AlignedBuf,
        rows: usize,
        k: usize,
        row_start: usize,
        row_end: usize,
    ) -> Result<()> {
        if k == 0 || !k.is_multiple_of(self.wblock) {
            return Err(KernelError::BadK {
                k,
                block: self.wblock,
            });
        }
        if y.len() != rows {
            return Err(KernelError::SizeMismatch {
                what: "output rows (f32 count)",
                got: y.len(),
                expected: rows,
            });
        }
        if row_start > row_end || row_end > rows {
            return Err(KernelError::BadRowRange {
                row_start,
                row_end,
                rows,
            });
        }
        if xq.len() != (self.act_len)(k) {
            return Err(KernelError::SizeMismatch {
                what: "activation buffer bytes",
                got: xq.len(),
                expected: (self.act_len)(k),
            });
        }
        if w.len() != (self.packed_len)(rows, k) {
            return Err(KernelError::SizeMismatch {
                what: "packed weight bytes",
                got: w.len(),
                expected: (self.packed_len)(rows, k),
            });
        }
        // SAFETY: every pointer/length/alignment precondition of the symbol
        // was validated above; AlignedBuf guarantees 32-byte alignment.
        unsafe {
            (self.gemv)(
                y.as_mut_ptr(),
                xq.as_ptr(),
                w.as_ptr(),
                k,
                row_start,
                row_end,
            )
        };
        Ok(())
    }
}

fn set(dtype: &DType, isa: KernelIsa) -> Option<KernelSet> {
    let s = match dtype {
        DType::F32 => KernelSet {
            dtype: DType::F32,
            isa,
            layout: "rs8",
            wblock: 1,
            pack: f32k::pack_f32_rs8,
            packed_len: f32k::packed_len_f32_rs8,
            act_len: |k| k * 4,
            quantize: None,
            gemv: match isa {
                KernelIsa::Scalar => f32k::inferno_gemv_f32_rs8_scalar,
                KernelIsa::Avx2 => f32k::inferno_gemv_f32_rs8_avx2,
            },
        },
        DType::Q8_0 => KernelSet {
            dtype: DType::Q8_0,
            isa,
            layout: "rs8",
            wblock: act::Q8A_BLOCK,
            pack: q8_0::pack_q8_0_rs8,
            packed_len: q8_0::packed_len_q8_0_rs8,
            act_len: act::q8a_len,
            quantize: Some(match isa {
                KernelIsa::Scalar => act::inferno_quantize_row_q8a_scalar,
                KernelIsa::Avx2 => act::inferno_quantize_row_q8a_avx2,
            }),
            gemv: match isa {
                KernelIsa::Scalar => q8_0::inferno_gemv_q8_0_rs8_scalar,
                KernelIsa::Avx2 => q8_0::inferno_gemv_q8_0_rs8_avx2,
            },
        },
        DType::Q4_K => KernelSet {
            dtype: DType::Q4_K,
            isa,
            layout: "rs8",
            wblock: act::Q8K_BLOCK,
            pack: q4_k::pack_q4_k_rs8,
            packed_len: q4_k::packed_len_q4_k_rs8,
            act_len: act::q8k_len,
            quantize: Some(match isa {
                KernelIsa::Scalar => act::inferno_quantize_row_q8k_scalar,
                KernelIsa::Avx2 => act::inferno_quantize_row_q8k_avx2,
            }),
            gemv: match isa {
                KernelIsa::Scalar => q4_k::inferno_gemv_q4_k_rs8_scalar,
                KernelIsa::Avx2 => q4_k::inferno_gemv_q4_k_rs8_avx2,
            },
        },
        DType::F16 | DType::BF16 | DType::Unsupported(_) => return None,
    };
    Some(s)
}

/// The SIMD kernel set for a target ISA level, or None if this dtype has no
/// kernels or the *running* CPU can't execute them (spec: the registry
/// refuses; scalar fallbacks come from [`reference_kernels`]).
pub fn kernels_for(dtype: &DType, isa: Isa) -> Option<KernelSet> {
    let kisa = match isa {
        // v4 ⊇ v3: no v4-specific kernels exist in M2, v4 CPUs run the AVX2 set.
        Isa::X86_64v3 | Isa::X86_64v4 => KernelIsa::Avx2,
    };
    if !kisa.available() {
        return None;
    }
    set(dtype, kisa)
}

/// Scalar kernels — always runnable, the portable fallback and debug aid.
pub fn reference_kernels(dtype: &DType) -> Option<KernelSet> {
    set(dtype, KernelIsa::Scalar)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelIsa;
    use inferno_formats::{DType, quant};
    use inferno_target::Isa;

    fn pseudo(mut seed: u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    fn selection_rules() {
        for dtype in [DType::F32, DType::Q8_0, DType::Q4_K] {
            assert!(reference_kernels(&dtype).is_some(), "{dtype:?}");
            if KernelIsa::Avx2.available() {
                let s = kernels_for(&dtype, Isa::X86_64v3).unwrap();
                assert_eq!(s.isa, KernelIsa::Avx2);
                assert_eq!(s.layout, "rs8");
                // v4 CPUs run v3 kernels.
                assert!(kernels_for(&dtype, Isa::X86_64v4).is_some());
            }
        }
        for dtype in [DType::F16, DType::BF16, DType::Unsupported("x".into())] {
            assert!(reference_kernels(&dtype).is_none(), "{dtype:?}");
            assert!(kernels_for(&dtype, Isa::X86_64v3).is_none(), "{dtype:?}");
        }
    }

    #[test]
    fn f32_quantize_row_is_le_bytes() {
        let s = reference_kernels(&DType::F32).unwrap();
        let x = [1.5f32, -2.0];
        let b = s.quantize_row(&x).unwrap();
        assert_eq!(b, [1.5f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat());
    }

    #[test]
    fn end_to_end_matches_direct_symbols() {
        let (rows, k) = (10usize, 64usize);
        let vals = pseudo(1, rows * k);
        let file = quant::pack(&DType::Q8_0, &vals).unwrap();
        let x = pseudo(2, k);
        let s = reference_kernels(&DType::Q8_0).unwrap();
        let w = s.pack(&file, rows, k).unwrap();
        let xq = s.quantize_row(&x).unwrap();
        let mut y = vec![f32::NAN; rows];
        s.gemv(&mut y, &xq, &w, rows, k, 0, rows).unwrap();
        let mut direct = vec![f32::NAN; rows];
        // SAFETY: same validated inputs as the wrapper call above.
        unsafe {
            crate::inferno_gemv_q8_0_rs8_scalar(
                direct.as_mut_ptr(),
                xq.as_ptr(),
                w.as_ptr(),
                k,
                0,
                rows,
            );
        }
        for (a, b) in y.iter().zip(&direct) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn gemv_wrapper_validates_everything() {
        let (rows, k) = (4usize, 32usize);
        let vals = pseudo(3, rows * k);
        let file = quant::pack(&DType::Q8_0, &vals).unwrap();
        let s = reference_kernels(&DType::Q8_0).unwrap();
        let w = s.pack(&file, rows, k).unwrap();
        let x = pseudo(4, k);
        let xq = s.quantize_row(&x).unwrap();
        let mut y = vec![0f32; rows];
        // Good call passes.
        s.gemv(&mut y, &xq, &w, rows, k, 0, rows).unwrap();
        // y too short.
        assert!(s.gemv(&mut y[..3], &xq, &w, rows, k, 0, rows).is_err());
        // Inverted / out-of-bounds row range.
        assert!(s.gemv(&mut y, &xq, &w, rows, k, 3, 2).is_err());
        assert!(s.gemv(&mut y, &xq, &w, rows, k, 0, rows + 1).is_err());
        // Wrong activation buffer length.
        assert!(
            s.gemv(&mut y, &xq[..xq.len() - 1], &w, rows, k, 0, rows)
                .is_err()
        );
        // k not matching the packed image.
        assert!(s.gemv(&mut y, &xq, &w, rows, 64, 0, rows).is_err());
        // k not a block multiple.
        assert!(s.gemv(&mut y, &xq, &w, rows, 33, 0, rows).is_err());
        // quantize_row validates too.
        assert!(s.quantize_row(&pseudo(5, 31)).is_err());
    }
}
