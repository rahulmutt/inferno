//! Activation-side quantization: f32 rows → integer blocks at the kernel
//! boundary, mirroring ggml's pairings (Q4_K×Q8_K, Q8_0×Q8_0). Kernel-internal
//! formats — never `inferno_formats::DType` (spec boundary rule).
//!
//! q8a, per 32 elems:  [d: f32 le][qs: 32 × i8]                      = 36 B
//! q8k, per 256 elems: [d: f32 le][qs: 256 × i8][bsums: 8 × i32 le]  = 292 B
//!
//! Rounding is ties-to-even in every variant (`round_ties_even` scalar,
//! `cvtps` under the default MXCSR mode in AVX2) so variants stay bitwise
//! equal; quantized values are clamped to [-127, 127].

use crate::{KernelError, KernelIsa, Result};

pub const Q8A_BLOCK: usize = 32;
pub const Q8A_BLOCK_BYTES: usize = 36;
pub const Q8K_BLOCK: usize = 256;
pub const Q8K_BLOCK_BYTES: usize = 292;

pub fn q8a_len(k: usize) -> usize {
    k / Q8A_BLOCK * Q8A_BLOCK_BYTES
}

pub fn q8k_len(k: usize) -> usize {
    k / Q8K_BLOCK * Q8K_BLOCK_BYTES
}

/// Scalar semantic core: quantize one block, returning its scale.
fn quantize_block(x: &[f32], qs: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
    for (q, v) in qs.iter_mut().zip(x) {
        *q = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
    }
    d
}

/// # Safety
/// - `x` valid for `k` f32 reads; `y` valid for `q8a_len(k)` byte writes.
/// - `k` is a multiple of 32. All inputs finite (NaN/Inf are precondition
///   violations — kernels do not check; the hot path stays branch-free).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_quantize_row_q8a_scalar(x: *const f32, y: *mut u8, k: usize) {
    let x = unsafe { std::slice::from_raw_parts(x, k) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, q8a_len(k)) };
    for (xb, yb) in x
        .chunks_exact(Q8A_BLOCK)
        .zip(y.chunks_exact_mut(Q8A_BLOCK_BYTES))
    {
        let mut qs = [0i8; Q8A_BLOCK];
        let d = quantize_block(xb, &mut qs);
        yb[..4].copy_from_slice(&d.to_le_bytes());
        for (dst, q) in yb[4..].iter_mut().zip(qs) {
            *dst = q as u8;
        }
    }
}

/// # Safety
/// As [`inferno_quantize_row_q8a_scalar`], with `k` a multiple of 256 and `y`
/// valid for `q8k_len(k)` byte writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_quantize_row_q8k_scalar(x: *const f32, y: *mut u8, k: usize) {
    let x = unsafe { std::slice::from_raw_parts(x, k) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, q8k_len(k)) };
    for (xb, yb) in x
        .chunks_exact(Q8K_BLOCK)
        .zip(y.chunks_exact_mut(Q8K_BLOCK_BYTES))
    {
        let mut qs = [0i8; Q8K_BLOCK];
        let d = quantize_block(xb, &mut qs);
        yb[..4].copy_from_slice(&d.to_le_bytes());
        for (dst, q) in yb[4..260].iter_mut().zip(qs) {
            *dst = q as u8;
        }
        for j in 0..8 {
            let s: i32 = qs[j * 32..(j + 1) * 32].iter().map(|&q| i32::from(q)).sum();
            yb[260 + j * 4..264 + j * 4].copy_from_slice(&s.to_le_bytes());
        }
    }
}

/// AVX2 core: quantize one 32-f32 chunk against a precomputed `inv`, writing
/// 32 i8 to `dst` and returning the four pre-narrowing i32 vectors' sum (the
/// caller uses it for q8k bsums). Must match `quantize_block` bitwise:
/// same mul, ties-to-even rounding, clamp to [-127, 127].
///
/// # Safety
/// - `x` must be valid for 32 f32 reads.
/// - `dst` must be valid for 32 byte writes.
/// - Caller must have AVX2 and FMA enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quant32_avx2(x: *const f32, inv: f32, dst: *mut u8) -> i32 {
    use std::arch::x86_64::*;
    let vinv = _mm256_set1_ps(inv);
    let lo127 = _mm256_set1_epi32(-127);
    let hi127 = _mm256_set1_epi32(127);
    let mut q = [_mm256_setzero_si256(); 4];
    let mut bsum = _mm256_setzero_si256();
    for (i, qi) in q.iter_mut().enumerate() {
        let v = unsafe { _mm256_loadu_ps(x.add(i * 8)) };
        let r = _mm256_cvtps_epi32(_mm256_mul_ps(v, vinv)); // ties-to-even
        let c = _mm256_max_epi32(lo127, _mm256_min_epi32(hi127, r));
        bsum = _mm256_add_epi32(bsum, c);
        *qi = c;
    }
    let p0 = _mm256_packs_epi32(q[0], q[1]);
    let p1 = _mm256_packs_epi32(q[2], q[3]);
    let packed = _mm256_packs_epi16(p0, p1);
    // packs interleaves 128-bit lanes; restore element order.
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let packed = _mm256_permutevar8x32_epi32(packed, order);
    unsafe { _mm256_storeu_si256(dst.cast(), packed) };
    hsum_i32(bsum)
}

/// Horizontal sum of 8 × i32. Exact (integer), so reduction order is free.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn hsum_i32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_hadd_epi32(s, s);
    let s = _mm_hadd_epi32(s, s);
    _mm_cvtsi128_si32(s)
}

/// amax of one 32-f32 chunk. max is exact and order-free on finite input.
///
/// # Safety
/// - `x` must be valid for 32 f32 reads.
/// - Caller must have AVX2 enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn amax32_avx2(x: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let sign = _mm256_set1_ps(-0.0);
    let mut m = _mm256_setzero_ps();
    for i in 0..4 {
        let v = unsafe { _mm256_loadu_ps(x.add(i * 8)) };
        m = _mm256_max_ps(m, _mm256_andnot_ps(sign, v));
    }
    let s = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps::<1>(m));
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_max_ss(s, _mm_shuffle_ps::<1>(s, s));
    _mm_cvtss_f32(s)
}

/// # Safety
/// As [`inferno_quantize_row_q8a_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_quantize_row_q8a_avx2(x: *const f32, y: *mut u8, k: usize) {
    for b in 0..k / Q8A_BLOCK {
        let xb = unsafe { x.add(b * Q8A_BLOCK) };
        let yb = unsafe { y.add(b * Q8A_BLOCK_BYTES) };
        let amax = unsafe { amax32_avx2(xb) };
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        unsafe { yb.cast::<[u8; 4]>().write_unaligned(d.to_le_bytes()) };
        unsafe { quant32_avx2(xb, inv, yb.add(4)) };
    }
}

/// # Safety
/// As [`inferno_quantize_row_q8k_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_quantize_row_q8k_avx2(x: *const f32, y: *mut u8, k: usize) {
    for b in 0..k / Q8K_BLOCK {
        let xb = unsafe { x.add(b * Q8K_BLOCK) };
        let yb = unsafe { y.add(b * Q8K_BLOCK_BYTES) };
        let mut amax = 0f32;
        for c in 0..8 {
            amax = amax.max(unsafe { amax32_avx2(xb.add(c * 32)) });
        }
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        unsafe { yb.cast::<[u8; 4]>().write_unaligned(d.to_le_bytes()) };
        for j in 0..8 {
            let bsum = unsafe { quant32_avx2(xb.add(j * 32), inv, yb.add(4 + j * 32)) };
            unsafe {
                yb.add(260 + j * 4)
                    .cast::<[u8; 4]>()
                    .write_unaligned(bsum.to_le_bytes())
            };
        }
    }
}

fn validate(isa: KernelIsa, k: usize, block: usize) -> Result<()> {
    // Guard AVX2 dispatch locally: a hand-built `KernelIsa::Avx2` reaching a
    // `#[target_feature(enable=…)]` symbol on a CPU without the feature is UB
    // from safe code. `available()` is the invariant, checked here rather than
    // delegated to the registry's selection path.
    if !isa.available() {
        return Err(KernelError::IsaUnavailable { isa: isa_name(isa) });
    }
    if k == 0 || !k.is_multiple_of(block) {
        return Err(KernelError::BadK { k, block });
    }
    // Reject dimensions the unchecked length helpers could overflow on before
    // any `q8*_len(k)` product is computed (see `crate::MAX_K`).
    if k > crate::MAX_K {
        return Err(KernelError::Overflow);
    }
    Ok(())
}

fn isa_name(isa: KernelIsa) -> &'static str {
    match isa {
        KernelIsa::Scalar => "scalar",
        KernelIsa::Avx2 => "avx2",
    }
}

/// Safe wrapper (tests, benches, M3 planner). The raw symbols stay unchecked.
pub fn quantize_row_q8a(isa: KernelIsa, x: &[f32]) -> Result<Vec<u8>> {
    validate(isa, x.len(), Q8A_BLOCK)?;
    let mut out = vec![0u8; q8a_len(x.len())];
    match isa {
        // SAFETY: x/out lengths validated against the symbol's contract; the
        // AVX2 feature invariant was checked locally by `validate` above.
        KernelIsa::Scalar => unsafe {
            inferno_quantize_row_q8a_scalar(x.as_ptr(), out.as_mut_ptr(), x.len())
        },
        // SAFETY: as above; `validate` returned Err unless AVX2+FMA is present.
        KernelIsa::Avx2 => unsafe {
            inferno_quantize_row_q8a_avx2(x.as_ptr(), out.as_mut_ptr(), x.len())
        },
    }
    Ok(out)
}

pub fn quantize_row_q8k(isa: KernelIsa, x: &[f32]) -> Result<Vec<u8>> {
    validate(isa, x.len(), Q8K_BLOCK)?;
    let mut out = vec![0u8; q8k_len(x.len())];
    match isa {
        // SAFETY: as quantize_row_q8a.
        KernelIsa::Scalar => unsafe {
            inferno_quantize_row_q8k_scalar(x.as_ptr(), out.as_mut_ptr(), x.len())
        },
        // SAFETY: as quantize_row_q8a.
        KernelIsa::Avx2 => unsafe {
            inferno_quantize_row_q8k_avx2(x.as_ptr(), out.as_mut_ptr(), x.len())
        },
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelIsa;
    use inferno_formats::DType;
    use inferno_graph::tolerance::roundtrip_rel_tol;
    use proptest::prelude::*;

    /// Deterministic pseudo-random f32s in [-1, 1) — cheaper than proptest
    /// vec strategies for large inputs, still seed-driven.
    pub(crate) fn pseudo(mut seed: u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn decode_q8a(buf: &[u8], k: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(k);
        for b in buf.chunks_exact(Q8A_BLOCK_BYTES) {
            let d = f32::from_le_bytes(b[..4].try_into().unwrap());
            out.extend(b[4..].iter().map(|&q| d * f32::from(q as i8)));
        }
        out
    }

    fn decode_q8k(buf: &[u8], k: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(k);
        for b in buf.chunks_exact(Q8K_BLOCK_BYTES) {
            let d = f32::from_le_bytes(b[..4].try_into().unwrap());
            out.extend(b[4..260].iter().map(|&q| d * f32::from(q as i8)));
        }
        out
    }

    fn check_roundtrip(vals: &[f32], decoded: &[f32]) {
        // 8-bit block quant: same error class as Q8_0 weights.
        let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
        let tol = roundtrip_rel_tol(&DType::Q8_0) * amax;
        for (i, (a, b)) in vals.iter().zip(decoded).enumerate() {
            assert!((a - b).abs() <= tol, "[{i}]: {a} vs {b} (tol {tol})");
        }
    }

    proptest! {
        #[test]
        fn q8a_roundtrip_all_isas(seed in any::<u64>(), blocks in 1usize..5) {
            let k = blocks * Q8A_BLOCK;
            let x = pseudo(seed, k);
            for isa in KernelIsa::all_available() {
                let buf = quantize_row_q8a(isa, &x).unwrap();
                prop_assert_eq!(buf.len(), q8a_len(k));
                check_roundtrip(&x, &decode_q8a(&buf, k));
            }
        }

        #[test]
        fn q8k_roundtrip_and_exact_bsums(seed in any::<u64>(), blocks in 1usize..3) {
            let k = blocks * Q8K_BLOCK;
            let x = pseudo(seed, k);
            for isa in KernelIsa::all_available() {
                let buf = quantize_row_q8k(isa, &x).unwrap();
                prop_assert_eq!(buf.len(), q8k_len(k));
                check_roundtrip(&x, &decode_q8k(&buf, k));
                for b in buf.chunks_exact(Q8K_BLOCK_BYTES) {
                    for j in 0..8 {
                        let want: i32 =
                            b[4 + j * 32..4 + (j + 1) * 32].iter().map(|&q| i32::from(q as i8)).sum();
                        let got =
                            i32::from_le_bytes(b[260 + j * 4..264 + j * 4].try_into().unwrap());
                        prop_assert_eq!(got, want, "bsum {}", j);
                    }
                }
            }
        }

        /// ISA variants must produce byte-identical activation buffers.
        #[test]
        fn quantize_isa_variants_bitwise_equal(seed in any::<u64>()) {
            if !KernelIsa::Avx2.available() { return Ok(()); }
            let x = pseudo(seed, 2 * Q8K_BLOCK);
            prop_assert_eq!(
                quantize_row_q8a(KernelIsa::Scalar, &x).unwrap(),
                quantize_row_q8a(KernelIsa::Avx2, &x).unwrap()
            );
            prop_assert_eq!(
                quantize_row_q8k(KernelIsa::Scalar, &x).unwrap(),
                quantize_row_q8k(KernelIsa::Avx2, &x).unwrap()
            );
        }
    }

    #[test]
    fn zero_block_has_zero_scale() {
        let buf = quantize_row_q8a(KernelIsa::Scalar, &[0f32; 32]).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn bad_k_rejected() {
        assert!(quantize_row_q8a(KernelIsa::Scalar, &[0f32; 31]).is_err());
        assert!(quantize_row_q8k(KernelIsa::Scalar, &[0f32; 255]).is_err());
        assert!(quantize_row_q8a(KernelIsa::Scalar, &[]).is_err());
    }

    /// The wrappers guard AVX2 dispatch on `isa.available()` *before* any
    /// pointer work, so a `KernelIsa::Avx2` call on a non-AVX2 CPU returns a
    /// typed error instead of executing an illegal instruction (UB). Scalar is
    /// always available. On an AVX2 machine the guard passes and the Avx2 path
    /// runs; on one without it, it short-circuits with `IsaUnavailable`.
    #[test]
    fn isa_availability_is_checked() {
        assert!(quantize_row_q8a(KernelIsa::Scalar, &[0f32; 32]).is_ok());
        assert!(quantize_row_q8k(KernelIsa::Scalar, &[0f32; 256]).is_ok());
        if KernelIsa::Avx2.available() {
            assert!(quantize_row_q8a(KernelIsa::Avx2, &[0f32; 32]).is_ok());
            assert!(quantize_row_q8k(KernelIsa::Avx2, &[0f32; 256]).is_ok());
        } else {
            assert!(matches!(
                quantize_row_q8a(KernelIsa::Avx2, &[0f32; 32]),
                Err(KernelError::IsaUnavailable { .. })
            ));
        }
    }
}
