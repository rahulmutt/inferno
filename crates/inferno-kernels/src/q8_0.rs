//! Q8_0 GEMV in the rs8 layout. Weights: ggml Q8_0 blocks (f16 scale + 32 i8)
//! repacked per strip with scales widened to f32. Activations: q8a. Integer
//! dots are exact; the f32 combine runs in block order in every variant.

use inferno_formats::quant::f16_to_f32;

use crate::act::{Q8A_BLOCK_BYTES, hsum_i32};
use crate::{AlignedBuf, KernelError, Result, STRIP};

const WBLOCK: usize = 32; // weight elements per block
const FILE_BLOCK_BYTES: usize = 34; // f16 d + 32 i8
const GROUP_BYTES: usize = 288; // 8 f32 d + 8×32 qs

pub fn packed_len_q8_0_rs8(rows: usize, k: usize) -> usize {
    rows.div_ceil(STRIP) * (k / WBLOCK) * GROUP_BYTES
}

/// Repack file-order Q8_0 blocks into rs8 groups. Clamps qs `-128 → -127`
/// so the AVX2 sign-trick stays exact on hostile files (plan Deviations §5);
/// ggml's own quantizer never emits −128, so real files are unchanged.
pub fn pack_q8_0_rs8(bytes: &[u8], rows: usize, k: usize) -> Result<AlignedBuf> {
    if rows == 0 {
        return Err(KernelError::ZeroRows);
    }
    if k == 0 || !k.is_multiple_of(WBLOCK) {
        return Err(KernelError::BadK { k, block: WBLOCK });
    }
    let nb = k / WBLOCK;
    let expected = rows
        .checked_mul(nb)
        .and_then(|n| n.checked_mul(FILE_BLOCK_BYTES))
        .ok_or(KernelError::Overflow)?;
    if bytes.len() != expected {
        return Err(KernelError::SizeMismatch {
            what: "Q8_0 weight bytes",
            got: bytes.len(),
            expected,
        });
    }
    let mut out = AlignedBuf::zeroed(packed_len_q8_0_rs8(rows, k));
    let dst = out.as_mut_slice();
    for r in 0..rows {
        let (strip, lane) = (r / STRIP, r % STRIP);
        for b in 0..nb {
            let s = (r * nb + b) * FILE_BLOCK_BYTES;
            let g = (strip * nb + b) * GROUP_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([bytes[s], bytes[s + 1]]));
            dst[g + lane * 4..g + lane * 4 + 4].copy_from_slice(&d.to_le_bytes());
            for (i, &q) in bytes[s + 2..s + 2 + WBLOCK].iter().enumerate() {
                dst[g + 32 + lane * WBLOCK + i] = if q as i8 == i8::MIN { -127i8 as u8 } else { q };
            }
        }
    }
    Ok(out)
}

/// Unified GEMV ABI: `y[row_start..row_end] = W · dequant(x)`.
///
/// # Safety
/// - `y` valid for f32 writes at `row_start..row_end`.
/// - `x` is a q8a buffer for this `k` (from `inferno_quantize_row_q8a_*`).
/// - `w` is a `pack_q8_0_rs8` image built with this exact `k` and at least
///   `row_end` rows, 32-byte aligned.
/// - `row_start <= row_end`; `k` a positive multiple of 32.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_q8_0_rs8_scalar(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    let nb = k / WBLOCK;
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let mut acc = 0f32;
        for b in 0..nb {
            let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
            let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
            let qw = unsafe { g.add(32 + lane * WBLOCK) };
            let qx = unsafe { xb.add(4) };
            let mut isum = 0i32;
            for i in 0..WBLOCK {
                let a = i32::from(unsafe { qw.add(i).cast::<i8>().read() });
                let b_ = i32::from(unsafe { qx.add(i).cast::<i8>().read() });
                isum += a * b_;
            }
            acc = (dw * dx).mul_add(isum as f32, acc);
        }
        unsafe { y.add(r).write(acc) };
    }
}

/// # Safety
/// As [`inferno_gemv_q8_0_rs8_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_q8_0_rs8_avx2(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let nb = k / WBLOCK;
    let ones = _mm256_set1_epi16(1);
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let mut acc = 0f32;
        for b in 0..nb {
            let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
            let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
            // Aligned: group is 32-aligned, +32, lane*32.
            let wv = unsafe { _mm256_load_si256(g.add(32 + lane * WBLOCK).cast()) };
            let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
            // Sign trick (both operands in [-127,127] by pack/quantize):
            // |w| as u8 × sign-adjusted x, exact in i16/i32.
            let aw = _mm256_sign_epi8(wv, wv);
            let sx = _mm256_sign_epi8(xv, wv);
            let p16 = _mm256_maddubs_epi16(aw, sx);
            let p32 = _mm256_madd_epi16(p16, ones);
            let isum = hsum_i32(p32);
            acc = (dw * dx).mul_add(isum as f32, acc);
        }
        unsafe { y.add(r).write(acc) };
    }
}
