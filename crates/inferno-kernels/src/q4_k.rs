//! Q4_K GEMV in the rs8 layout. Weights: ggml 144-byte super-blocks with the
//! 6-bit (scale, min) packing decoded to plain u8 at pack time. Activations:
//! q8k (bsums feed the dmin correction). Integer dots exact; f32 combine in
//! fixed order — ISA variants are bit-identical.

use inferno_formats::quant::{f16_to_f32, get_scale_min_k4};

use crate::act::{Q8K_BLOCK_BYTES, hsum_i32};
use crate::{AlignedBuf, KernelError, Result, STRIP};

const WBLOCK: usize = 256; // weight elements per super-block
const FILE_SB_BYTES: usize = 144;
const GROUP_BYTES: usize = 1216; // 32 d + 32 dmin + 64 sc + 64 m + 1024 qs
const OFF_DMIN: usize = 32;
const OFF_SC: usize = 64;
const OFF_M: usize = 128;
const OFF_QS: usize = 192;

pub fn packed_len_q4_k_rs8(rows: usize, k: usize) -> usize {
    rows.div_ceil(STRIP) * (k / WBLOCK) * GROUP_BYTES
}

pub fn pack_q4_k_rs8(bytes: &[u8], rows: usize, k: usize) -> Result<AlignedBuf> {
    if rows == 0 {
        return Err(KernelError::ZeroRows);
    }
    if k == 0 || !k.is_multiple_of(WBLOCK) {
        return Err(KernelError::BadK { k, block: WBLOCK });
    }
    let nsb = k / WBLOCK;
    let expected = rows
        .checked_mul(nsb)
        .and_then(|n| n.checked_mul(FILE_SB_BYTES))
        .ok_or(KernelError::Overflow)?;
    if bytes.len() != expected {
        return Err(KernelError::SizeMismatch {
            what: "Q4_K weight bytes",
            got: bytes.len(),
            expected,
        });
    }
    let mut out = AlignedBuf::zeroed(packed_len_q4_k_rs8(rows, k));
    let dst = out.as_mut_slice();
    for r in 0..rows {
        let (strip, lane) = (r / STRIP, r % STRIP);
        for sb in 0..nsb {
            let s = (r * nsb + sb) * FILE_SB_BYTES;
            let g = (strip * nsb + sb) * GROUP_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([bytes[s], bytes[s + 1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([bytes[s + 2], bytes[s + 3]]));
            dst[g + lane * 4..g + lane * 4 + 4].copy_from_slice(&d.to_le_bytes());
            dst[g + OFF_DMIN + lane * 4..g + OFF_DMIN + lane * 4 + 4]
                .copy_from_slice(&dmin.to_le_bytes());
            for j in 0..8 {
                let (sc, m) = get_scale_min_k4(j, &bytes[s + 4..s + 16]);
                dst[g + OFF_SC + lane * 8 + j] = sc;
                dst[g + OFF_M + lane * 8 + j] = m;
            }
            dst[g + OFF_QS + lane * 128..g + OFF_QS + (lane + 1) * 128]
                .copy_from_slice(&bytes[s + 16..s + 144]);
        }
    }
    Ok(out)
}

/// Unified GEMV ABI: `y[row_start..row_end] = W · dequant(x)`.
///
/// # Safety
/// - `y` valid for f32 writes at `row_start..row_end`.
/// - `x` is a q8k buffer for this `k` (from `inferno_quantize_row_q8k_*`).
/// - `w` is a `pack_q4_k_rs8` image built with this exact `k` and at least
///   `row_end` rows, 32-byte aligned.
/// - `row_start <= row_end`; `k` a positive multiple of 256.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_q4_k_rs8_scalar(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    let nsb = k / WBLOCK;
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let mut acc = 0f32;
        for sb in 0..nsb {
            let g = unsafe { w.add((strip * nsb + sb) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let dmin =
                f32::from_le_bytes(unsafe { g.add(OFF_DMIN + lane * 4).cast::<[u8; 4]>().read() });
            let sc = unsafe { g.add(OFF_SC + lane * 8) };
            let m = unsafe { g.add(OFF_M + lane * 8) };
            let qs = unsafe { g.add(OFF_QS + lane * 128) };
            let xb = unsafe { x.add(sb * Q8K_BLOCK_BYTES) };
            let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
            let xqs = unsafe { xb.add(4) };
            let mut summ = 0i32;
            for j in 0..8 {
                let bsum = i32::from_le_bytes(unsafe {
                    xb.add(260 + j * 4).cast::<[u8; 4]>().read_unaligned()
                });
                summ += i32::from(unsafe { m.add(j).read() }) * bsum;
            }
            let mut sumd = 0i32;
            for c in 0..4 {
                let (mut dlo, mut dhi) = (0i32, 0i32);
                for i in 0..32 {
                    let qb = unsafe { qs.add(c * 32 + i).read() };
                    let lo = i32::from(qb & 0xF);
                    let hi = i32::from(qb >> 4);
                    dlo += lo * i32::from(unsafe { xqs.add(c * 64 + i).cast::<i8>().read() });
                    dhi += hi * i32::from(unsafe { xqs.add(c * 64 + 32 + i).cast::<i8>().read() });
                }
                sumd += i32::from(unsafe { sc.add(2 * c).read() }) * dlo
                    + i32::from(unsafe { sc.add(2 * c + 1).read() }) * dhi;
            }
            acc = (dw * dx).mul_add(sumd as f32, acc);
            acc = (dmin * dx).mul_add(-(summ as f32), acc);
        }
        unsafe { y.add(r).write(acc) };
    }
}

/// # Safety
/// As [`inferno_gemv_q4_k_rs8_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_q4_k_rs8_avx2(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let nsb = k / WBLOCK;
    let ones = _mm256_set1_epi16(1);
    let nib = _mm256_set1_epi8(0x0F);
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let mut acc = 0f32;
        for sb in 0..nsb {
            let g = unsafe { w.add((strip * nsb + sb) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let dmin =
                f32::from_le_bytes(unsafe { g.add(OFF_DMIN + lane * 4).cast::<[u8; 4]>().read() });
            let sc = unsafe { g.add(OFF_SC + lane * 8) };
            let m = unsafe { g.add(OFF_M + lane * 8) };
            let qs = unsafe { g.add(OFF_QS + lane * 128) };
            let xb = unsafe { x.add(sb * Q8K_BLOCK_BYTES) };
            let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
            let xqs = unsafe { xb.add(4) };
            let mut summ = 0i32;
            for j in 0..8 {
                let bsum = i32::from_le_bytes(unsafe {
                    xb.add(260 + j * 4).cast::<[u8; 4]>().read_unaligned()
                });
                summ += i32::from(unsafe { m.add(j).read() }) * bsum;
            }
            let mut sumd = 0i32;
            for c in 0..4 {
                // Aligned: g 32-aligned, OFF_QS=192, lane*128, c*32.
                let qv = unsafe { _mm256_load_si256(qs.add(c * 32).cast()) };
                let lo = _mm256_and_si256(qv, nib);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qv), nib);
                let x_lo = unsafe { _mm256_loadu_si256(xqs.add(c * 64).cast()) };
                let x_hi = unsafe { _mm256_loadu_si256(xqs.add(c * 64 + 32).cast()) };
                // Nibbles are unsigned 0..=15 → valid u8 operand for maddubs;
                // pairs sum ≤ 2·15·127 < i16::MAX, no saturation.
                let dlo = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(lo, x_lo), ones));
                let dhi = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(hi, x_hi), ones));
                sumd += i32::from(unsafe { sc.add(2 * c).read() }) * dlo
                    + i32::from(unsafe { sc.add(2 * c + 1).read() }) * dhi;
            }
            acc = (dw * dx).mul_add(sumd as f32, acc);
            acc = (dmin * dx).mul_add(-(summ as f32), acc);
        }
        unsafe { y.add(r).write(acc) };
    }
}
