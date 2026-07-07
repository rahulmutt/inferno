//! Q4_K GEMV in the rs8 layout. Weights: ggml 144-byte super-blocks with the
//! 6-bit (scale, min) packing decoded to plain u8 at pack time. Activations:
//! q8k (bsums feed the dmin correction). Integer dots exact; f32 combine in
//! fixed order — ISA variants are bit-identical.

use inferno_formats::quant::{f16_to_f32, get_scale_min_k4};

use crate::act::{Q8K_BLOCK_BYTES, hsum_i32};
#[cfg(target_arch = "x86_64")]
use crate::q8_0::hsum8_i32;
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
    let mut r = row_start;
    while r < row_end {
        let strip = r / STRIP;
        let lane0 = r - strip * STRIP;
        // Fast path: a whole strip lies in range → process its 8 rows lane-
        // parallel (acc lane = row), reading each group and the activation
        // super-block once instead of 8×. Within each row the chunk products
        // are accumulated in the vector domain (mullo by the broadcast scale)
        // and reduced once per super-block instead of 8× per-chunk hsums.
        if lane0 == 0 && r + STRIP <= row_end {
            let mut acc = _mm256_setzero_ps();
            for sb in 0..nsb {
                let g = unsafe { w.add((strip * nsb + sb) * GROUP_BYTES) };
                let xb = unsafe { x.add(sb * Q8K_BLOCK_BYTES) };
                let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                let xqs = unsafe { xb.add(4) };
                // Activation qs + bsums, shared across the strip's 8 rows.
                let mut xlo = [_mm256_setzero_si256(); 4];
                let mut xhi = [_mm256_setzero_si256(); 4];
                for c in 0..4 {
                    xlo[c] = unsafe { _mm256_loadu_si256(xqs.add(c * 64).cast()) };
                    xhi[c] = unsafe { _mm256_loadu_si256(xqs.add(c * 64 + 32).cast()) };
                }
                let bsums = unsafe { _mm256_loadu_si256(xb.add(260).cast()) };
                let sc0 = unsafe { g.add(OFF_SC) };
                let m0 = unsafe { g.add(OFF_M) };
                let qs0 = unsafe { g.add(OFF_QS) };
                let mut dv = [_mm256_setzero_si256(); STRIP]; // per-lane Σ sc·dot
                let mut mv = [_mm256_setzero_si256(); STRIP]; // per-lane m·bsum products
                for lane in 0..STRIP {
                    let sc = unsafe { sc0.add(lane * 8) };
                    let qs = unsafe { qs0.add(lane * 128) };
                    let mut sumd = _mm256_setzero_si256();
                    for c in 0..4 {
                        // Aligned: g 32-aligned, OFF_QS=192, lane*128, c*32.
                        let qv = unsafe { _mm256_load_si256(qs.add(c * 32).cast()) };
                        let lo = _mm256_and_si256(qv, nib);
                        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qv), nib);
                        // Nibbles unsigned 0..=15 → valid u8 operand for maddubs;
                        // pairs sum ≤ 2·15·127 < i16::MAX, no saturation.
                        let plo = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xlo[c]), ones);
                        let phi = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xhi[c]), ones);
                        // Weight each chunk's 8 partial sums by its 6-bit scale
                        // and accumulate; sc·dot < 2^22, Σ < 2^25 → mullo exact.
                        let sclo = _mm256_set1_epi32(i32::from(unsafe { sc.add(2 * c).read() }));
                        let schi =
                            _mm256_set1_epi32(i32::from(unsafe { sc.add(2 * c + 1).read() }));
                        sumd = _mm256_add_epi32(sumd, _mm256_mullo_epi32(plo, sclo));
                        sumd = _mm256_add_epi32(sumd, _mm256_mullo_epi32(phi, schi));
                    }
                    dv[lane] = sumd;
                    // m·bsum products (8 lanes, j = sub-block index); reduced below.
                    let mw =
                        _mm256_cvtepu8_epi32(unsafe { _mm_loadl_epi64(m0.add(lane * 8).cast()) });
                    mv[lane] = _mm256_mullo_epi32(mw, bsums);
                }
                // Transpose-reduce: sumd/summ lane i = row i (integer-exact).
                let sumd = _mm256_cvtepi32_ps(hsum8_i32(dv));
                let summ = _mm256_cvtepi32_ps(hsum8_i32(mv));
                // 8 groups' d / dmin are contiguous → one aligned load each,
                // lane = row, matching acc lane order.
                let dw = unsafe { _mm256_load_ps(g.cast()) };
                let dmin = unsafe { _mm256_load_ps(g.add(OFF_DMIN).cast()) };
                let dxv = _mm256_set1_ps(dx);
                // Per lane: acc = (dw*dx).mul_add(sumd, acc) then
                // (dmin*dx).mul_add(-summ, acc) — bit-identical to scalar.
                acc = _mm256_fmadd_ps(_mm256_mul_ps(dw, dxv), sumd, acc);
                // Bit-identical to the scalar `-(summ as f32)`: `summ` comes
                // from `cvtepi32_ps` of an integer sum, and 0-summ negates a
                // finite value exactly. Even were `summ` +0.0, `0.0 - 0.0` is
                // +0.0 (not −0.0), and `acc` is never −0.0 — it starts at +0.0
                // and fma products into a round-to-nearest +0.0 accumulator
                // cannot yield −0.0 — so any ±0.0 sign difference is absorbed.
                let neg_summ = _mm256_sub_ps(_mm256_setzero_ps(), summ);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(dmin, dxv), neg_summ, acc);
            }
            unsafe { _mm256_storeu_ps(y.add(r), acc) };
            r += STRIP;
            continue;
        }
        // Partial head/tail row: the original per-row path (already bit-identical).
        let lane = lane0;
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
        r += 1;
    }
}

/// Batched Q4_K GEMV (GEMM): `y[t*rows + r] = W[r] · dequant(xq_t)` for every
/// token `t in 0..m` and row `r in row_start..row_end`. Each weight super-block
/// is read once per batch (outer `sb`, inner `t`); per (t,r) the super-block
/// order is `0..nsb` with the same f32 combine as `inferno_gemv_q4_k_rs8_*`,
/// so `gemm(m=1)` is bitwise-equal to `gemv`.
///
/// # Safety
/// As the GEMV symbol, with: `xq` valid for `m` contiguous q8k rows of `k`
/// (`m * q8k_len(k)` bytes); `y` valid for `m * rows` f32 writes; `m >= 1`.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemm_q4_k_rs8_scalar(
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    m: usize,
    rows: usize,
    row_start: usize,
    row_end: usize,
) {
    let nsb = k / WBLOCK;
    let act = Q8K_BLOCK_BYTES * nsb; // per-token activation stride
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        // One accumulator per token; super-blocks visited in order → gemv order.
        let mut acc = vec![0f32; m];
        for sb in 0..nsb {
            let g = unsafe { w.add((strip * nsb + sb) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let dmin =
                f32::from_le_bytes(unsafe { g.add(OFF_DMIN + lane * 4).cast::<[u8; 4]>().read() });
            let sc = unsafe { g.add(OFF_SC + lane * 8) };
            let mp = unsafe { g.add(OFF_M + lane * 8) };
            let qs = unsafe { g.add(OFF_QS + lane * 128) };
            for (t, at) in acc.iter_mut().enumerate() {
                let xb = unsafe { xq.add(t * act + sb * Q8K_BLOCK_BYTES) };
                let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                let xqs = unsafe { xb.add(4) };
                let mut summ = 0i32;
                for j in 0..8 {
                    let bsum = i32::from_le_bytes(unsafe {
                        xb.add(260 + j * 4).cast::<[u8; 4]>().read_unaligned()
                    });
                    summ += i32::from(unsafe { mp.add(j).read() }) * bsum;
                }
                let mut sumd = 0i32;
                for c in 0..4 {
                    let (mut dlo, mut dhi) = (0i32, 0i32);
                    for i in 0..32 {
                        let qb = unsafe { qs.add(c * 32 + i).read() };
                        let lo = i32::from(qb & 0xF);
                        let hi = i32::from(qb >> 4);
                        dlo += lo * i32::from(unsafe { xqs.add(c * 64 + i).cast::<i8>().read() });
                        dhi +=
                            hi * i32::from(unsafe { xqs.add(c * 64 + 32 + i).cast::<i8>().read() });
                    }
                    sumd += i32::from(unsafe { sc.add(2 * c).read() }) * dlo
                        + i32::from(unsafe { sc.add(2 * c + 1).read() }) * dhi;
                }
                *at = (dw * dx).mul_add(sumd as f32, *at);
                *at = (dmin * dx).mul_add(-(summ as f32), *at);
            }
        }
        for (t, at) in acc.iter().enumerate() {
            unsafe { y.add(t * rows + r).write(*at) };
        }
    }
}

/// # Safety
/// As [`inferno_gemm_q4_k_rs8_scalar`]; additionally requires AVX2+FMA.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemm_q4_k_rs8_avx2(
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    m: usize,
    rows: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let nsb = k / WBLOCK;
    let act = Q8K_BLOCK_BYTES * nsb;
    let ones = _mm256_set1_epi16(1);
    let nib = _mm256_set1_epi8(0x0F);
    let mut r = row_start;
    while r < row_end {
        let strip = r / STRIP;
        let lane0 = r - strip * STRIP;
        // Full-strip fast path: 8 rows lane-parallel, one acc per token. Each
        // weight super-block is loaded once (outer `sb`); the per-token combine
        // matches the GEMV fast path exactly.
        if lane0 == 0 && r + STRIP <= row_end {
            let mut acc = vec![_mm256_setzero_ps(); m];
            for sb in 0..nsb {
                let g = unsafe { w.add((strip * nsb + sb) * GROUP_BYTES) };
                let sc0 = unsafe { g.add(OFF_SC) };
                let m0 = unsafe { g.add(OFF_M) };
                let qs0 = unsafe { g.add(OFF_QS) };
                // 8 groups' d / dmin contiguous → one aligned load each (weight).
                let dw = unsafe { _mm256_load_ps(g.cast()) };
                let dmin = unsafe { _mm256_load_ps(g.add(OFF_DMIN).cast()) };
                for (t, at) in acc.iter_mut().enumerate() {
                    let xb = unsafe { xq.add(t * act + sb * Q8K_BLOCK_BYTES) };
                    let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                    let xqs = unsafe { xb.add(4) };
                    let mut xlo = [_mm256_setzero_si256(); 4];
                    let mut xhi = [_mm256_setzero_si256(); 4];
                    for c in 0..4 {
                        xlo[c] = unsafe { _mm256_loadu_si256(xqs.add(c * 64).cast()) };
                        xhi[c] = unsafe { _mm256_loadu_si256(xqs.add(c * 64 + 32).cast()) };
                    }
                    let bsums = unsafe { _mm256_loadu_si256(xb.add(260).cast()) };
                    let mut dv = [_mm256_setzero_si256(); STRIP]; // per-lane Σ sc·dot
                    let mut mv = [_mm256_setzero_si256(); STRIP]; // per-lane m·bsum products
                    for lane in 0..STRIP {
                        let sc = unsafe { sc0.add(lane * 8) };
                        let qs = unsafe { qs0.add(lane * 128) };
                        let mut sumd = _mm256_setzero_si256();
                        for c in 0..4 {
                            // Aligned: g 32-aligned, OFF_QS=192, lane*128, c*32.
                            let qv = unsafe { _mm256_load_si256(qs.add(c * 32).cast()) };
                            let lo = _mm256_and_si256(qv, nib);
                            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qv), nib);
                            let plo = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xlo[c]), ones);
                            let phi = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xhi[c]), ones);
                            let sclo =
                                _mm256_set1_epi32(i32::from(unsafe { sc.add(2 * c).read() }));
                            let schi =
                                _mm256_set1_epi32(i32::from(unsafe { sc.add(2 * c + 1).read() }));
                            sumd = _mm256_add_epi32(sumd, _mm256_mullo_epi32(plo, sclo));
                            sumd = _mm256_add_epi32(sumd, _mm256_mullo_epi32(phi, schi));
                        }
                        dv[lane] = sumd;
                        let mw = _mm256_cvtepu8_epi32(unsafe {
                            _mm_loadl_epi64(m0.add(lane * 8).cast())
                        });
                        mv[lane] = _mm256_mullo_epi32(mw, bsums);
                    }
                    // Transpose-reduce: sumd/summ lane i = row i (integer-exact).
                    let sumd = _mm256_cvtepi32_ps(hsum8_i32(dv));
                    let summ = _mm256_cvtepi32_ps(hsum8_i32(mv));
                    let dxv = _mm256_set1_ps(dx);
                    // Per lane, in gemv order: (dw*dx).mul_add(sumd, acc) then
                    // (dmin*dx).mul_add(-summ, acc).
                    *at = _mm256_fmadd_ps(_mm256_mul_ps(dw, dxv), sumd, *at);
                    let neg_summ = _mm256_sub_ps(_mm256_setzero_ps(), summ);
                    *at = _mm256_fmadd_ps(_mm256_mul_ps(dmin, dxv), neg_summ, *at);
                }
            }
            for (t, at) in acc.iter().enumerate() {
                unsafe { _mm256_storeu_ps(y.add(t * rows + r), *at) };
            }
            r += STRIP;
            continue;
        }
        // Partial head/tail row: per-row path, one acc per token.
        let lane = lane0;
        let mut acc = vec![0f32; m];
        for sb in 0..nsb {
            let g = unsafe { w.add((strip * nsb + sb) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let dmin =
                f32::from_le_bytes(unsafe { g.add(OFF_DMIN + lane * 4).cast::<[u8; 4]>().read() });
            let sc = unsafe { g.add(OFF_SC + lane * 8) };
            let mp = unsafe { g.add(OFF_M + lane * 8) };
            let qs = unsafe { g.add(OFF_QS + lane * 128) };
            for (t, at) in acc.iter_mut().enumerate() {
                let xb = unsafe { xq.add(t * act + sb * Q8K_BLOCK_BYTES) };
                let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                let xqs = unsafe { xb.add(4) };
                let mut summ = 0i32;
                for j in 0..8 {
                    let bsum = i32::from_le_bytes(unsafe {
                        xb.add(260 + j * 4).cast::<[u8; 4]>().read_unaligned()
                    });
                    summ += i32::from(unsafe { mp.add(j).read() }) * bsum;
                }
                let mut sumd = 0i32;
                for c in 0..4 {
                    // Aligned: g 32-aligned, OFF_QS=192, lane*128, c*32.
                    let qv = unsafe { _mm256_load_si256(qs.add(c * 32).cast()) };
                    let lo = _mm256_and_si256(qv, nib);
                    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qv), nib);
                    let x_lo = unsafe { _mm256_loadu_si256(xqs.add(c * 64).cast()) };
                    let x_hi = unsafe { _mm256_loadu_si256(xqs.add(c * 64 + 32).cast()) };
                    let dlo = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(lo, x_lo), ones));
                    let dhi = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(hi, x_hi), ones));
                    sumd += i32::from(unsafe { sc.add(2 * c).read() }) * dlo
                        + i32::from(unsafe { sc.add(2 * c + 1).read() }) * dhi;
                }
                *at = (dw * dx).mul_add(sumd as f32, *at);
                *at = (dmin * dx).mul_add(-(summ as f32), *at);
            }
        }
        for (t, at) in acc.iter().enumerate() {
            unsafe { y.add(t * rows + r).write(*at) };
        }
        r += 1;
    }
}
