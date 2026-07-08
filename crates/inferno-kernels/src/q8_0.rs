//! Q8_0 GEMV in the rs8 layout. Weights: ggml Q8_0 blocks (f16 scale + 32 i8)
//! repacked per strip with scales widened to f32. Activations: q8a. Integer
//! dots are exact; the f32 combine runs in block order in every variant.

use inferno_formats::quant::f16_to_f32;

use crate::act::{Q8A_BLOCK_BYTES, hsum_i32};
use crate::{AlignedBuf, KernelError, Result, STRIP};

const WBLOCK: usize = 32; // weight elements per block
const FILE_BLOCK_BYTES: usize = 34; // f16 d + 32 i8
const GROUP_BYTES: usize = 288; // 8 f32 d + 8×32 qs

/// Weight groups to software-prefetch ahead in the AVX2 GEMV (M4b.4). A
/// strip's `nb` groups are contiguous (`nb × GROUP_BYTES`), so prefetching
/// `PF_DIST` groups ahead reaches cleanly across the block loop and into the
/// next strip. Plan default (4); the Task 2 sweep was deferred to quiet
/// hardware — see the spec Amendment. Pure hint, so it never affects output bits.
const PF_DIST: usize = 4;

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

/// Transpose-reduce 8 lane-parallel i32 accumulators into one vector whose
/// lane `i` holds the horizontal sum of `v[i]`'s 8 lanes. Pure integer adds,
/// so — like [`hsum_i32`] — the reduction structure is unconstrained by the
/// numeric contract; it lets a strip emit all 8 rows' block dots at once.
///
/// Callers must have AVX2 enabled (`target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn hsum8_i32(v: [std::arch::x86_64::__m256i; 8]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // Each hadd halves the live lanes per 128-bit half; two rounds leave, per
    // 128-bit half, the four inputs' lower- resp. upper-half partial sums.
    let s01 = _mm256_hadd_epi32(v[0], v[1]);
    let s23 = _mm256_hadd_epi32(v[2], v[3]);
    let s45 = _mm256_hadd_epi32(v[4], v[5]);
    let s67 = _mm256_hadd_epi32(v[6], v[7]);
    // s0123 = [v0lo v1lo v2lo v3lo | v0hi v1hi v2hi v3hi]; s4567 likewise for 4..8.
    let s0123 = _mm256_hadd_epi32(s01, s23);
    let s4567 = _mm256_hadd_epi32(s45, s67);
    // Recombine the low/high 128-bit halves so lane i = full sum of v[i].
    let lo = _mm256_permute2x128_si256::<0x20>(s0123, s4567);
    let hi = _mm256_permute2x128_si256::<0x31>(s0123, s4567);
    _mm256_add_epi32(lo, hi)
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
    let mut r = row_start;
    while r < row_end {
        let strip = r / STRIP;
        let lane0 = r - strip * STRIP;
        // Fast path: a whole strip lies in range → process its 8 rows lane-
        // parallel (acc lane = row), reading each group once instead of 8×.
        if lane0 == 0 && r + STRIP <= row_end {
            let mut acc = _mm256_setzero_ps();
            for b in 0..nb {
                let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                // Prefetch a future weight group into L1 to overlap DRAM latency
                // with this block's int8 dot. `wrapping_add` (not `add`) because
                // the last strip's tail offsets point past the buffer end;
                // `_mm_prefetch` never dereferences and never faults, so it stays
                // a pure hint — output is unchanged.
                let pf_addr = w
                    .wrapping_add((strip * nb + b + PF_DIST) * GROUP_BYTES)
                    .cast();
                _mm_prefetch::<_MM_HINT_T0>(pf_addr);
                let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
                let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                // Activation qs shared across the strip's 8 rows.
                let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                let qs = unsafe { g.add(32) };
                let mut p = [_mm256_setzero_si256(); STRIP];
                for (lane, pl) in p.iter_mut().enumerate() {
                    // Aligned: group is 32-aligned, +32, lane*32.
                    let wv = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                    // Sign trick, per lane (as the per-row path): |w| as u8 ×
                    // sign-adjusted x, exact in i16/i32.
                    let aw = _mm256_sign_epi8(wv, wv);
                    let sx = _mm256_sign_epi8(xv, wv);
                    *pl = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                }
                // isum lane i = row i's block dot (integer-exact reduction).
                let isum = _mm256_cvtepi32_ps(hsum8_i32(p));
                // 8 groups' d's are contiguous at g (lane*4) → one aligned load,
                // lane = row, matching acc/isum lane order.
                let dw = unsafe { _mm256_load_ps(g.cast()) };
                let dwdx = _mm256_mul_ps(dw, _mm256_set1_ps(dx));
                // Per lane: acc = (dw*dx).mul_add(isum, acc) — bit-identical to scalar.
                acc = _mm256_fmadd_ps(dwdx, isum, acc);
            }
            unsafe { _mm256_storeu_ps(y.add(r), acc) };
            r += STRIP;
            continue;
        }
        // Partial head/tail row: the original per-row path (already bit-identical).
        let lane = lane0;
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
        r += 1;
    }
}

/// Batched Q8_0 GEMV (GEMM): `y[t*rows + r] = W[r] · dequant(xq_t)` for every
/// token `t in 0..m` and row `r in row_start..row_end`. Each weight block is
/// read once per batch (outer `b`, inner `t`); per (t,r) the block order is
/// `0..nb`, identical to `inferno_gemv_q8_0_rs8_*`, so `gemm(m=1)` is
/// bitwise-equal to `gemv`.
///
/// # Safety
/// As the GEMV symbol, with: `xq` valid for `m` contiguous q8a rows of `k`
/// (`m * q8a_len(k)` bytes); `y` valid for `m * rows` f32 writes; `m >= 1`.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemm_q8_0_rs8_scalar(
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    m: usize,
    rows: usize,
    row_start: usize,
    row_end: usize,
) {
    let nb = k / WBLOCK;
    let act = Q8A_BLOCK_BYTES * nb; // per-token activation stride
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        // One accumulator per token; blocks visited in order → gemv order.
        let mut acc = vec![0f32; m];
        for b in 0..nb {
            let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let qw = unsafe { g.add(32 + lane * WBLOCK) };
            for (t, at) in acc.iter_mut().enumerate() {
                let xb = unsafe { xq.add(t * act + b * Q8A_BLOCK_BYTES) };
                let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                let qx = unsafe { xb.add(4) };
                let mut isum = 0i32;
                for i in 0..WBLOCK {
                    let a = i32::from(unsafe { qw.add(i).cast::<i8>().read() });
                    let bb = i32::from(unsafe { qx.add(i).cast::<i8>().read() });
                    isum += a * bb;
                }
                *at = (dw * dx).mul_add(isum as f32, *at);
            }
        }
        for (t, at) in acc.iter().enumerate() {
            unsafe { y.add(t * rows + r).write(*at) };
        }
    }
}

/// # Safety
/// As [`inferno_gemm_q8_0_rs8_scalar`]; additionally requires AVX2+FMA.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemm_q8_0_rs8_avx2(
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
    let nb = k / WBLOCK;
    let act = Q8A_BLOCK_BYTES * nb;
    let ones = _mm256_set1_epi16(1);
    let mut r = row_start;
    while r < row_end {
        let strip = r / STRIP;
        let lane0 = r - strip * STRIP;
        // Full-strip fast path: 8 rows lane-parallel, one acc per token.
        if lane0 == 0 && r + STRIP <= row_end {
            let mut acc = vec![_mm256_setzero_ps(); m];
            for b in 0..nb {
                let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                let qs = unsafe { g.add(32) };
                // Weight group's 8 per-row scales (lane = row), loaded once.
                let dw = unsafe { _mm256_load_ps(g.cast()) };
                for (t, at) in acc.iter_mut().enumerate() {
                    let xb = unsafe { xq.add(t * act + b * Q8A_BLOCK_BYTES) };
                    let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                    let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                    let mut p = [_mm256_setzero_si256(); STRIP];
                    for (lane, pl) in p.iter_mut().enumerate() {
                        let wv = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                        let aw = _mm256_sign_epi8(wv, wv);
                        let sx = _mm256_sign_epi8(xv, wv);
                        *pl = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                    }
                    let isum = _mm256_cvtepi32_ps(hsum8_i32(p));
                    let dwdx = _mm256_mul_ps(dw, _mm256_set1_ps(dx));
                    *at = _mm256_fmadd_ps(dwdx, isum, *at);
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
        for b in 0..nb {
            let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let wv = unsafe { _mm256_load_si256(g.add(32 + lane * WBLOCK).cast()) };
            for (t, at) in acc.iter_mut().enumerate() {
                let xb = unsafe { xq.add(t * act + b * Q8A_BLOCK_BYTES) };
                let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                let aw = _mm256_sign_epi8(wv, wv);
                let sx = _mm256_sign_epi8(xv, wv);
                let isum = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones));
                *at = (dw * dx).mul_add(isum as f32, *at);
            }
        }
        for (t, at) in acc.iter().enumerate() {
            unsafe { y.add(t * rows + r).write(*at) };
        }
        r += 1;
    }
}
