//! F32 GEMV in the rs8 layout — the trivial baseline that validates the rig
//! itself (spec §Scope). Layout: rows padded to strips of 8; per strip, K
//! columns of 8 consecutive f32 — one aligned 32-byte vector per column.

use crate::{AlignedBuf, KernelError, Result, STRIP};

/// Columns to software-prefetch ahead in the AVX2 GEMV (M4b.4). The f32 rs8
/// layout stores one aligned 32-byte vector per column; `PF_DIST_F32` columns
/// ahead ≈ one cache line beyond the current fetch. Plan default (16); the
/// Task 2 sweep was deferred to quiet hardware — see the spec Amendment. Pure
/// hint — never affects output bits.
const PF_DIST_F32: usize = 16;

pub fn packed_len_f32_rs8(rows: usize, k: usize) -> usize {
    rows.next_multiple_of(STRIP) * k * 4
}

pub fn pack_f32_rs8(bytes: &[u8], rows: usize, k: usize) -> Result<AlignedBuf> {
    if rows == 0 {
        return Err(KernelError::ZeroRows);
    }
    if k == 0 {
        return Err(KernelError::BadK { k, block: 1 });
    }
    let expected = rows
        .checked_mul(k)
        .and_then(|n| n.checked_mul(4))
        .ok_or(KernelError::Overflow)?;
    if bytes.len() != expected {
        return Err(KernelError::SizeMismatch {
            what: "f32 weight bytes",
            got: bytes.len(),
            expected,
        });
    }
    let mut out = AlignedBuf::zeroed(packed_len_f32_rs8(rows, k));
    let dst = out.as_mut_slice();
    for r in 0..rows {
        let (strip, lane) = (r / STRIP, r % STRIP);
        for c in 0..k {
            let s = (r * k + c) * 4;
            let d = (((strip * k) + c) * STRIP + lane) * 4;
            dst[d..d + 4].copy_from_slice(&bytes[s..s + 4]);
        }
    }
    Ok(out)
}

/// Scalar row loop. Both ISA symbols route partial strips here so every
/// variant computes the identical fma sequence per row (bitwise contract).
///
/// # Safety
/// Contract of [`inferno_gemv_f32_rs8_scalar`].
unsafe fn gemv_rows(y: *mut f32, x: *const f32, w: *const f32, k: usize, r0: usize, r1: usize) {
    for r in r0..r1 {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let base = unsafe { w.add(strip * k * STRIP + lane) };
        let mut acc = 0f32;
        for c in 0..k {
            let wv = unsafe { base.add(c * STRIP).read() };
            // x may have any alignment (a `&[u8]` subslice cast to f32).
            acc = wv.mul_add(unsafe { x.add(c).read_unaligned() }, acc);
        }
        unsafe { y.add(r).write(acc) };
    }
}

/// Unified GEMV ABI (all dtypes): `y[row_start..row_end] = W · x`.
///
/// # Safety
/// - `y` valid for f32 writes at indices `row_start..row_end`.
/// - `x` points at the activation buffer — for F32, `k` raw little-endian
///   f32 values, any alignment (the loads are unaligned).
/// - `w` is a `pack_f32_rs8` image built with this exact `k` and at least
///   `row_end` rows, 32-byte aligned (guaranteed by `AlignedBuf`).
/// - `row_start <= row_end`; all values finite.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_f32_rs8_scalar(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    unsafe { gemv_rows(y, x.cast(), w.cast(), k, row_start, row_end) }
}

/// # Safety
/// As [`inferno_gemv_f32_rs8_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_f32_rs8_avx2(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let xf = x.cast::<f32>();
    let wf = w.cast::<f32>();
    let mut r = row_start;
    let head = row_start.next_multiple_of(STRIP).min(row_end);
    if head > r {
        unsafe { gemv_rows(y, xf, wf, k, r, head) };
        r = head;
    }
    while r + STRIP <= row_end {
        let base = unsafe { wf.add((r / STRIP) * k * STRIP) };
        let mut acc = _mm256_setzero_ps();
        for c in 0..k {
            // Prefetch a future column vector; `wrapping_add` (not `add`) because
            // the strip's tail columns point past the buffer end. `_mm_prefetch`
            // never dereferences and never faults — pure hint, output unchanged.
            // (See the q8_0 kernel.)
            let pf_addr = base.wrapping_add((c + PF_DIST_F32) * STRIP).cast();
            _mm_prefetch::<_MM_HINT_T0>(pf_addr);
            let wv = unsafe { _mm256_load_ps(base.add(c * STRIP)) };
            // x may have any alignment; broadcast from an unaligned scalar read.
            let xv = _mm256_set1_ps(unsafe { xf.add(c).read_unaligned() });
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        unsafe { _mm256_storeu_ps(y.add(r), acc) };
        r += STRIP;
    }
    if r < row_end {
        unsafe { gemv_rows(y, xf, wf, k, r, row_end) };
    }
}

/// Batched F32 GEMM. Same per-(t,r) fma order as `inferno_gemv_f32_rs8_*`
/// (`gemm(m=1) ≡ gemv`). `xq` is `m` contiguous rows of `k` LE f32.
///
/// # Safety
/// As the F32 GEMV symbol, with `xq` valid for `m*k` f32, `y` for `m*rows`.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemm_f32_rs8_scalar(
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    m: usize,
    rows: usize,
    row_start: usize,
    row_end: usize,
) {
    let x = xq.cast::<f32>();
    let wf = w.cast::<f32>();
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let base = unsafe { wf.add(strip * k * STRIP + lane) };
        let mut acc = vec![0f32; m];
        for c in 0..k {
            let wv = unsafe { base.add(c * STRIP).read() };
            for (t, at) in acc.iter_mut().enumerate() {
                let xv = unsafe { x.add(t * k + c).read_unaligned() };
                *at = wv.mul_add(xv, *at);
            }
        }
        for (t, at) in acc.iter().enumerate() {
            unsafe { y.add(t * rows + r).write(*at) };
        }
    }
}

/// # Safety
/// As [`inferno_gemm_f32_rs8_scalar`]; additionally requires AVX2+FMA.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemm_f32_rs8_avx2(
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
    let x = xq.cast::<f32>();
    let wf = w.cast::<f32>();
    let mut r = row_start;
    let head = row_start.next_multiple_of(STRIP).min(row_end);
    if head > r {
        // Partial head: scalar per-row (bit-identical), one acc per token.
        unsafe { inferno_gemm_f32_rs8_scalar(y, xq, w, k, m, rows, r, head) };
        r = head;
    }
    while r + STRIP <= row_end {
        let base = unsafe { wf.add((r / STRIP) * k * STRIP) };
        let mut acc = vec![_mm256_setzero_ps(); m];
        for c in 0..k {
            let wv = unsafe { _mm256_load_ps(base.add(c * STRIP)) };
            for (t, at) in acc.iter_mut().enumerate() {
                let xv = _mm256_set1_ps(unsafe { x.add(t * k + c).read_unaligned() });
                *at = _mm256_fmadd_ps(wv, xv, *at);
            }
        }
        for (t, at) in acc.iter().enumerate() {
            unsafe { _mm256_storeu_ps(y.add(t * rows + r), *at) };
        }
        r += STRIP;
    }
    if r < row_end {
        unsafe { inferno_gemm_f32_rs8_scalar(y, xq, w, k, m, rows, r, row_end) };
    }
}
