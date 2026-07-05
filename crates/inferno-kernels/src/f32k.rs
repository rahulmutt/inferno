//! F32 GEMV in the rs8 layout — the trivial baseline that validates the rig
//! itself (spec §Scope). Layout: rows padded to strips of 8; per strip, K
//! columns of 8 consecutive f32 — one aligned 32-byte vector per column.

use crate::{AlignedBuf, KernelError, Result, STRIP};

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
            acc = wv.mul_add(unsafe { x.add(c).read() }, acc);
        }
        unsafe { y.add(r).write(acc) };
    }
}

/// Unified GEMV ABI (all dtypes): `y[row_start..row_end] = W · x`.
///
/// # Safety
/// - `y` valid for f32 writes at indices `row_start..row_end`.
/// - `x` points at the activation buffer — for F32, `k` raw little-endian
///   f32 values (4-byte aligned).
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
            let wv = unsafe { _mm256_load_ps(base.add(c * STRIP)) };
            let xv = _mm256_set1_ps(unsafe { xf.add(c).read() });
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        unsafe { _mm256_storeu_ps(y.add(r), acc) };
        r += STRIP;
    }
    if r < row_end {
        unsafe { gemv_rows(y, xf, wf, k, r, row_end) };
    }
}
