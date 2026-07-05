//! Kernel-vs-oracle rig (spec §Testing): every kernel is compared against
//! `inferno_formats::quant::dequant` + the scalar reference matmul, ISA
//! variants are compared bitwise, and row-range partitioning is bit-stable.

use inferno_formats::{DType, quant};
use inferno_graph::Tensor;
use inferno_graph::tolerance::gemv_rel_tol;
use inferno_kernels::{KernelIsa, f32k};
use proptest::prelude::*;

/// Deterministic pseudo-random f32s in [-1, 1).
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

/// Trusted reference: dequantize the same file-order weight bytes the kernel
/// packed, then the obviously-correct scalar matmul.
fn oracle(dtype: &DType, wbytes: &[u8], rows: usize, k: usize, x: &[f32]) -> Vec<f32> {
    let wf = quant::dequant(dtype, wbytes, rows * k).unwrap();
    let xt = Tensor {
        shape: vec![1, k],
        data: x.to_vec(),
    };
    inferno_graph::ops::matmul(&xt, &wf, rows, k, None).data
}

fn assert_close(dtype: &DType, got: &[f32], want: &[f32]) {
    let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));
    let tol = gemv_rel_tol(dtype) * scale;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "row {i}: got {g}, want {w} (tol {tol})"
        );
    }
}

// ---------- F32 ----------

fn gemv_f32(
    isa: KernelIsa,
    w: &inferno_kernels::AlignedBuf,
    x: &[f32],
    rows: usize,
    k: usize,
) -> Vec<f32> {
    let mut y = vec![f32::NAN; rows];
    let xb: &[u8] = bytemuck_free_cast(x); // see helper below
    // SAFETY: w is a pack_f32_rs8 image for (rows, k); x has k f32; y has rows.
    unsafe {
        match isa {
            KernelIsa::Scalar => inferno_kernels::inferno_gemv_f32_rs8_scalar(
                y.as_mut_ptr(),
                xb.as_ptr(),
                w.as_ptr(),
                k,
                0,
                rows,
            ),
            KernelIsa::Avx2 => inferno_kernels::inferno_gemv_f32_rs8_avx2(
                y.as_mut_ptr(),
                xb.as_ptr(),
                w.as_ptr(),
                k,
                0,
                rows,
            ),
        }
    }
    y
}

/// f32 slice → its little-endian bytes (test-only; no bytemuck dep needed).
fn bytemuck_free_cast(x: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; alignment shrinks; lifetime tied to input.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len() * 4) }
}

proptest! {
    #[test]
    fn f32_gemv_matches_oracle(seed in any::<u64>(), rows in 1usize..20, k in 1usize..48) {
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 0x9e3779b97f4a7c15, k);
        let wbytes = quant::pack(&DType::F32, &vals).unwrap();
        let w = f32k::pack_f32_rs8(&wbytes, rows, k).unwrap();
        let want = oracle(&DType::F32, &wbytes, rows, k, &x);
        for isa in KernelIsa::all_available() {
            assert_close(&DType::F32, &gemv_f32(isa, &w, &x, rows, k), &want);
        }
    }

    #[test]
    fn f32_isa_variants_bitwise_equal(seed in any::<u64>(), rows in 1usize..20, k in 1usize..48) {
        if !KernelIsa::Avx2.available() { return Ok(()); }
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 1, k);
        let w = f32k::pack_f32_rs8(&quant::pack(&DType::F32, &vals).unwrap(), rows, k).unwrap();
        let a = gemv_f32(KernelIsa::Scalar, &w, &x, rows, k);
        let b = gemv_f32(KernelIsa::Avx2, &w, &x, rows, k);
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
        }
    }

    /// GEMV over [0, rows) must equal any two-part split, bitwise — the
    /// property M3's thread partitioning relies on.
    #[test]
    fn f32_range_partition_bitwise(seed in any::<u64>(), rows in 2usize..24, k in 1usize..32) {
        let split = (seed % rows as u64) as usize;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 2, k);
        let w = f32k::pack_f32_rs8(&quant::pack(&DType::F32, &vals).unwrap(), rows, k).unwrap();
        let full = gemv_f32(KernelIsa::Scalar, &w, &x, rows, k);
        for isa in KernelIsa::all_available() {
            let mut y = vec![f32::NAN; rows];
            let xb = bytemuck_free_cast(&x);
            // SAFETY: as gemv_f32, split ranges stay within rows.
            unsafe {
                let f = match isa {
                    KernelIsa::Scalar => inferno_kernels::inferno_gemv_f32_rs8_scalar
                        as unsafe extern "C" fn(*mut f32, *const u8, *const u8, usize, usize, usize),
                    KernelIsa::Avx2 => inferno_kernels::inferno_gemv_f32_rs8_avx2,
                };
                f(y.as_mut_ptr(), xb.as_ptr(), w.as_ptr(), k, 0, split);
                f(y.as_mut_ptr(), xb.as_ptr(), w.as_ptr(), k, split, rows);
            }
            for (i, (a, b)) in full.iter().zip(&y).enumerate() {
                prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
            }
        }
    }
}

#[test]
// `8usize / 8 * k` spells out "row block 1 of the padded region" to mirror
// the packing formula; clippy's identity_op lint doesn't see the intent.
#[allow(clippy::identity_op)]
fn f32_pack_inverse() {
    let rows = 11; // partial strip
    let k = 7;
    let vals = pseudo(3, rows * k);
    let bytes = quant::pack(&DType::F32, &vals).unwrap();
    let w = f32k::pack_f32_rs8(&bytes, rows, k).unwrap();
    // Unpack: read each (row, col) back out of the strip layout.
    let p = w.as_slice();
    for r in 0..rows {
        for c in 0..k {
            let off = (((r / 8) * k + c) * 8 + r % 8) * 4;
            let got = f32::from_le_bytes(p[off..off + 4].try_into().unwrap());
            assert_eq!(got.to_bits(), vals[r * k + c].to_bits(), "({r},{c})");
        }
    }
    // Padding rows are zero.
    assert_eq!(w.len(), f32k::packed_len_f32_rs8(rows, k));
    for c in 0..k {
        for lane in 3..8 {
            // rows 11..16
            let off = ((8usize / 8 * k + c) * 8 + lane) * 4;
            assert_eq!(&p[off..off + 4], &[0u8; 4]);
        }
    }
}

#[test]
fn f32_empty_range_is_noop_and_pack_validates() {
    let vals = pseudo(4, 8 * 4);
    let w = f32k::pack_f32_rs8(&quant::pack(&DType::F32, &vals).unwrap(), 8, 4).unwrap();
    let x = pseudo(5, 4);
    let mut y = vec![42f32; 8];
    // SAFETY: valid image; empty range must not touch y.
    unsafe {
        inferno_kernels::inferno_gemv_f32_rs8_scalar(
            y.as_mut_ptr(),
            bytemuck_free_cast(&x).as_ptr(),
            w.as_ptr(),
            4,
            5,
            5,
        );
    }
    assert!(y.iter().all(|&v| v == 42.0));
    assert!(f32k::pack_f32_rs8(&[0u8; 12], 2, 2).is_err()); // 12 != 16
    assert!(f32k::pack_f32_rs8(&[], 0, 4).is_err());
    assert!(f32k::pack_f32_rs8(&[], 4, 0).is_err());
}
