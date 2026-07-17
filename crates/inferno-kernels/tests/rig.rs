//! Kernel-vs-oracle rig (spec §Testing): every kernel is compared against
//! `inferno_formats::quant::dequant` + the scalar reference matmul, ISA
//! variants are compared bitwise, and row-range partitioning is bit-stable.

use inferno_formats::{DType, quant};
use inferno_graph::Tensor;
use inferno_graph::tolerance::{attn_rel_tol, gemv_rel_tol};
use inferno_kernels::{KernelIsa, act, f32k, q8_0};
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

/// Decode a q8a buffer (act.rs layout: per 36-byte block, `d: f32 LE` then
/// 32 × `i8` qs) back to f32 — mirrors act.rs's private `#[cfg(test)]`
/// helper, duplicated here because integration tests can't see it.
fn decode_q8a(buf: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    for b in buf.chunks_exact(36) {
        let d = f32::from_le_bytes(b[..4].try_into().unwrap());
        out.extend(b[4..36].iter().map(|&q| d * f32::from(q as i8)));
    }
    out
}

/// Decode a q8k buffer (act.rs layout: per 292-byte block, `d: f32 LE`,
/// 256 × `i8` qs, then 8 × `i32` bsums, ignored here) back to f32.
fn decode_q8k(buf: &[u8]) -> Vec<f32> {
    let mut out = Vec::new();
    for b in buf.chunks_exact(292) {
        let d = f32::from_le_bytes(b[..4].try_into().unwrap());
        out.extend(b[4..260].iter().map(|&q| d * f32::from(q as i8)));
    }
    out
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

#[allow(clippy::too_many_arguments)]
fn gemm_f32(
    isa: KernelIsa,
    w: &inferno_kernels::AlignedBuf,
    xq_panel: &[u8],
    k: usize,
    m: usize,
    rows: usize,
    range: (usize, usize),
    y: &mut [f32],
) {
    // SAFETY: w is a pack_f32_rs8 image for (rows, k); xq_panel is m rows of k
    // LE f32, contiguous; y has m*rows elements; range within rows.
    unsafe {
        match isa {
            KernelIsa::Scalar => inferno_kernels::inferno_gemm_f32_rs8_scalar(
                y.as_mut_ptr(),
                xq_panel.as_ptr(),
                w.as_ptr(),
                k,
                m,
                rows,
                range.0,
                range.1,
            ),
            KernelIsa::Avx2 => inferno_kernels::inferno_gemm_f32_rs8_avx2(
                y.as_mut_ptr(),
                xq_panel.as_ptr(),
                w.as_ptr(),
                k,
                m,
                rows,
                range.0,
                range.1,
            ),
        }
    }
}

proptest! {
    /// gemm(m=1) is bit-identical to gemv over the same range.
    #[test]
    fn f32_gemm_m1_equals_gemv(seed in any::<u64>(), rows in 1usize..20, k in 1usize..48) {
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 0xa1, k);
        let w = f32k::pack_f32_rs8(&quant::pack(&DType::F32, &vals).unwrap(), rows, k).unwrap();
        let panel = bytemuck_free_cast(&x);
        for isa in KernelIsa::all_available() {
            let yv = gemv_f32(isa, &w, &x, rows, k);
            let mut yg = vec![f32::NAN; rows];
            gemm_f32(isa, &w, panel, k, 1, rows, (0, rows), &mut yg);
            for (i, (a, b)) in yv.iter().zip(&yg).enumerate() {
                prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
            }
        }
    }

    /// Each token in an m-row panel matches an independent gemv on that token.
    #[test]
    fn f32_gemm_rows_match_per_token_gemv(seed in any::<u64>(), rows in 1usize..16, k in 1usize..40, m in 1usize..6) {
        let vals = pseudo(seed, rows * k);
        let w = f32k::pack_f32_rs8(&quant::pack(&DType::F32, &vals).unwrap(), rows, k).unwrap();
        let mut panel_f = Vec::new();
        let mut per_token = Vec::new();
        for t in 0..m {
            let x = pseudo(seed ^ (0x100 + t as u64), k);
            per_token.push(gemv_f32(KernelIsa::Scalar, &w, &x, rows, k));
            panel_f.extend_from_slice(&x);
        }
        let panel = bytemuck_free_cast(&panel_f);
        for isa in KernelIsa::all_available() {
            let mut yg = vec![f32::NAN; m * rows];
            gemm_f32(isa, &w, panel, k, m, rows, (0, rows), &mut yg);
            for t in 0..m {
                for r in 0..rows {
                    prop_assert_eq!(yg[t * rows + r].to_bits(), per_token[t][r].to_bits(), "t{} r{}", t, r);
                }
            }
        }
    }

    /// Row-range partitioning is bit-stable.
    #[test]
    fn f32_gemm_range_partition_bitwise(seed in any::<u64>(), rows in 2usize..24, k in 1usize..32, m in 1usize..4) {
        let split = (seed % rows as u64) as usize;
        let vals = pseudo(seed, rows * k);
        let w = f32k::pack_f32_rs8(&quant::pack(&DType::F32, &vals).unwrap(), rows, k).unwrap();
        let mut panel_f = Vec::new();
        for t in 0..m {
            panel_f.extend_from_slice(&pseudo(seed ^ (0x200 + t as u64), k));
        }
        let panel = bytemuck_free_cast(&panel_f);
        for isa in KernelIsa::all_available() {
            let mut full = vec![f32::NAN; m * rows];
            gemm_f32(isa, &w, panel, k, m, rows, (0, rows), &mut full);
            let mut split_y = vec![f32::NAN; m * rows];
            gemm_f32(isa, &w, panel, k, m, rows, (0, split), &mut split_y);
            gemm_f32(isa, &w, panel, k, m, rows, (split, rows), &mut split_y);
            for i in 0..m * rows {
                prop_assert_eq!(full[i].to_bits(), split_y[i].to_bits(), "i {}", i);
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

// ---------- Q8_0 ----------

fn gemv_q8_0(
    isa: KernelIsa,
    w: &inferno_kernels::AlignedBuf,
    xq: &[u8],
    _rows: usize,
    k: usize,
    range: (usize, usize),
    y: &mut [f32],
) {
    // SAFETY: w is a pack_q8_0_rs8 image for (rows, k); xq is a q8a buffer
    // for k; y has rows elements; range within rows.
    unsafe {
        match isa {
            KernelIsa::Scalar => inferno_kernels::inferno_gemv_q8_0_rs8_scalar(
                y.as_mut_ptr(),
                xq.as_ptr(),
                w.as_ptr(),
                k,
                range.0,
                range.1,
            ),
            KernelIsa::Avx2 => inferno_kernels::inferno_gemv_q8_0_rs8_avx2(
                y.as_mut_ptr(),
                xq.as_ptr(),
                w.as_ptr(),
                k,
                range.0,
                range.1,
            ),
        }
    }
}

proptest! {
    #[test]
    fn q8_0_gemv_matches_oracle(seed in any::<u64>(), rows in 1usize..20, nb in 1usize..5) {
        let k = nb * 32;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 0xabcdef, k);
        let wbytes = quant::pack(&DType::Q8_0, &vals).unwrap();
        let w = q8_0::pack_q8_0_rs8(&wbytes, rows, k).unwrap();
        let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
        // Oracle consumes the same quantized weights AND the same
        // kernel-quantized activations (x_hat, decoded from xq) as the
        // kernel itself, so gemv_rel_tol only has to bound
        // accumulation-order/fma rounding differences, not the much larger
        // activation-quantization noise tail (see tolerance.rs doc comment
        // for why comparing against the raw f32 activations was abandoned; a
        // 2026-07-05 investigation on the dev Ryzen 9 3900 measured that tail).
        let x_hat = decode_q8a(&xq);
        let want = oracle(&DType::Q8_0, &wbytes, rows, k, &x_hat);
        for isa in KernelIsa::all_available() {
            let mut y = vec![f32::NAN; rows];
            gemv_q8_0(isa, &w, &xq, rows, k, (0, rows), &mut y);
            assert_close(&DType::Q8_0, &y, &want);
        }
    }

    #[test]
    fn q8_0_isa_variants_bitwise_equal(seed in any::<u64>(), rows in 1usize..20, nb in 1usize..5) {
        if !KernelIsa::Avx2.available() { return Ok(()); }
        let k = nb * 32;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 3, k);
        let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
        let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
        let (mut a, mut b) = (vec![f32::NAN; rows], vec![f32::NAN; rows]);
        gemv_q8_0(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut a);
        gemv_q8_0(KernelIsa::Avx2, &w, &xq, rows, k, (0, rows), &mut b);
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
        }
    }

    #[test]
    fn q8_0_range_partition_bitwise(seed in any::<u64>(), rows in 2usize..24, nb in 1usize..4) {
        let k = nb * 32;
        let split = (seed % rows as u64) as usize;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 4, k);
        let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
        let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
        let mut full = vec![f32::NAN; rows];
        gemv_q8_0(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut full);
        for isa in KernelIsa::all_available() {
            let mut y = vec![f32::NAN; rows];
            gemv_q8_0(isa, &w, &xq, rows, k, (0, split), &mut y);
            gemv_q8_0(isa, &w, &xq, rows, k, (split, rows), &mut y);
            for (i, (a, b)) in full.iter().zip(&y).enumerate() {
                prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_q8_0(
    isa: KernelIsa,
    w: &inferno_kernels::AlignedBuf,
    xq_panel: &[u8],
    k: usize,
    m: usize,
    rows: usize,
    range: (usize, usize),
    y: &mut [f32],
) {
    // SAFETY: w is a pack_q8_0_rs8 image for (rows, k); xq_panel is m q8a rows
    // for k, contiguous; y has m*rows elements; range within rows.
    unsafe {
        match isa {
            KernelIsa::Scalar => inferno_kernels::inferno_gemm_q8_0_rs8_scalar(
                y.as_mut_ptr(),
                xq_panel.as_ptr(),
                w.as_ptr(),
                k,
                m,
                rows,
                range.0,
                range.1,
            ),
            KernelIsa::Avx2 => inferno_kernels::inferno_gemm_q8_0_rs8_avx2(
                y.as_mut_ptr(),
                xq_panel.as_ptr(),
                w.as_ptr(),
                k,
                m,
                rows,
                range.0,
                range.1,
            ),
        }
    }
}

proptest! {
    /// gemm(m=1) is bit-identical to gemv over the same range.
    #[test]
    fn q8_0_gemm_m1_equals_gemv(seed in any::<u64>(), rows in 1usize..20, nb in 1usize..5) {
        let k = nb * 32;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 0xa1, k);
        let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
        let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
        for isa in KernelIsa::all_available() {
            let mut yv = vec![f32::NAN; rows];
            gemv_q8_0(isa, &w, &xq, rows, k, (0, rows), &mut yv);
            let mut yg = vec![f32::NAN; rows];
            gemm_q8_0(isa, &w, &xq, k, 1, rows, (0, rows), &mut yg);
            for (i, (a, b)) in yv.iter().zip(&yg).enumerate() {
                prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
            }
        }
    }

    /// Each token in an m-row panel matches an independent gemv on that token.
    #[test]
    fn q8_0_gemm_rows_match_per_token_gemv(seed in any::<u64>(), rows in 1usize..16, nb in 1usize..4, m in 1usize..20) {
        let k = nb * 32;
        let vals = pseudo(seed, rows * k);
        let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
        let mut panel = Vec::new();
        let mut per_token = Vec::new();
        for t in 0..m {
            let x = pseudo(seed ^ (0x100 + t as u64), k);
            let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
            panel.extend_from_slice(&xq);
            let mut yv = vec![f32::NAN; rows];
            gemv_q8_0(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut yv);
            per_token.push(yv);
        }
        for isa in KernelIsa::all_available() {
            let mut yg = vec![f32::NAN; m * rows];
            gemm_q8_0(isa, &w, &panel, k, m, rows, (0, rows), &mut yg);
            for t in 0..m {
                for r in 0..rows {
                    prop_assert_eq!(yg[t * rows + r].to_bits(), per_token[t][r].to_bits(), "t{} r{}", t, r);
                }
            }
        }
    }

    /// Row-range partitioning is bit-stable (the property par_gemm relies on).
    #[test]
    fn q8_0_gemm_range_partition_bitwise(seed in any::<u64>(), rows in 2usize..24, nb in 1usize..4, m in 1usize..20) {
        let k = nb * 32;
        let split = (seed % rows as u64) as usize;
        let vals = pseudo(seed, rows * k);
        let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
        let mut panel = Vec::new();
        for t in 0..m {
            let x = pseudo(seed ^ (0x200 + t as u64), k);
            panel.extend_from_slice(&act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap());
        }
        for isa in KernelIsa::all_available() {
            let mut full = vec![f32::NAN; m * rows];
            gemm_q8_0(isa, &w, &panel, k, m, rows, (0, rows), &mut full);
            let mut split_y = vec![f32::NAN; m * rows];
            gemm_q8_0(isa, &w, &panel, k, m, rows, (0, split), &mut split_y);
            gemm_q8_0(isa, &w, &panel, k, m, rows, (split, rows), &mut split_y);
            for i in 0..m * rows {
                prop_assert_eq!(full[i].to_bits(), split_y[i].to_bits(), "i {}", i);
            }
        }
    }
}

/// M4b.13: a PREFILL_TILE-shaped panel (m = 64) crossing many register
/// tiles, with rows spanning full strips plus a partial tail — the shape
/// the tiled fast path sees in production. Every token must bit-equal an
/// independent scalar gemv on that token, on every runnable ISA.
#[test]
fn q8_0_gemm_prefill_tile_matches_per_token_gemv() {
    let (rows, k, m) = (28usize, 128usize, 64usize);
    let vals = pseudo(7, rows * k);
    let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
    let mut panel = Vec::new();
    let mut per_token = Vec::new();
    for t in 0..m {
        let x = pseudo(0x300 + t as u64, k);
        let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
        panel.extend_from_slice(&xq);
        let mut yv = vec![f32::NAN; rows];
        gemv_q8_0(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut yv);
        per_token.push(yv);
    }
    for isa in KernelIsa::all_available() {
        let mut yg = vec![f32::NAN; m * rows];
        gemm_q8_0(isa, &w, &panel, k, m, rows, (0, rows), &mut yg);
        for t in 0..m {
            for r in 0..rows {
                assert_eq!(
                    yg[t * rows + r].to_bits(),
                    per_token[t][r].to_bits(),
                    "t{t} r{r} isa {isa:?}"
                );
            }
        }
    }
}

/// Pack inverse via normalized blocks: parse the file bytes and the packed
/// image to the same (d, qs) structure — localizes layout bugs (spec §Testing).
#[test]
fn q8_0_pack_inverse() {
    let (rows, k) = (11usize, 64usize); // partial strip, 2 blocks
    let nb = k / 32;
    let vals = pseudo(7, rows * k);
    let bytes = quant::pack(&DType::Q8_0, &vals).unwrap();
    let w = q8_0::pack_q8_0_rs8(&bytes, rows, k).unwrap();
    let p = w.as_slice();
    for r in 0..rows {
        for b in 0..nb {
            let s = (r * nb + b) * 34;
            let file_d = quant::f16_to_f32(u16::from_le_bytes([bytes[s], bytes[s + 1]]));
            let file_qs = &bytes[s + 2..s + 34];
            let g = ((r / 8) * nb + b) * 288;
            let lane = r % 8;
            let packed_d =
                f32::from_le_bytes(p[g + lane * 4..g + lane * 4 + 4].try_into().unwrap());
            let packed_qs = &p[g + 32 + lane * 32..g + 32 + (lane + 1) * 32];
            assert_eq!(packed_d.to_bits(), file_d.to_bits(), "({r},{b}) d");
            assert_eq!(packed_qs, file_qs, "({r},{b}) qs");
        }
    }
}

#[test]
fn q8_0_pack_clamps_minus_128() {
    // Hand-build one block whose qs are all -128 (hostile file).
    let mut bytes = vec![0u8; 34];
    bytes[..2].copy_from_slice(&quant::f32_to_f16(1.0).to_le_bytes());
    for b in &mut bytes[2..] {
        *b = (-128i8) as u8;
    }
    let w = q8_0::pack_q8_0_rs8(&bytes, 1, 32).unwrap();
    let p = w.as_slice();
    for i in 0..32 {
        assert_eq!(p[32 + i] as i8, -127);
    }
}

/// Max-scale block (spec edge case): every value at the block amax, so
/// quantized weights and activations all saturate to ±127. The oracle
/// consumes the same kernel-quantized activations (x_hat) as the kernel, so
/// this only checks accumulation-order rounding, not quantization noise.
#[test]
fn q8_0_saturated_block_matches_oracle() {
    let (rows, k) = (3usize, 32usize);
    let vals: Vec<f32> = (0..rows * k)
        .map(|i| if i % 2 == 0 { 10.0 } else { -10.0 })
        .collect();
    let x: Vec<f32> = (0..k)
        .map(|i| if i % 2 == 0 { 8.0 } else { -8.0 })
        .collect();
    let wbytes = quant::pack(&DType::Q8_0, &vals).unwrap();
    let w = q8_0::pack_q8_0_rs8(&wbytes, rows, k).unwrap();
    let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
    let x_hat = decode_q8a(&xq);
    let want = oracle(&DType::Q8_0, &wbytes, rows, k, &x_hat);
    for isa in KernelIsa::all_available() {
        let mut y = vec![f32::NAN; rows];
        gemv_q8_0(isa, &w, &xq, rows, k, (0, rows), &mut y);
        assert_close(&DType::Q8_0, &y, &want);
    }
}

#[test]
fn q8_0_pack_validates() {
    assert!(q8_0::pack_q8_0_rs8(&[0u8; 34], 1, 31).is_err()); // k not multiple of 32
    assert!(q8_0::pack_q8_0_rs8(&[0u8; 33], 1, 32).is_err()); // wrong byte count
    assert!(q8_0::pack_q8_0_rs8(&[], 0, 32).is_err());
}

/// One (seed, rows, k) case's max relative error (same normalization as
/// `assert_close`: scale = max(1, max|want|)).
fn q8_0_case_max_rel(seed: u64, rows: usize, k: usize) -> f32 {
    let vals = pseudo(seed, rows * k);
    let x = pseudo(seed ^ 99, k);
    let wbytes = quant::pack(&DType::Q8_0, &vals).unwrap();
    let w = q8_0::pack_q8_0_rs8(&wbytes, rows, k).unwrap();
    let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
    let want = oracle(&DType::Q8_0, &wbytes, rows, k, &x);
    let mut y = vec![f32::NAN; rows];
    gemv_q8_0(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut y);
    let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));
    let mut max_rel = 0f32;
    for (g, w_) in y.iter().zip(&want) {
        max_rel = max_rel.max((g - w_).abs() / scale);
    }
    max_rel
}

/// Ignored diagnostic: end-to-end noise instrument. Unlike
/// `q8_0_gemv_matches_oracle` (which compares against an oracle fed the
/// kernel's own quantized activations, so it only measures rounding-order
/// error and directly informs `gemv_rel_tol`), this measures the oracle
/// against the RAW f32 activations — the full real-world error a caller
/// sees comparing a quantized kernel's output to an unquantized reference,
/// dominated by activation-quantization noise. Kept as a measurement tool,
/// not a tolerance source; see `tolerance.rs`'s `gemv_rel_tol` doc comment
/// for the 2026-07-05 data this produced (Q8_0 3.37e-2 @2k seeds → 6.25e-2
/// @500k) and why that noise, not rounding error, made a constant-tolerance
/// oracle-vs-raw-x comparison unworkable. Sweeps both the original fixed
/// shape (rows=16, k=128) AND the property tests' actual shape distribution
/// (rows 1..20, nb 1..5, k = 32*nb) — the fixed shape alone under-observes
/// because it never hits the small-k, small-rows corner where
/// activation-quant noise doesn't average out.
/// Run: cargo nextest run -p inferno-kernels --run-ignored all observed_error_q8_0 --no-capture
#[test]
#[ignore = "diagnostic; prints observed gemv error distribution"]
fn observed_error_q8_0() {
    let mut overall_max = 0f32;

    let mut fixed_max = 0f32;
    for seed in 0..500u64 {
        fixed_max = fixed_max.max(q8_0_case_max_rel(seed, 16, 128));
    }
    println!("q8_0 [fixed rows=16 k=128, 500 seeds] max rel error: {fixed_max:e}");
    overall_max = overall_max.max(fixed_max);

    // Property-test shape distribution: rows 1..20, nb 1..5 (k = 32*nb).
    const SEEDS_PER_SHAPE: u64 = 30; // 19 * 4 * 30 = 2280 total cases
    let mut total_cases = 0u64;
    for rows in 1usize..20 {
        for nb in 1usize..5 {
            let k = nb * 32;
            let mut shape_max = 0f32;
            for seed in 0..SEEDS_PER_SHAPE {
                shape_max = shape_max.max(q8_0_case_max_rel(seed, rows, k));
                total_cases += 1;
            }
            println!("q8_0 [rows={rows} nb={nb} k={k}] max rel error: {shape_max:e}");
            overall_max = overall_max.max(shape_max);
        }
    }
    println!(
        "q8_0 observed OVERALL max rel error: {overall_max:e} over {total_cases} swept cases (+500 fixed-shape) (tol {:e})",
        gemv_rel_tol(&DType::Q8_0)
    );
}

// ---------- Q4_K ----------

use inferno_kernels::q4_k;

fn gemv_q4_k(
    isa: KernelIsa,
    w: &inferno_kernels::AlignedBuf,
    xq: &[u8],
    _rows: usize,
    k: usize,
    range: (usize, usize),
    y: &mut [f32],
) {
    // SAFETY: w is a pack_q4_k_rs8 image for (rows, k); xq is a q8k buffer
    // for k; y has rows elements; range within rows.
    unsafe {
        match isa {
            KernelIsa::Scalar => inferno_kernels::inferno_gemv_q4_k_rs8_scalar(
                y.as_mut_ptr(),
                xq.as_ptr(),
                w.as_ptr(),
                k,
                range.0,
                range.1,
            ),
            KernelIsa::Avx2 => inferno_kernels::inferno_gemv_q4_k_rs8_avx2(
                y.as_mut_ptr(),
                xq.as_ptr(),
                w.as_ptr(),
                k,
                range.0,
                range.1,
            ),
        }
    }
}

proptest! {
    #[test]
    fn q4_k_gemv_matches_oracle(seed in any::<u64>(), rows in 1usize..20, nsb in 1usize..3) {
        let k = nsb * 256;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 0x51ed, k);
        let wbytes = quant::pack(&DType::Q4_K, &vals).unwrap();
        let w = q4_k::pack_q4_k_rs8(&wbytes, rows, k).unwrap();
        let xq = act::quantize_row_q8k(KernelIsa::Scalar, &x).unwrap();
        // Oracle consumes the same quantized weights AND the same
        // kernel-quantized activations (x_hat, decoded from xq) as the
        // kernel itself, so gemv_rel_tol only has to bound
        // accumulation-order/fma rounding differences, not the much larger
        // activation-quantization noise tail (see tolerance.rs doc comment
        // for why comparing against the raw f32 activations was abandoned; a
        // 2026-07-05 investigation on the dev Ryzen 9 3900 measured that tail).
        let x_hat = decode_q8k(&xq);
        let want = oracle(&DType::Q4_K, &wbytes, rows, k, &x_hat);
        for isa in KernelIsa::all_available() {
            let mut y = vec![f32::NAN; rows];
            gemv_q4_k(isa, &w, &xq, rows, k, (0, rows), &mut y);
            assert_close(&DType::Q4_K, &y, &want);
        }
    }

    #[test]
    fn q4_k_isa_variants_bitwise_equal(seed in any::<u64>(), rows in 1usize..20) {
        if !KernelIsa::Avx2.available() { return Ok(()); }
        let k = 512usize;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 5, k);
        let w = q4_k::pack_q4_k_rs8(&quant::pack(&DType::Q4_K, &vals).unwrap(), rows, k).unwrap();
        let xq = act::quantize_row_q8k(KernelIsa::Scalar, &x).unwrap();
        let (mut a, mut b) = (vec![f32::NAN; rows], vec![f32::NAN; rows]);
        gemv_q4_k(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut a);
        gemv_q4_k(KernelIsa::Avx2, &w, &xq, rows, k, (0, rows), &mut b);
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
        }
    }

    #[test]
    fn q4_k_range_partition_bitwise(seed in any::<u64>(), rows in 2usize..24) {
        let k = 256usize;
        let split = (seed % rows as u64) as usize;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 6, k);
        let w = q4_k::pack_q4_k_rs8(&quant::pack(&DType::Q4_K, &vals).unwrap(), rows, k).unwrap();
        let xq = act::quantize_row_q8k(KernelIsa::Scalar, &x).unwrap();
        let mut full = vec![f32::NAN; rows];
        gemv_q4_k(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut full);
        for isa in KernelIsa::all_available() {
            let mut y = vec![f32::NAN; rows];
            gemv_q4_k(isa, &w, &xq, rows, k, (0, split), &mut y);
            gemv_q4_k(isa, &w, &xq, rows, k, (split, rows), &mut y);
            for (i, (a, b)) in full.iter().zip(&y).enumerate() {
                prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_q4_k(
    isa: KernelIsa,
    w: &inferno_kernels::AlignedBuf,
    xq_panel: &[u8],
    k: usize,
    m: usize,
    rows: usize,
    range: (usize, usize),
    y: &mut [f32],
) {
    // SAFETY: w is a pack_q4_k_rs8 image for (rows, k); xq_panel is m q8k rows
    // for k, contiguous; y has m*rows elements; range within rows.
    unsafe {
        match isa {
            KernelIsa::Scalar => inferno_kernels::inferno_gemm_q4_k_rs8_scalar(
                y.as_mut_ptr(),
                xq_panel.as_ptr(),
                w.as_ptr(),
                k,
                m,
                rows,
                range.0,
                range.1,
            ),
            KernelIsa::Avx2 => inferno_kernels::inferno_gemm_q4_k_rs8_avx2(
                y.as_mut_ptr(),
                xq_panel.as_ptr(),
                w.as_ptr(),
                k,
                m,
                rows,
                range.0,
                range.1,
            ),
        }
    }
}

proptest! {
    /// gemm(m=1) is bit-identical to gemv over the same range.
    #[test]
    fn q4_k_gemm_m1_equals_gemv(seed in any::<u64>(), rows in 1usize..20, nsb in 1usize..3) {
        let k = nsb * 256;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 0xa1, k);
        let w = q4_k::pack_q4_k_rs8(&quant::pack(&DType::Q4_K, &vals).unwrap(), rows, k).unwrap();
        let xq = act::quantize_row_q8k(KernelIsa::Scalar, &x).unwrap();
        for isa in KernelIsa::all_available() {
            let mut yv = vec![f32::NAN; rows];
            gemv_q4_k(isa, &w, &xq, rows, k, (0, rows), &mut yv);
            let mut yg = vec![f32::NAN; rows];
            gemm_q4_k(isa, &w, &xq, k, 1, rows, (0, rows), &mut yg);
            for (i, (a, b)) in yv.iter().zip(&yg).enumerate() {
                prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
            }
        }
    }

    /// Each token in an m-row panel matches an independent gemv on that token.
    #[test]
    fn q4_k_gemm_rows_match_per_token_gemv(seed in any::<u64>(), rows in 1usize..16, nsb in 1usize..3, m in 1usize..5) {
        let k = nsb * 256;
        let vals = pseudo(seed, rows * k);
        let w = q4_k::pack_q4_k_rs8(&quant::pack(&DType::Q4_K, &vals).unwrap(), rows, k).unwrap();
        let mut panel = Vec::new();
        let mut per_token = Vec::new();
        for t in 0..m {
            let x = pseudo(seed ^ (0x100 + t as u64), k);
            let xq = act::quantize_row_q8k(KernelIsa::Scalar, &x).unwrap();
            panel.extend_from_slice(&xq);
            let mut yv = vec![f32::NAN; rows];
            gemv_q4_k(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut yv);
            per_token.push(yv);
        }
        for isa in KernelIsa::all_available() {
            let mut yg = vec![f32::NAN; m * rows];
            gemm_q4_k(isa, &w, &panel, k, m, rows, (0, rows), &mut yg);
            for t in 0..m {
                for r in 0..rows {
                    prop_assert_eq!(yg[t * rows + r].to_bits(), per_token[t][r].to_bits(), "t{} r{}", t, r);
                }
            }
        }
    }

    /// Row-range partitioning is bit-stable.
    #[test]
    fn q4_k_gemm_range_partition_bitwise(seed in any::<u64>(), rows in 2usize..24, nsb in 1usize..3, m in 1usize..3) {
        let k = nsb * 256;
        let split = (seed % rows as u64) as usize;
        let vals = pseudo(seed, rows * k);
        let w = q4_k::pack_q4_k_rs8(&quant::pack(&DType::Q4_K, &vals).unwrap(), rows, k).unwrap();
        let mut panel = Vec::new();
        for t in 0..m {
            let x = pseudo(seed ^ (0x200 + t as u64), k);
            panel.extend_from_slice(&act::quantize_row_q8k(KernelIsa::Scalar, &x).unwrap());
        }
        for isa in KernelIsa::all_available() {
            let mut full = vec![f32::NAN; m * rows];
            gemm_q4_k(isa, &w, &panel, k, m, rows, (0, rows), &mut full);
            let mut split_y = vec![f32::NAN; m * rows];
            gemm_q4_k(isa, &w, &panel, k, m, rows, (0, split), &mut split_y);
            gemm_q4_k(isa, &w, &panel, k, m, rows, (split, rows), &mut split_y);
            for i in 0..m * rows {
                prop_assert_eq!(full[i].to_bits(), split_y[i].to_bits(), "i {}", i);
            }
        }
    }
}

/// Pack inverse via normalized super-blocks (spec §Testing).
#[test]
fn q4_k_pack_inverse() {
    use inferno_formats::quant::get_scale_min_k4;
    let (rows, k) = (9usize, 256usize);
    let vals = pseudo(11, rows * k);
    let bytes = quant::pack(&DType::Q4_K, &vals).unwrap();
    let w = q4_k::pack_q4_k_rs8(&bytes, rows, k).unwrap();
    let p = w.as_slice();
    for r in 0..rows {
        let s = r * 144; // one super-block per row at k=256
        let file_d = quant::f16_to_f32(u16::from_le_bytes([bytes[s], bytes[s + 1]]));
        let file_dmin = quant::f16_to_f32(u16::from_le_bytes([bytes[s + 2], bytes[s + 3]]));
        let g = (r / 8) * 1216;
        let lane = r % 8;
        let pd = f32::from_le_bytes(p[g + lane * 4..g + lane * 4 + 4].try_into().unwrap());
        let pdmin = f32::from_le_bytes(
            p[g + 32 + lane * 4..g + 32 + lane * 4 + 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(pd.to_bits(), file_d.to_bits(), "row {r} d");
        assert_eq!(pdmin.to_bits(), file_dmin.to_bits(), "row {r} dmin");
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &bytes[s + 4..s + 16]);
            assert_eq!(p[g + 64 + lane * 8 + j], sc, "row {r} sc[{j}]");
            assert_eq!(p[g + 128 + lane * 8 + j], m, "row {r} m[{j}]");
        }
        assert_eq!(
            &p[g + 192 + lane * 128..g + 192 + (lane + 1) * 128],
            &bytes[s + 16..s + 144]
        );
    }
}

#[test]
fn q4_k_pack_validates() {
    assert!(q4_k::pack_q4_k_rs8(&[0u8; 144], 1, 255).is_err());
    assert!(q4_k::pack_q4_k_rs8(&[0u8; 143], 1, 256).is_err());
    assert!(q4_k::pack_q4_k_rs8(&[], 0, 256).is_err());
}

/// One (seed, rows, k) case's max relative error (see `q8_0_case_max_rel`).
fn q4_k_case_max_rel(seed: u64, rows: usize, k: usize) -> f32 {
    let vals = pseudo(seed, rows * k);
    let x = pseudo(seed ^ 77, k);
    let wbytes = quant::pack(&DType::Q4_K, &vals).unwrap();
    let w = q4_k::pack_q4_k_rs8(&wbytes, rows, k).unwrap();
    let xq = act::quantize_row_q8k(KernelIsa::Scalar, &x).unwrap();
    let want = oracle(&DType::Q4_K, &wbytes, rows, k, &x);
    let mut y = vec![f32::NAN; rows];
    gemv_q4_k(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut y);
    let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));
    let mut max_rel = 0f32;
    for (g, w_) in y.iter().zip(&want) {
        max_rel = max_rel.max((g - w_).abs() / scale);
    }
    max_rel
}

/// Ignored diagnostic (see `observed_error_q8_0`): sweeps both the original
/// fixed shape (rows=16, k=512) AND the property tests' actual shape
/// distribution (rows 1..20, nsb 1..3, k = 256*nsb) — the fixed shape alone
/// under-observes because it never hits the single-super-block corner.
#[test]
#[ignore = "diagnostic; prints observed gemv error distribution"]
fn observed_error_q4_k() {
    let mut overall_max = 0f32;

    let mut fixed_max = 0f32;
    for seed in 0..500u64 {
        fixed_max = fixed_max.max(q4_k_case_max_rel(seed, 16, 512));
    }
    println!("q4_k [fixed rows=16 k=512, 500 seeds] max rel error: {fixed_max:e}");
    overall_max = overall_max.max(fixed_max);

    // Property-test shape distribution: rows 1..20, nsb 1..3 (k = 256*nsb).
    const SEEDS_PER_SHAPE: u64 = 60; // 19 * 2 * 60 = 2280 total cases
    let mut total_cases = 0u64;
    for rows in 1usize..20 {
        for nsb in 1usize..3 {
            let k = nsb * 256;
            let mut shape_max = 0f32;
            for seed in 0..SEEDS_PER_SHAPE {
                shape_max = shape_max.max(q4_k_case_max_rel(seed, rows, k));
                total_cases += 1;
            }
            println!("q4_k [rows={rows} nsb={nsb} k={k}] max rel error: {shape_max:e}");
            overall_max = overall_max.max(shape_max);
        }
    }
    println!(
        "q4_k observed OVERALL max rel error: {overall_max:e} over {total_cases} swept cases (+500 fixed-shape) (tol {:e})",
        gemv_rel_tol(&DType::Q4_K)
    );
}

// ---------- Attention ----------

/// Reference: the interpreter attention over a single query row at `pos`,
/// with the KV cache pre-populated for positions 0..pos and this token's
/// k/v appended. Returns the [n_heads*head_dim] output row.
#[allow(clippy::too_many_arguments)]
fn attn_oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kcache: &mut [f32],
    vcache: &mut [f32],
    pos: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    // Append this token's k/v at position `pos`.
    kcache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k);
    vcache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(v);
    let qt = inferno_graph::Tensor {
        shape: vec![1, n_heads * head_dim],
        data: q.to_vec(),
    };
    inferno_graph::ops::attention(
        &qt,
        kcache,
        vcache,
        pos + 1,
        n_heads,
        n_kv_heads,
        head_dim,
        pos,
    )
    .data
}

/// Drive the scalar attention kernel for one token; returns [n_heads*head_dim].
/// Appends this token's k/v into `kv` at `pos` first (the caller's job — the
/// kernel is read-only), matching what codegen does before the call.
#[allow(clippy::too_many_arguments)]
fn attn_kernel_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv: &mut [f32],
    seq_len: usize,
    pos: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    // Append k/v at pos: K region [0..seq*kv_dim), V region after it.
    kv[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k);
    let vreg = seq_len * kv_dim;
    kv[vreg + pos * kv_dim..vreg + (pos + 1) * kv_dim].copy_from_slice(v);
    let mut out = vec![f32::NAN; n_heads * head_dim];
    let mut scores = vec![0f32; seq_len];
    // Single-layer cache: kv_base=0, v_off=seq_len*kv_dim.
    // SAFETY: buffers sized to the documented contract; pos < seq_len.
    unsafe {
        inferno_kernels::inferno_attention_f32_scalar(
            out.as_mut_ptr(),
            q.as_ptr(),
            kv.as_mut_ptr(),
            scores.as_mut_ptr(),
            0,
            seq_len * kv_dim,
            pos,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
        );
    }
    out
}

/// Drive the AVX2 attention kernel for one token; returns [n_heads*head_dim].
/// Appends this token's k/v into `kv` at `pos` first (the caller's job — the
/// kernel is read-only), matching what codegen does before the call.
#[allow(clippy::too_many_arguments)]
fn attn_kernel_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv: &mut [f32],
    seq_len: usize,
    pos: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    // Append k/v at pos (caller's job; kernel is read-only) — same as scalar.
    kv[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k);
    let vreg = seq_len * kv_dim;
    kv[vreg + pos * kv_dim..vreg + (pos + 1) * kv_dim].copy_from_slice(v);
    let mut out = vec![f32::NAN; n_heads * head_dim];
    let mut scores = vec![0f32; seq_len];
    // SAFETY: same contract as the scalar driver; avx2 checked by caller.
    unsafe {
        inferno_kernels::inferno_attention_f32_avx2(
            out.as_mut_ptr(),
            q.as_ptr(),
            kv.as_mut_ptr(),
            scores.as_mut_ptr(),
            0,
            seq_len * kv_dim,
            pos,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
        );
    }
    out
}

/// Throwaway diagnostic (run manually): print the max relative error of the
/// attention kernel vs the std-exp interpreter across the shape distribution,
/// so `attn_rel_tol` is armed from data, not guessed. Run:
///   cargo test -p inferno-kernels --test rig observed_error_attention -- --ignored --nocapture
#[test]
#[ignore]
fn observed_error_attention() {
    let mut worst = 0f32;
    for seed in 0..20_000u64 {
        for &hd in &[8usize, 16, 64] {
            let (n_heads, n_kv_heads, head_dim) = (4usize, 2usize, hd);
            let kv_dim = n_kv_heads * head_dim;
            let seq_len = 16usize;
            let pos = (seed as usize) % 12;
            let mut kv = pseudo(seed, 2 * seq_len * kv_dim);
            let q = pseudo(seed ^ 1, n_heads * head_dim);
            let k = pseudo(seed ^ 2, kv_dim);
            let v = pseudo(seed ^ 3, kv_dim);
            let mut kc = kv[..seq_len * kv_dim].to_vec();
            let mut vc = kv[seq_len * kv_dim..].to_vec();
            let want = attn_oracle(
                &q, &k, &v, &mut kc, &mut vc, pos, kv_dim, n_heads, n_kv_heads, head_dim,
            );
            let got = attn_kernel_scalar(
                &q, &k, &v, &mut kv, seq_len, pos, kv_dim, n_heads, n_kv_heads, head_dim,
            );
            let scale = want.iter().fold(1f32, |m, x| m.max(x.abs())).max(1.0);
            for (g, w) in got.iter().zip(&want) {
                worst = worst.max((g - w).abs() / scale);
            }
        }
    }
    println!("observed_error_attention: max rel {worst:e}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn attention_scalar_matches_interpreter(
        seed in any::<u64>(), pos in 0usize..12, hd in prop::sample::select(vec![8usize, 16, 64]),
    ) {
        let (n_heads, n_kv_heads, head_dim) = (4usize, 2usize, hd);
        let kv_dim = n_kv_heads * head_dim;
        let seq_len = 16usize;
        // Random pre-populated cache for positions < pos, plus this token's k/v.
        let mut kv = pseudo(seed, 2 * seq_len * kv_dim); // scalar-kernel KV (K then V regions)
        let q = pseudo(seed ^ 1, n_heads * head_dim);
        let k = pseudo(seed ^ 2, kv_dim);
        let v = pseudo(seed ^ 3, kv_dim);
        // Oracle uses separate k/v caches; seed them from the same bytes as
        // the kernel's single kv buffer (K region = kv[..seq*kv_dim], V region after).
        let mut kc = kv[..seq_len * kv_dim].to_vec();
        let mut vc = kv[seq_len * kv_dim..].to_vec();
        let want = attn_oracle(&q, &k, &v, &mut kc, &mut vc, pos, kv_dim, n_heads, n_kv_heads, head_dim);
        let got = attn_kernel_scalar(&q, &k, &v, &mut kv, seq_len, pos, kv_dim, n_heads, n_kv_heads, head_dim);
        // Poly exp vs std exp: bounded, not bitwise. Tolerance derived from the
        // observed_error_attention sweep (see tolerance.rs::attn_rel_tol).
        let scale = want.iter().fold(1f32, |m, x| m.max(x.abs()));
        let tol = attn_rel_tol() * scale.max(1.0);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            prop_assert!((g - w).abs() <= tol, "elem {i}: got {g} want {w} (tol {tol})");
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn attention_isa_variants_bitwise_equal(
        seed in any::<u64>(), pos in 0usize..12, hd in prop::sample::select(vec![8usize, 16, 64]),
    ) {
        if !std::is_x86_feature_detected!("avx2") { return Ok(()); }
        let (n_heads, n_kv_heads, head_dim) = (4usize, 2usize, hd);
        let kv_dim = n_kv_heads * head_dim;
        let seq_len = 16usize;
        let base_kv = pseudo(seed, 2 * seq_len * kv_dim);
        let q = pseudo(seed ^ 1, n_heads * head_dim);
        let k = pseudo(seed ^ 2, kv_dim);
        let v = pseudo(seed ^ 3, kv_dim);
        let mut kv_s = base_kv.clone();
        let mut kv_a = base_kv;
        let a = attn_kernel_scalar(&q, &k, &v, &mut kv_s, seq_len, pos, kv_dim, n_heads, n_kv_heads, head_dim);
        let b = attn_kernel_avx2(&q, &k, &v, &mut kv_a, seq_len, pos, kv_dim, n_heads, n_kv_heads, head_dim);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            prop_assert_eq!(x.to_bits(), y.to_bits(), "elem {}: scalar {} avx2 {}", i, x, y);
        }
        // KV appends must also be bit-identical.
        prop_assert!(kv_s.iter().zip(&kv_a).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}

mod attention_hspan {
    //! M4b.11: the head-span kernels must be bitwise-identical to the
    //! whole-call kernels — per head, under any tiling of the head range —
    //! and scalar↔AVX2 bit-identity must extend to hspan.
    #[cfg(target_arch = "x86_64")]
    use inferno_kernels::{inferno_attention_f32_avx2, inferno_attention_f32_avx2_hspan};
    use inferno_kernels::{inferno_attention_f32_scalar, inferno_attention_f32_scalar_hspan};

    struct Case {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        pos: usize,
    }

    // Bench-model GQA shape (14/2), MHA, and a small odd-group shape.
    const CASES: &[Case] = &[
        Case {
            n_heads: 14,
            n_kv_heads: 2,
            head_dim: 8,
            pos: 0,
        },
        Case {
            n_heads: 14,
            n_kv_heads: 2,
            head_dim: 8,
            pos: 37,
        },
        Case {
            n_heads: 8,
            n_kv_heads: 8,
            head_dim: 16,
            pos: 100,
        },
        Case {
            n_heads: 6,
            n_kv_heads: 3,
            head_dim: 8,
            pos: 5,
        },
    ];

    fn lcg_fill(mut seed: u64, buf: &mut [f32]) {
        for v in buf.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((seed >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
        }
    }

    /// Head-range tilings to check: whole, per-head, and a ragged 3-way
    /// split that crosses GQA group boundaries for every CASES shape.
    fn tilings(n: usize) -> Vec<Vec<(usize, usize)>> {
        vec![
            vec![(0, n)],
            (0..n).map(|h| (h, h + 1)).collect(),
            vec![(0, 1), (1, 5.min(n - 1)), (5.min(n - 1), n)],
        ]
    }

    fn buffers(c: &Case) -> (Vec<f32>, Vec<f32>, usize, usize) {
        let kv_dim = c.n_kv_heads * c.head_dim;
        let seq = c.pos + 1;
        let v_off = seq * kv_dim;
        let mut q = vec![0f32; c.n_heads * c.head_dim];
        let mut kv = vec![0f32; 2 * v_off];
        lcg_fill(0x5eed_0001, &mut q);
        lcg_fill(0x5eed_0002, &mut kv);
        (q, kv, kv_dim, v_off)
    }

    fn whole_scalar(c: &Case) -> Vec<f32> {
        let (q, mut kv, kv_dim, v_off) = buffers(c);
        let mut out = vec![0f32; c.n_heads * c.head_dim];
        let mut scores = vec![0f32; c.pos + 1];
        // SAFETY: buffers sized per the AttnFn contract above.
        unsafe {
            inferno_attention_f32_scalar(
                out.as_mut_ptr(),
                q.as_ptr(),
                kv.as_mut_ptr(),
                scores.as_mut_ptr(),
                0,
                v_off,
                c.pos,
                kv_dim,
                c.n_heads,
                c.n_kv_heads,
                c.head_dim,
            );
        }
        out
    }

    fn hspan_scalar(c: &Case, spans: &[(usize, usize)]) -> Vec<f32> {
        let (q, mut kv, kv_dim, v_off) = buffers(c);
        let mut out = vec![0f32; c.n_heads * c.head_dim];
        for &(h0, h1) in spans {
            // Fresh scratch per span, mimicking lane-local scratch.
            let mut scores = vec![0f32; c.pos + 1];
            // SAFETY: buffers sized per the hspan contract; spans tile 0..n_heads.
            unsafe {
                inferno_attention_f32_scalar_hspan(
                    out.as_mut_ptr(),
                    q.as_ptr(),
                    kv.as_mut_ptr(),
                    scores.as_mut_ptr(),
                    0,
                    v_off,
                    c.pos,
                    kv_dim,
                    c.n_heads,
                    c.n_kv_heads,
                    c.head_dim,
                    h0,
                    h1,
                );
            }
        }
        out
    }

    fn assert_bits_eq(a: &[f32], b: &[f32], ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: element {i}: {x} vs {y}");
        }
    }

    #[test]
    fn hspan_scalar_bitwise_matches_whole_call_under_any_tiling() {
        for (ci, c) in CASES.iter().enumerate() {
            let whole = whole_scalar(c);
            for (ti, spans) in tilings(c.n_heads).iter().enumerate() {
                let tiled = hspan_scalar(c, spans);
                assert_bits_eq(&whole, &tiled, &format!("case {ci} tiling {ti}"));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn whole_avx2(c: &Case) -> Vec<f32> {
        let (q, mut kv, kv_dim, v_off) = buffers(c);
        let mut out = vec![0f32; c.n_heads * c.head_dim];
        let mut scores = vec![0f32; c.pos + 1];
        // SAFETY: contract as scalar, plus the avx2 guard in the test below.
        unsafe {
            inferno_attention_f32_avx2(
                out.as_mut_ptr(),
                q.as_ptr(),
                kv.as_mut_ptr(),
                scores.as_mut_ptr(),
                0,
                v_off,
                c.pos,
                kv_dim,
                c.n_heads,
                c.n_kv_heads,
                c.head_dim,
            );
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    fn hspan_avx2(c: &Case, spans: &[(usize, usize)]) -> Vec<f32> {
        let (q, mut kv, kv_dim, v_off) = buffers(c);
        let mut out = vec![0f32; c.n_heads * c.head_dim];
        for &(h0, h1) in spans {
            let mut scores = vec![0f32; c.pos + 1];
            // SAFETY: contract as scalar hspan, plus the avx2 guard below.
            unsafe {
                inferno_attention_f32_avx2_hspan(
                    out.as_mut_ptr(),
                    q.as_ptr(),
                    kv.as_mut_ptr(),
                    scores.as_mut_ptr(),
                    0,
                    v_off,
                    c.pos,
                    kv_dim,
                    c.n_heads,
                    c.n_kv_heads,
                    c.head_dim,
                    h0,
                    h1,
                );
            }
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn hspan_avx2_bitwise_matches_whole_call_and_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: no avx2");
            return;
        }
        for (ci, c) in CASES.iter().enumerate() {
            let whole = whole_avx2(c);
            // scalar↔AVX2 bit-identity (M4b.3) must extend to hspan.
            assert_bits_eq(&whole, &whole_scalar(c), &format!("case {ci} isa"));
            for (ti, spans) in tilings(c.n_heads).iter().enumerate() {
                let tiled = hspan_avx2(c, spans);
                assert_bits_eq(&whole, &tiled, &format!("case {ci} avx2 tiling {ti}"));
            }
        }
    }
}
