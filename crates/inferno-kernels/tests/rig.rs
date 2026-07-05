//! Kernel-vs-oracle rig (spec §Testing): every kernel is compared against
//! `inferno_formats::quant::dequant` + the scalar reference matmul, ISA
//! variants are compared bitwise, and row-range partitioning is bit-stable.

use inferno_formats::{DType, quant};
use inferno_graph::Tensor;
use inferno_graph::tolerance::gemv_rel_tol;
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
