//! Thread-count bit-identity: the same GEMV dispatched at t=1 / t=4 / t=12
//! must produce EXACTLY the same bits as one direct single-threaded kernel
//! call, per dtype. Row-partitioned shards never reassociate any f32 op, so
//! this extends the kernels' "ISA variants are bit-identical" contract to
//! thread count. Exact equality (`to_bits`), never tolerance.

use inferno_formats::{DType, quant};
use inferno_kernels::{AlignedBuf, reference_kernels};
use inferno_pool::{GemvFn, Pool};

/// Deterministic pseudo-random f32s in [-1, 1) (same generator as the
/// kernels' rig — xorshift, no deps).
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

/// Pack weights + quantize the activation for `dtype` via the scalar
/// KernelSet, returning (packed weights, activation bytes).
fn prep(dtype: &DType, rows: usize, k: usize, seed: u64) -> (AlignedBuf, Vec<u8>) {
    let set = reference_kernels(dtype).expect("scalar set always available");
    let wvals = pseudo(seed, rows * k);
    let wbytes = quant::pack(dtype, &wvals).unwrap();
    let w = set.pack(&wbytes, rows, k).unwrap();
    let x = pseudo(seed ^ 0x9e3779b97f4a7c15, k);
    let xq = set.quantize_row(&x).unwrap();
    (w, xq)
}

fn serial(kernel: GemvFn, w: &AlignedBuf, xq: &[u8], rows: usize, k: usize) -> Vec<f32> {
    let mut y = vec![f32::NAN; rows];
    // SAFETY: w/xq built by prep() for exactly (rows, k); y has rows f32s.
    unsafe { kernel(y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, 0, rows) };
    y
}

fn pooled(
    pool: &Pool,
    kernel: GemvFn,
    w: &AlignedBuf,
    xq: &[u8],
    rows: usize,
    k: usize,
) -> Vec<f32> {
    let mut y = vec![f32::NAN; rows];
    // SAFETY: same contract as `serial`; the pool only splits the range.
    unsafe { pool.par_gemv(kernel, y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, rows) };
    y
}

/// Thread counts exercised for bit-identity: 1/4/12 (baseline spread), 64
/// (oversubscription — far more lanes than any plausible strip count), and
/// the strip-count edges for rows=1003 (1003 rows = 126 8-row strips: 125
/// undersplits by one strip, 126 is exact, 127 exceeds the strip count so
/// `shard_table` collapses back to 126 shards).
const THREAD_COUNTS: [usize; 7] = [1, 4, 12, 64, 125, 126, 127];

fn assert_bit_identical(dtype: &DType, kernel: GemvFn, rows: usize, k: usize) {
    let (w, xq) = prep(dtype, rows, k, 0xfeed_beef);
    let want = serial(kernel, &w, &xq, rows, k);
    for threads in THREAD_COUNTS {
        let pool = Pool::new(threads);
        let got = pooled(&pool, kernel, &w, &xq, rows, k);
        for (i, (g, s)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                s.to_bits(),
                "{dtype:?} t={threads} row {i}: {g} != {s}"
            );
        }
    }
}

#[test]
fn shard_align_matches_kernel_strip() {
    assert_eq!(inferno_pool::SHARD_ALIGN, inferno_kernels::STRIP);
}

#[test]
fn f32_thread_count_is_bit_invisible() {
    // rows deliberately not a multiple of 8; k unconstrained for f32.
    assert_bit_identical(
        &DType::F32,
        inferno_kernels::inferno_gemv_f32_rs8_scalar,
        1003,
        33,
    );
}

#[test]
fn q8_0_thread_count_is_bit_invisible() {
    // k must be a multiple of 32 (Q8_0 block).
    assert_bit_identical(
        &DType::Q8_0,
        inferno_kernels::inferno_gemv_q8_0_rs8_scalar,
        1003,
        64,
    );
}

#[test]
fn q4_k_thread_count_is_bit_invisible() {
    // k must be a multiple of 256 (Q4_K superblock).
    assert_bit_identical(
        &DType::Q4_K,
        inferno_kernels::inferno_gemv_q4_k_rs8_scalar,
        1003,
        256,
    );
}

/// M4b.5: decode_cap must be bit-invisible. Fix a 12-lane pool, sweep the
/// decode cap 1..=12, and require every capped dispatch to match one direct
/// serial kernel call exactly — capping only regroups rows into shards.
#[test]
fn q8_0_decode_cap_is_bit_invisible() {
    let dtype = DType::Q8_0;
    let kernel = inferno_kernels::inferno_gemv_q8_0_rs8_scalar;
    let (rows, k) = (1003usize, 64usize);
    let (w, xq) = prep(&dtype, rows, k, 0xfeed_beef);
    let want = serial(kernel, &w, &xq, rows, k);
    let pool = Pool::new(12);
    for cap in 1..=12usize {
        pool.set_decode_threads(cap);
        let got = pooled(&pool, kernel, &w, &xq, rows, k);
        for (i, (g, s)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), s.to_bits(), "cap={cap} row {i}: {g} != {s}");
        }
    }
}

#[test]
fn par_gemm_bit_identical_across_threads() {
    use inferno_kernels::{KernelIsa, act, q8_0};
    let (rows, k, m) = (129usize, 64usize, 5usize);
    let vals: Vec<f32> = (0..rows * k).map(|i| ((i as f32) * 0.001).sin()).collect();
    let w = q8_0::pack_q8_0_rs8(
        &inferno_formats::quant::pack(&inferno_formats::DType::Q8_0, &vals).unwrap(),
        rows,
        k,
    )
    .unwrap();
    let mut panel = Vec::new();
    for t in 0..m {
        let x: Vec<f32> = (0..k).map(|i| ((i + t) as f32 * 0.01).cos()).collect();
        panel.extend_from_slice(&act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap());
    }
    let kernel: inferno_pool::GemmFn = q8_0::inferno_gemm_q8_0_rs8_avx2;
    // Serial reference (1 lane).
    let mut want = vec![f32::NAN; m * rows];
    // SAFETY: sized buffers; full range.
    unsafe {
        kernel(
            want.as_mut_ptr(),
            panel.as_ptr(),
            w.as_ptr(),
            k,
            m,
            rows,
            0,
            rows,
        )
    };
    for threads in [1usize, 2, 3, 8] {
        let pool = inferno_pool::Pool::new(threads);
        let mut got = vec![f32::NAN; m * rows];
        // SAFETY: buffers live for the call; no overlapping dispatch.
        unsafe {
            pool.par_gemm(
                kernel,
                got.as_mut_ptr(),
                panel.as_ptr(),
                w.as_ptr(),
                k,
                m,
                rows,
            )
        };
        for i in 0..m * rows {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "threads {threads} i {i}"
            );
        }
    }
}
