//! GEMV throughput on real Llama-family shapes (Qwen2.5-0.5B: 896/4864/151936;
//! Llama-3-8B: 4096/14336/128256), side by side with the devenv-pinned
//! llama.cpp CPU kernels when built with --features ggml-compare.
//! Run via `mise run bench-kernels` inside the devenv shell on quiet hardware;
//! numbers from shared CI runners are noise (spec §Benchmarks).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use inferno_formats::DType;
use inferno_formats::quant::f32_to_f16;
use inferno_kernels::registry::{KernelSet, kernels_for, reference_kernels};
use inferno_target::Isa;

const SHAPES_F32: &[(usize, usize)] = &[(4096, 4096)];
const SHAPES_Q8_0: &[(usize, usize)] = &[
    (896, 896),
    (4864, 896),
    (896, 4864),
    (151936, 896),
    (4096, 4096),
    (14336, 4096),
];
const SHAPES_Q4_K: &[(usize, usize)] =
    &[(4096, 4096), (14336, 4096), (4096, 14336), (128256, 4096)];

fn pseudo_f32(mut seed: u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn pseudo_bytes(mut seed: u64, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u8
        })
        .collect()
}

/// Plausible file-order weight bytes without quantizing gigabytes of f32:
/// random quant payloads with small fixed scales (perf is value-independent).
fn gen_weights(dtype: &DType, rows: usize, k: usize) -> Vec<u8> {
    match dtype {
        DType::F32 => pseudo_f32(1, rows * k)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DType::Q8_0 => {
            let nb = rows * k / 32;
            let mut out = Vec::with_capacity(nb * 34);
            let d = f32_to_f16(0.05).to_le_bytes();
            let qs = pseudo_bytes(2, nb * 32);
            for b in 0..nb {
                out.extend_from_slice(&d);
                out.extend_from_slice(&qs[b * 32..(b + 1) * 32]);
            }
            out
        }
        DType::Q4_K => {
            let nsb = rows * k / 256;
            let mut out = Vec::with_capacity(nsb * 144);
            let d = f32_to_f16(0.05).to_le_bytes();
            let dmin = f32_to_f16(0.02).to_le_bytes();
            let payload = pseudo_bytes(3, nsb * 140);
            for b in 0..nsb {
                out.extend_from_slice(&d);
                out.extend_from_slice(&dmin);
                out.extend_from_slice(&payload[b * 140..(b + 1) * 140]);
            }
            out
        }
        _ => unreachable!("no benches for {dtype:?}"),
    }
}

fn sets_for(dtype: &DType) -> Vec<(&'static str, KernelSet)> {
    let mut v = vec![("inferno-scalar", reference_kernels(dtype).unwrap())];
    if let Some(s) = kernels_for(dtype, Isa::X86_64v3) {
        v.push(("inferno-avx2", s));
    }
    v
}

fn bench_dtype(c: &mut Criterion, dtype: DType, shapes: &[(usize, usize)]) {
    let mut group = c.benchmark_group(format!("gemv/{dtype:?}"));
    group.sample_size(20);
    for &(rows, k) in shapes {
        let file = gen_weights(&dtype, rows, k);
        let x = pseudo_f32(42, k);
        for (name, set) in sets_for(&dtype) {
            let w = set.pack(&file, rows, k).unwrap();
            let xq = set.quantize_row(&x).unwrap();
            let mut y = vec![0f32; rows];
            group.throughput(Throughput::Bytes(w.len() as u64));
            group.bench_function(BenchmarkId::new(name, format!("{rows}x{k}")), |b| {
                b.iter(|| set.gemv(&mut y, &xq, &w, rows, k, 0, rows).unwrap())
            });
        }
        // Pure weight-streaming ceiling: the max read bandwidth this machine
        // sustains over the packed weight image, with no dot-product compute.
        // GEMV GB/s far below this ⇒ memory-latency/MLP-bound (prefetch helps);
        // GEMV GB/s near this ⇒ compute-bound (pivot to op-reduction). Uses the
        // AVX2 pack so the byte count matches the inferno-avx2 GEMV arm.
        if let Some(set) = kernels_for(&dtype, Isa::X86_64v3) {
            let w = set.pack(&file, rows, k).unwrap();
            group.throughput(Throughput::Bytes(w.len() as u64));
            group.bench_function(
                BenchmarkId::new("stream-read", format!("{rows}x{k}")),
                |b| {
                    b.iter(|| {
                        let mut acc = 0u64;
                        // 8-wide u64 reduction: enough ILP to expose read bandwidth,
                        // no cross-lane dependency that would serialize on latency.
                        for chunk in w.as_slice().chunks_exact(64) {
                            for w8 in chunk.chunks_exact(8) {
                                acc = acc.wrapping_add(u64::from_le_bytes(w8.try_into().unwrap()));
                            }
                        }
                        std::hint::black_box(acc)
                    })
                },
            );
        }
        // M4b.6 Task 1: reduce/combine ceiling arms (cost models, wrong
        // numbers by design — see the reduce_ceiling module docs). Q8_0 only;
        // every SHAPES_Q8_0 rows value is a multiple of STRIP, asserted here
        // so a future shape can't silently hit the arms' whole-strip limit.
        #[cfg(target_arch = "x86_64")]
        if matches!(dtype, DType::Q8_0) && std::arch::is_x86_feature_detected!("avx2") {
            assert_eq!(
                rows % inferno_kernels::STRIP,
                0,
                "ceiling arms need whole strips"
            );
            let set = kernels_for(&dtype, Isa::X86_64v3).unwrap();
            let w = set.pack(&file, rows, k).unwrap();
            let xq = set.quantize_row(&x).unwrap();
            let mut y = vec![0f32; rows];
            group.throughput(Throughput::Bytes(w.len() as u64));
            group.bench_function(
                BenchmarkId::new("reduce-ceiling", format!("{rows}x{k}")),
                |b| {
                    b.iter(|| {
                        // SAFETY: y/xq/w built above for exactly this rows/k;
                        // rows % STRIP == 0 asserted; AVX2 runtime-detected.
                        unsafe {
                            reduce_ceiling::gemv_no_reduce_no_combine(
                                y.as_mut_ptr(),
                                xq.as_slice().as_ptr(),
                                w.as_slice().as_ptr(),
                                k,
                                rows,
                            );
                        }
                        std::hint::black_box(y[0]);
                    })
                },
            );
            group.bench_function(
                BenchmarkId::new("combine-stub", format!("{rows}x{k}")),
                |b| {
                    b.iter(|| {
                        // SAFETY: as the reduce-ceiling arm above.
                        unsafe {
                            reduce_ceiling::gemv_no_combine(
                                y.as_mut_ptr(),
                                xq.as_slice().as_ptr(),
                                w.as_slice().as_ptr(),
                                k,
                                rows,
                            );
                        }
                        std::hint::black_box(y[0]);
                    })
                },
            );
            // M4b.6 restructure candidate 1: the same kernel with the
            // unpack/add reduce tree. Correct numbers — prove bitwise equality
            // against the library kernel once per process, then time it.
            {
                let mut y_ref = vec![f32::NAN; rows];
                set.gemv(&mut y_ref, &xq, &w, rows, k, 0, rows).unwrap();
                // SAFETY: buffers built above for exactly this rows/k;
                // rows % STRIP == 0 asserted; AVX2 runtime-detected.
                unsafe {
                    reduce_unpack::gemv(
                        y.as_mut_ptr(),
                        xq.as_slice().as_ptr(),
                        w.as_slice().as_ptr(),
                        k,
                        rows,
                    );
                }
                for (i, (a, b)) in y_ref.iter().zip(&y).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "reduce-unpack arm diverges from library kernel at row {i} ({rows}x{k})"
                    );
                }
            }
            group.bench_function(
                BenchmarkId::new("reduce-unpack", format!("{rows}x{k}")),
                |b| {
                    b.iter(|| {
                        // SAFETY: as the bitwise pre-check above.
                        unsafe {
                            reduce_unpack::gemv(
                                y.as_mut_ptr(),
                                xq.as_slice().as_ptr(),
                                w.as_slice().as_ptr(),
                                k,
                                rows,
                            );
                        }
                        std::hint::black_box(y[0]);
                    })
                },
            );
        }
        #[cfg(feature = "ggml-compare")]
        ggml::bench(&mut group, &dtype, &file, &x, rows, k);
    }
    group.finish();
}

#[cfg(feature = "ggml-compare")]
mod ggml {
    //! dlopen the pinned ggml CPU backend and drive its row-dot kernels on
    //! identical data. ggml consumes file-order weights directly (no repack),
    //! so its throughput basis is the file byte count.
    use std::ffi::c_void;
    use std::sync::OnceLock;

    use criterion::{BenchmarkId, Throughput, measurement::WallTime};
    use inferno_formats::DType;

    // void ggml_vec_dot_*(int n, float *s, size_t bs, const void *x, size_t bx,
    //                     const void *y, size_t by, int nrc)
    type VecDot =
        unsafe extern "C" fn(i32, *mut f32, usize, *const c_void, usize, *const c_void, usize, i32);
    // void quantize_row_*(const float *x, void *y, int64_t k)
    type QuantRow = unsafe extern "C" fn(*const f32, *mut c_void, i64);

    fn lib() -> &'static libloading::Library {
        static LIB: OnceLock<libloading::Library> = OnceLock::new();
        LIB.get_or_init(|| {
            let path = std::env::var("INFERNO_GGML_CPU_LIB")
                .expect("INFERNO_GGML_CPU_LIB not set — run inside `devenv shell`");
            // SAFETY: loading the version-pinned ggml CPU backend for benching.
            unsafe { libloading::Library::new(&path) }
                .unwrap_or_else(|e| panic!("cannot load {path}: {e}"))
        })
    }

    pub fn bench(
        group: &mut criterion::BenchmarkGroup<'_, WallTime>,
        dtype: &DType,
        file: &[u8],
        x: &[f32],
        rows: usize,
        k: usize,
    ) {
        let (dot_sym, quant_sym, act_bytes, row_bytes): (&[u8], Option<&[u8]>, usize, usize) =
            match dtype {
                DType::F32 => (b"ggml_vec_dot_f32", None, k * 4, k * 4),
                DType::Q8_0 => (
                    b"ggml_vec_dot_q8_0_q8_0",
                    Some(b"quantize_row_q8_0"),
                    k / 32 * 34,
                    k / 32 * 34,
                ),
                DType::Q4_K => (
                    b"ggml_vec_dot_q4_K_q8_K",
                    Some(b"quantize_row_q8_K"),
                    k / 256 * 292,
                    k / 256 * 144,
                ),
                _ => return,
            };
        // SAFETY: signatures match the pinned ggml's headers.
        let dot: libloading::Symbol<'_, VecDot> = unsafe { lib().get(dot_sym).unwrap() };
        let xq: Vec<u8> = match quant_sym {
            Some(qs) => {
                let quant: libloading::Symbol<'_, QuantRow> = unsafe { lib().get(qs).unwrap() };
                let mut buf = vec![0u8; act_bytes];
                // SAFETY: x has k f32; buf sized per ggml's activation block.
                unsafe { quant(x.as_ptr(), buf.as_mut_ptr().cast(), k as i64) };
                buf
            }
            None => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        let mut y = vec![0f32; rows];
        group.throughput(Throughput::Bytes((rows * row_bytes) as u64));
        group.bench_function(BenchmarkId::new("ggml", format!("{rows}x{k}")), |b| {
            b.iter(|| {
                for r in 0..rows {
                    // SAFETY: file holds rows*row_bytes; xq is ggml's own
                    // activation layout for k; one row per call (nrc=1).
                    unsafe {
                        dot(
                            k as i32,
                            y.as_mut_ptr().add(r),
                            0,
                            file.as_ptr().add(r * row_bytes).cast(),
                            0,
                            xq.as_ptr().cast(),
                            0,
                            1,
                        )
                    };
                }
            })
        });
    }

    /// ggml's naive batched path for comparison against `KernelSet::gemm`:
    /// ggml's plain CPU kernels expose no fused M-panel GEMM entry point over
    /// this dlopen ABI (that lives in the templated llamafile/tinyBLAS path),
    /// so the honest baseline is the same per-row `vec_dot` called once per
    /// token (`nrc=1`) with no weight-row reuse across tokens — i.e. exactly
    /// `m` independent GEMV passes. Same MACs throughput basis as the inferno
    /// `gemm` benchmark for a direct comparison.
    pub fn bench_gemm(
        group: &mut criterion::BenchmarkGroup<'_, WallTime>,
        dtype: &DType,
        file: &[u8],
        rows: usize,
        k: usize,
        m: usize,
    ) {
        let (dot_sym, quant_sym, act_bytes, row_bytes): (&[u8], Option<&[u8]>, usize, usize) =
            match dtype {
                DType::F32 => (b"ggml_vec_dot_f32", None, k * 4, k * 4),
                DType::Q8_0 => (
                    b"ggml_vec_dot_q8_0_q8_0",
                    Some(b"quantize_row_q8_0"),
                    k / 32 * 34,
                    k / 32 * 34,
                ),
                DType::Q4_K => (
                    b"ggml_vec_dot_q4_K_q8_K",
                    Some(b"quantize_row_q8_K"),
                    k / 256 * 292,
                    k / 256 * 144,
                ),
                _ => return,
            };
        // SAFETY: signatures match the pinned ggml's headers.
        let dot: libloading::Symbol<'_, VecDot> = unsafe { lib().get(dot_sym).unwrap() };
        // m-row activation panel, quantized through ggml's own quantize_row
        // (or raw f32 bytes for the unquantized path), one block per token.
        let mut panel = vec![0u8; act_bytes * m];
        for t in 0..m {
            let x = super::pseudo_f32(42 + t as u64, k);
            match quant_sym {
                Some(qs) => {
                    let quant: libloading::Symbol<'_, QuantRow> = unsafe { lib().get(qs).unwrap() };
                    // SAFETY: x has k f32; slot sized per ggml's activation block.
                    unsafe {
                        quant(
                            x.as_ptr(),
                            panel[t * act_bytes..(t + 1) * act_bytes]
                                .as_mut_ptr()
                                .cast(),
                            k as i64,
                        )
                    };
                }
                None => {
                    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
                    panel[t * act_bytes..(t + 1) * act_bytes].copy_from_slice(&bytes);
                }
            }
        }
        let mut y = vec![0f32; m * rows];
        group.throughput(Throughput::Elements((m * rows * k) as u64));
        group.bench_function(BenchmarkId::new("ggml", format!("{rows}x{k}/m{m}")), |b| {
            b.iter(|| {
                for t in 0..m {
                    for r in 0..rows {
                        // SAFETY: file holds rows*row_bytes; panel holds m
                        // act_bytes-sized activation rows; one (row, token)
                        // pair per call (nrc=1) — ggml's un-batched baseline.
                        unsafe {
                            dot(
                                k as i32,
                                y.as_mut_ptr().add(t * rows + r),
                                0,
                                file.as_ptr().add(r * row_bytes).cast(),
                                0,
                                panel.as_ptr().add(t * act_bytes).cast(),
                                0,
                                1,
                            )
                        };
                    }
                }
            })
        });
    }
}

/// M4b.6 Task 1 diagnostic arms (spec 2026-07-09-m4b6): copies of the AVX2
/// full-strip GEMV body with the per-block reduce/combine progressively
/// stubbed. Wrong results by design — these are cost models, not kernels,
/// and must never ship in the library. Layout consts mirror `q8_0.rs`.
///
/// Conservative by construction: keeping each lane's dot product live costs
/// one `vpaddd` per lane per block (arm A) or one per block (arm B), so the
/// measured baseline-vs-arm delta *understates* the true reduce/combine cost
/// slightly — the gate can under-claim headroom, never over-claim it.
#[cfg(target_arch = "x86_64")]
mod reduce_ceiling {
    use std::arch::x86_64::__m256i;

    use inferno_kernels::STRIP;

    const WBLOCK: usize = 32; // weight elements per block (q8_0.rs:10)
    const GROUP_BYTES: usize = 288; // 8 f32 d + 8×32 qs (q8_0.rs:12)
    const Q8A_BLOCK_BYTES: usize = 36; // f32 d + 32 i8 (act.rs:15)
    const PF_DIST: usize = 4; // mirrors q8_0.rs:19 so arms model the shipped kernel

    /// Copied verbatim from `q8_0.rs::hsum8_i32` (pub(crate) there, so the
    /// bench keeps its own copy): transpose-reduce 8 lane-parallel i32
    /// accumulators into one vector whose lane `i` holds v[i]'s horizontal sum.
    #[target_feature(enable = "avx2")]
    fn hsum8_i32(v: [__m256i; 8]) -> __m256i {
        use std::arch::x86_64::*;
        let s01 = _mm256_hadd_epi32(v[0], v[1]);
        let s23 = _mm256_hadd_epi32(v[2], v[3]);
        let s45 = _mm256_hadd_epi32(v[4], v[5]);
        let s67 = _mm256_hadd_epi32(v[6], v[7]);
        let s0123 = _mm256_hadd_epi32(s01, s23);
        let s4567 = _mm256_hadd_epi32(s45, s67);
        let lo = _mm256_permute2x128_si256::<0x20>(s0123, s4567);
        let hi = _mm256_permute2x128_si256::<0x31>(s0123, s4567);
        _mm256_add_epi32(lo, hi)
    }

    /// Arm A — "reduce+combine → 0". Full weight/activation streaming, the
    /// prefetch hint, and the complete int8 dot (sign×2/maddubs/madd per lane)
    /// are intact; `hsum8_i32` and the f32 combine (dx load, cvt, dw load,
    /// mul, fmadd) are replaced by one `vpaddd` per lane into a sink that is
    /// stored once per strip. Lower bound on kernel time if the per-block
    /// reduce/combine were free.
    ///
    /// # Safety
    /// As `inferno_gemv_q8_0_rs8_avx2` (`y` writable for `rows` f32; `x` a
    /// q8a buffer for this `k`; `w` an rs8 pack for exactly this `k`/`rows`,
    /// 32-byte aligned; `k` a positive multiple of 32), plus: whole strips
    /// only (`rows % STRIP == 0`) and AVX2+FMA present.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemv_no_reduce_no_combine(
        y: *mut f32,
        x: *const u8,
        w: *const u8,
        k: usize,
        rows: usize,
    ) {
        use std::arch::x86_64::*;
        let nb = k / WBLOCK;
        let ones = _mm256_set1_epi16(1);
        for strip in 0..rows / STRIP {
            let mut sink = _mm256_setzero_si256();
            for b in 0..nb {
                let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                // Same pure-hint prefetch as the shipped kernel (q8_0.rs:160-163).
                let pf_addr = w
                    .wrapping_add((strip * nb + b + PF_DIST) * GROUP_BYTES)
                    .cast();
                _mm_prefetch::<_MM_HINT_T0>(pf_addr);
                let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
                let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                let qs = unsafe { g.add(32) };
                for lane in 0..STRIP {
                    let wv = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                    let aw = _mm256_sign_epi8(wv, wv);
                    let sx = _mm256_sign_epi8(xv, wv);
                    let p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                    // Cheapest possible liveness: without this add the whole
                    // lane's dot chain is dead and the optimizer deletes it.
                    sink = _mm256_add_epi32(sink, p);
                }
            }
            // One garbage store per strip keeps `sink` observable.
            unsafe { _mm256_storeu_ps(y.add(strip * STRIP), _mm256_castsi256_ps(sink)) };
        }
    }

    /// Arm B — "combine → 0, reduce kept". Identical to arm A except the
    /// per-block `hsum8_i32` transpose-reduce still runs; only the f32
    /// combine is stubbed (one `vpaddd` of the reduced vector into the sink).
    /// Attribution: (baseline − B) ≈ combine cost, (B − A) ≈ reduce cost.
    ///
    /// # Safety
    /// As [`gemv_no_reduce_no_combine`].
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemv_no_combine(y: *mut f32, x: *const u8, w: *const u8, k: usize, rows: usize) {
        use std::arch::x86_64::*;
        let nb = k / WBLOCK;
        let ones = _mm256_set1_epi16(1);
        for strip in 0..rows / STRIP {
            let mut sink = _mm256_setzero_si256();
            for b in 0..nb {
                let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                let pf_addr = w
                    .wrapping_add((strip * nb + b + PF_DIST) * GROUP_BYTES)
                    .cast();
                _mm_prefetch::<_MM_HINT_T0>(pf_addr);
                let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
                let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                let qs = unsafe { g.add(32) };
                let mut p = [_mm256_setzero_si256(); STRIP];
                for (lane, pl) in p.iter_mut().enumerate() {
                    let wv = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                    let aw = _mm256_sign_epi8(wv, wv);
                    let sx = _mm256_sign_epi8(xv, wv);
                    *pl = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                }
                // Reduce kept, combine stubbed: cvt/dw/dx/mul/fmadd → one add.
                sink = _mm256_add_epi32(sink, hsum8_i32(p));
            }
            unsafe { _mm256_storeu_ps(y.add(strip * STRIP), _mm256_castsi256_ps(sink)) };
        }
    }
}

/// M4b.6 restructure candidate 1 (spec §Task 2+): the shipped AVX2 full-strip
/// GEMV with exactly one delta — `hsum8_i32` swapped for an unpack/add
/// transpose-reduce. Unlike the `reduce_ceiling` cost models this arm computes
/// CORRECT numbers; the bench setup asserts bitwise equality against the
/// library kernel before timing it. Whole strips only (`rows % STRIP == 0`).
/// Deleted once the A/B is decided — the winning tree ships in `q8_0.rs`, and
/// a losing tree is recorded in the spec Amendments and removed.
#[cfg(target_arch = "x86_64")]
mod reduce_unpack {
    use std::arch::x86_64::__m256i;

    use inferno_kernels::STRIP;

    const WBLOCK: usize = 32; // weight elements per block (q8_0.rs:10)
    const GROUP_BYTES: usize = 288; // 8 f32 d + 8×32 qs (q8_0.rs:12)
    const Q8A_BLOCK_BYTES: usize = 36; // f32 d + 32 i8 (act.rs:15)
    const PF_DIST: usize = 4; // mirrors q8_0.rs:19 so the A/B isolates the tree

    /// Candidate-1 transpose-reduce: same contract as `q8_0.rs::hsum8_i32`
    /// (lane `i` of the result = horizontal sum of `v[i]`), built from
    /// `vpunpck`/`vpaddd` instead of `vphaddd`. Wrapping i32 adds are
    /// associative, so this is bit-identical to the hadd tree by construction.
    #[target_feature(enable = "avx2")]
    fn hsum8_i32_unpack(v: [__m256i; 8]) -> __m256i {
        use std::arch::x86_64::*;
        // Round 1 — 32-bit interleave + add. Per 128-bit half, lo/hi unpack of
        // (a, b) give [a0 b0 a1 b1] and [a2 b2 a3 b3]; their sum holds each
        // input's even/odd 2-element partials interleaved: [a02 b02 a13 b13].
        let s01 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(v[0], v[1]),
            _mm256_unpackhi_epi32(v[0], v[1]),
        );
        let s23 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(v[2], v[3]),
            _mm256_unpackhi_epi32(v[2], v[3]),
        );
        let s45 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(v[4], v[5]),
            _mm256_unpackhi_epi32(v[4], v[5]),
        );
        let s67 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(v[6], v[7]),
            _mm256_unpackhi_epi32(v[6], v[7]),
        );
        // Round 2 — 64-bit interleave + add: per half [x02 y02] + [x13 y13]
        // leaves [v0half v1half v2half v3half], i.e. the same
        // [v0lo v1lo v2lo v3lo | v0hi v1hi v2hi v3hi] layout the hadd tree's
        // second round produces.
        let s0123 = _mm256_add_epi32(
            _mm256_unpacklo_epi64(s01, s23),
            _mm256_unpackhi_epi64(s01, s23),
        );
        let s4567 = _mm256_add_epi32(
            _mm256_unpacklo_epi64(s45, s67),
            _mm256_unpackhi_epi64(s45, s67),
        );
        // Round 3 — unchanged cross-128 recombine: lane i = full sum of v[i].
        let lo = _mm256_permute2x128_si256::<0x20>(s0123, s4567);
        let hi = _mm256_permute2x128_si256::<0x31>(s0123, s4567);
        _mm256_add_epi32(lo, hi)
    }

    /// The shipped full-strip fast path (`q8_0.rs::inferno_gemv_q8_0_rs8_avx2`,
    /// whole-strip branch) verbatim — prefetch, sign-trick dot, and the f32
    /// combine included — with only the reduce tree swapped.
    ///
    /// # Safety
    /// As `inferno_gemv_q8_0_rs8_avx2` (`y` writable for `rows` f32; `x` a q8a
    /// buffer for this `k`; `w` an rs8 pack for exactly this `k`/`rows`,
    /// 32-byte aligned; `k` a positive multiple of 32), plus: whole strips
    /// only (`rows % STRIP == 0`) and AVX2+FMA present.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemv(y: *mut f32, x: *const u8, w: *const u8, k: usize, rows: usize) {
        use std::arch::x86_64::*;
        let nb = k / WBLOCK;
        let ones = _mm256_set1_epi16(1);
        for strip in 0..rows / STRIP {
            let mut acc = _mm256_setzero_ps();
            for b in 0..nb {
                let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                // Same pure-hint prefetch as the shipped kernel (q8_0.rs:160-163).
                let pf_addr = w
                    .wrapping_add((strip * nb + b + PF_DIST) * GROUP_BYTES)
                    .cast();
                _mm_prefetch::<_MM_HINT_T0>(pf_addr);
                let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
                let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                let qs = unsafe { g.add(32) };
                let mut p = [_mm256_setzero_si256(); STRIP];
                for (lane, pl) in p.iter_mut().enumerate() {
                    let wv = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                    let aw = _mm256_sign_epi8(wv, wv);
                    let sx = _mm256_sign_epi8(xv, wv);
                    *pl = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                }
                // The one delta vs the shipped kernel: unpack tree, not hadd.
                let isum = _mm256_cvtepi32_ps(hsum8_i32_unpack(p));
                let dw = unsafe { _mm256_load_ps(g.cast()) };
                let dwdx = _mm256_mul_ps(dw, _mm256_set1_ps(dx));
                acc = _mm256_fmadd_ps(dwdx, isum, acc);
            }
            unsafe { _mm256_storeu_ps(y.add(strip * STRIP), acc) };
        }
    }
}

fn benches(c: &mut Criterion) {
    bench_dtype(c, DType::F32, SHAPES_F32);
    bench_dtype(c, DType::Q8_0, SHAPES_Q8_0);
    bench_dtype(c, DType::Q4_K, SHAPES_Q4_K);
}

/// M-loop batched GEMM: same shapes as the GEMV group, but drives
/// `KernelSet::gemm` for a representative `m` panel (1, 16, 64) so the
/// weight-reuse win of the batched kernel is visible as `m` grows.
/// Throughput is reported in MACs (`m * rows * k`), the compute-bound unit
/// GEMM performance is conventionally judged by (as opposed to the GEMV
/// group's weight-bytes-per-call basis).
const GEMM_MS: &[usize] = &[1, 16, 64];

fn bench_dtype_gemm(c: &mut Criterion, dtype: DType, shapes: &[(usize, usize)]) {
    let mut group = c.benchmark_group(format!("gemm/{dtype:?}"));
    group.sample_size(20);
    for &(rows, k) in shapes {
        let file = gen_weights(&dtype, rows, k);
        for (name, set) in sets_for(&dtype) {
            let w = set.pack(&file, rows, k).unwrap();
            for &m in GEMM_MS {
                // m-row activation panel: quantize_row per token, concatenated
                // (matches the registry's documented GEMM panel layout).
                let mut panel = Vec::new();
                for t in 0..m {
                    let x = pseudo_f32(42 + t as u64, k);
                    panel.extend_from_slice(&set.quantize_row(&x).unwrap());
                }
                let mut y = vec![0f32; m * rows];
                group.throughput(Throughput::Elements((m * rows * k) as u64));
                group.bench_function(BenchmarkId::new(name, format!("{rows}x{k}/m{m}")), |b| {
                    b.iter(|| set.gemm(&mut y, &panel, &w, m, rows, k, 0, rows).unwrap())
                });
            }
        }
        #[cfg(feature = "ggml-compare")]
        for &m in GEMM_MS {
            ggml::bench_gemm(&mut group, &dtype, &file, rows, k, m);
        }
    }
    group.finish();
}

fn benches_gemm(c: &mut Criterion) {
    bench_dtype_gemm(c, DType::F32, SHAPES_F32);
    bench_dtype_gemm(c, DType::Q8_0, SHAPES_Q8_0);
    bench_dtype_gemm(c, DType::Q4_K, SHAPES_Q4_K);
}

criterion_group!(gemv, benches);
criterion_group!(gemm, benches_gemm);
criterion_main!(gemv, gemm);
