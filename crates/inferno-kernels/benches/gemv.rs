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
