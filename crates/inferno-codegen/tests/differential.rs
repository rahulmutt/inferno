//! THE GATE (Task 12): compiled-vs-interpreter differential.
//!
//! Compiles a tiny model to `model.so`, `dlopen`s it, runs `prefill`, and
//! asserts the last-token logits match the scalar interpreter oracle within
//! `logits_abs_tol`. This is the milestone's correctness gate: the whole
//! compiled path (planner + LLVM lowering + object emit + link + kernel
//! dispatch) executes end to end here for the first time.
//!
//! # unsafe
//! This is `#[cfg(test)]` integration code. It `dlopen`s a shared object and
//! calls a raw C-ABI entry point through pointers, which is inherently
//! `unsafe`. Production codegen stays `unsafe`-free (`[lints] workspace = true`
//! denies `unsafe_code`); a `tests/` integration file is a separate crate, but
//! the workspace lint table still applies, so we scope the allow to this file
//! with this justification rather than loosening the crate policy.
#![allow(unsafe_code)]

use std::path::Path;

use inferno_codegen::{CompileOptions, compile};
use inferno_formats::{DType, ModelDesc, load_desc};
use inferno_graph::tolerance::logits_abs_tol;
use inferno_graph::{Interpreter, KvCache, build_graph};
use inferno_target::TargetDesc;
use serde::Deserialize;

/// Sidecar metadata read back from `meta.json` (the codegen `Meta` derives only
/// `Serialize`; this is the read side).
#[derive(Debug, Deserialize)]
struct Meta {
    vocab: usize,
    arena_f32: usize,
    kv_total_bytes: usize,
}

/// `prefill(tokens, n, pos_off, weights, kv, arena, logits_out)` — the C ABI of
/// the generated entry point (see `declare_entry_points`). `n`/`pos_off` are
/// i64 params, surfaced here as `usize`.
type PrefillFn = unsafe extern "C" fn(
    *const u32, // tokens
    usize,      // n
    usize,      // pos_off
    *const u8,  // weights image base
    *mut f32,   // kv cache
    *mut f32,   // arena
    *mut f32,   // logits_out
);

/// Force the linker to retain (and, with `-rdynamic` from build.rs, export) the
/// kernel symbols *and* the `inferno_par_gemv` dispatcher (M4b.1) the compiled
/// `model.so` resolves against the host binary. Without at least one
/// reference the linker may drop `inferno-kernels`/`inferno-pool` entirely,
/// leaving nothing to export and `dlopen` failing on the first undefined
/// `inferno_gemv_*` / `inferno_quantize_row_*` / `inferno_par_gemv` symbol.
fn retain_kernel_symbols() {
    use std::hint::black_box;
    let p = |f: *const ()| black_box(f as usize);
    p(inferno_kernels::inferno_gemv_f32_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemv_f32_rs8_avx2 as *const ());
    p(inferno_kernels::inferno_gemv_q8_0_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemv_q8_0_rs8_avx2 as *const ());
    p(inferno_kernels::inferno_gemv_q4_k_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemv_q4_k_rs8_avx2 as *const ());
    p(inferno_kernels::act::inferno_quantize_row_q8a_scalar as *const ());
    p(inferno_kernels::act::inferno_quantize_row_q8a_avx2 as *const ());
    p(inferno_kernels::act::inferno_quantize_row_q8k_scalar as *const ());
    p(inferno_kernels::act::inferno_quantize_row_q8k_avx2 as *const ());
    p(inferno_pool::inferno_par_gemv as *const ());
}

/// dlopen `model.so`, run `prefill(tokens)`, and return the last-token logits.
///
/// # Safety
/// `art_dir` must be a valid compiled artifact directory (`model.so` +
/// `weights.bin`) produced by [`compile`] for a model whose `meta` sizes match.
unsafe fn run_compiled(art_dir: &Path, tokens: &[u32], meta: &Meta) -> Vec<f32> {
    retain_kernel_symbols();

    // The loaded lib resolves its undefined kernel symbols against already
    // loaded globals, including the `-rdynamic`-exported test executable.
    let lib =
        unsafe { libloading::Library::new(art_dir.join("model.so")) }.expect("dlopen model.so");
    let prefill: libloading::Symbol<PrefillFn> =
        unsafe { lib.get(b"prefill\0") }.expect("resolve prefill symbol");

    // The rs8 GEMV kernels use *aligned* AVX2 loads (`_mm256_load_*`) on the
    // weight base; packed weight offsets are multiples of 32, so the base
    // pointer must be 32-byte aligned too. The production loader (Task 14)
    // mmaps `weights.bin` (page-aligned); here we copy it into a 32-aligned
    // `AlignedBuf` to satisfy the same contract. A plain `Vec<u8>` from
    // `std::fs::read` is only ~16-aligned and segfaults on the first strip.
    let raw = std::fs::read(art_dir.join("weights.bin")).expect("read weights.bin");
    let mut weights = inferno_kernels::AlignedBuf::zeroed(raw.len());
    weights.as_mut_slice().copy_from_slice(&raw);
    let mut arena = vec![0.0f32; meta.arena_f32];
    let mut kv = vec![0.0f32; meta.kv_total_bytes / 4];
    let mut logits = vec![0.0f32; meta.vocab];

    unsafe {
        prefill(
            tokens.as_ptr(),
            tokens.len(),
            0,
            weights.as_ptr(),
            kv.as_mut_ptr(),
            arena.as_mut_ptr(),
            logits.as_mut_ptr(),
        );
    }
    logits
}

/// The least-precise (widest error band) weight dtype present, used only to
/// pick the `logits_abs_tol` arm: any quantized weight (Q4_K/Q8_0) drives the
/// whole model onto the quant tolerance (~1e-2), else the f32 arm (~1e-4).
fn widest_dtype(desc: &ModelDesc) -> DType {
    let rank = |d: &DType| match d {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::Q8_0 => 3,
        DType::Q4_K => 4,
        DType::Unsupported(_) => 0,
    };
    desc.tensors
        .iter()
        .map(|t| t.dtype.clone())
        .max_by_key(|d| rank(d))
        .unwrap_or(DType::F32)
}

fn differential_for(fixture: &str) {
    let desc = load_desc(Path::new(fixture)).unwrap();
    let graph = build_graph(&desc).unwrap();
    let target = TargetDesc::detect().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let art = compile(
        &desc,
        &graph,
        &target,
        64,
        &CompileOptions::default(),
        tmp.path(),
    )
    .unwrap();

    let tokens: Vec<u32> = vec![1, 5, 9, 3]; // any in-vocab prompt
    let meta: Meta =
        serde_json::from_slice(&std::fs::read(art.dir.join("meta.json")).unwrap()).unwrap();
    let got = unsafe { run_compiled(&art.dir, &tokens, &meta) };

    let mut interp = Interpreter::new();
    let mut kv = KvCache::new(&graph, 64).unwrap();
    let want = interp.run(&desc, &graph, &tokens, &mut kv).unwrap();
    let vocab = desc.hyperparams.vocab_size as usize;
    let want_last = &want.data[(tokens.len() - 1) * vocab..][..vocab];

    let tol = logits_abs_tol(&widest_dtype(&desc));
    let max_abs = got
        .iter()
        .zip(want_last)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[differential] {fixture}: max |Δlogit| = {max_abs:e} (tol {tol:e})");
    assert!(
        max_abs <= tol,
        "compiled vs interp max |Δlogit| = {max_abs} > tol {tol} (fixture {fixture})"
    );
}

/// The profiler must not perturb the math: a `--profile` build's last-token
/// logits must be **bitwise** identical to the plain build's (not merely within
/// tolerance). `readcyclecounter` only reads a clock; if any logit's bits
/// differ, the instrumentation changed an SSA value or basic-block boundary and
/// the codegen is wrong. This is a hard correctness gate.
#[test]
fn profiling_does_not_change_logits() {
    let fixture = "../inferno-formats/tests/fixtures/tiny.gguf";
    let desc = load_desc(Path::new(fixture)).unwrap();
    let graph = build_graph(&desc).unwrap();
    let target = TargetDesc::detect().unwrap();
    let tokens: Vec<u32> = vec![1, 5, 9, 3];

    let plain_dir = tempfile::tempdir().unwrap();
    let a = compile(
        &desc,
        &graph,
        &target,
        64,
        &CompileOptions::default(),
        plain_dir.path(),
    )
    .unwrap();
    let prof_dir = tempfile::tempdir().unwrap();
    let b = compile(
        &desc,
        &graph,
        &target,
        64,
        &CompileOptions {
            profile: true,
            prefill_tile: 64,
        },
        prof_dir.path(),
    )
    .unwrap();

    let read_meta = |art: &inferno_codegen::Artifact| -> Meta {
        serde_json::from_slice(&std::fs::read(art.dir.join("meta.json")).unwrap()).unwrap()
    };
    let la = unsafe { run_compiled(&a.dir, &tokens, &read_meta(&a)) };
    let lb = unsafe { run_compiled(&b.dir, &tokens, &read_meta(&b)) };
    for (i, (x, y)) in la.iter().zip(&lb).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "logit {i} differs with --profile");
    }
}

#[test]
fn differential_tiny_gguf() {
    // tiny.gguf is already GQA: n_heads=2, n_kv_heads=1 (group=2), so this
    // exercises the `n_kv_heads < n_heads` path.
    differential_for("../inferno-formats/tests/fixtures/tiny.gguf");
}

#[test]
fn differential_tiny_mlx() {
    differential_for("../inferno-formats/tests/fixtures/mlx");
}

#[test]
fn differential_tiny_bias() {
    // tiny_bias.gguf carries q/k/v attention biases (as Qwen2/Qwen2.5 do), so
    // `build_graph` emits `Op::MatMul { bias: Some(_) }` and the compiled
    // `Step::Bias` lowering (`lower_bias`) is exercised. This is the ONLY
    // differential that puts the compiled bias-add under the correctness gate.
    let fixture = "../inferno-formats/tests/fixtures/tiny_bias.gguf";

    // Guard: the fixture must genuinely produce biased MatMuls, or the gate is
    // vacuous. If this fails, the fixture is wrong (missing/misnamed biases).
    let desc = load_desc(Path::new(fixture)).unwrap();
    let graph = build_graph(&desc).unwrap();
    let biased = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, inferno_graph::Op::MatMul { bias: Some(_), .. }))
        .count();
    assert!(
        biased > 0,
        "tiny_bias.gguf produced no biased MatMuls — bias lowering would not be exercised"
    );

    differential_for(fixture);
}
