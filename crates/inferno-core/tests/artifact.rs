//! Task 14 gate: the verified artifact loader.
//!
//! Two tests, both driving the PUBLIC `Artifact` API:
//! 1. `compiled_prefill_matches_interpreter` — a real differential: compile +
//!    mmap + dlopen + `prefill`, last-token logits vs the scalar interpreter
//!    within `logits_abs_tol`.
//! 2. `tampered_meta_is_rejected` — corrupt a cached `weights.bin`, assert the
//!    loader detects the hash mismatch and recompiles rather than `dlopen`ing
//!    stale/tampered code.
//!
//! Lives in `tests/` (not a `#[cfg(test)]` module) so the crate's `build.rs`
//! `cargo:rustc-link-arg-tests=-rdynamic` applies to this binary, exporting the
//! statically-linked kernel symbols the `dlopen`ed `model.so` resolves against.

use std::path::{Path, PathBuf};

use inferno_core::artifact::verify_cache;
use inferno_core::{Artifact, CoreError, Meta, cache_dir, cache_key, content_hash};
use inferno_formats::load_desc;
use inferno_graph::tolerance::logits_abs_tol;
use inferno_graph::{Interpreter, KvCache, build_graph};
use inferno_target::TargetDesc;

// tiny.gguf is already GQA (n_heads=2, n_kv_heads=1); the brief's tiny_gqa.gguf
// was never created, so per the Task 14 instructions we use tiny.gguf.
const MODEL: &str = "../inferno-formats/tests/fixtures/tiny.gguf";

/// Point the cache at a fixed per-process temp dir. Every test sets the SAME
/// string, so concurrent `set_var` is race-free on the value; per-test cache
/// isolation comes from a distinct `max_seq_len` (part of the cache key).
fn use_temp_cache() {
    let dir = std::env::temp_dir().join("inferno-core-artifact-tests");
    // SAFETY: the value is identical across all callers, so interleaving with
    // another test thread's identical `set_var`/`var_os` read is benign.
    unsafe { std::env::set_var("XDG_CACHE_HOME", &dir) };
}

fn model_path() -> PathBuf {
    Path::new(MODEL).to_path_buf()
}

#[test]
fn compiled_prefill_matches_interpreter() {
    use_temp_cache();
    let model = model_path();
    let target = TargetDesc::detect().unwrap();
    let art = Artifact::load_or_compile(&model, &target, 64).unwrap();

    let desc = load_desc(&model).unwrap();
    let graph = build_graph(&desc).unwrap();
    let vocab = desc.hyperparams.vocab_size as usize;
    let tokens = vec![1u32, 4, 7, 2];

    let mut kv = vec![0f32; art.meta().kv_total_bytes / 4];
    let mut arena = vec![0f32; art.meta().arena_f32];
    let mut logits = vec![0f32; vocab];
    art.prefill(&tokens, 0, &mut kv, &mut arena, &mut logits);

    let mut interp = Interpreter::new();
    let mut ikv = KvCache::new(&graph, 64).unwrap();
    let want = interp.run(&desc, &graph, &tokens, &mut ikv).unwrap();
    let want_last = &want.data[(tokens.len() - 1) * vocab..][..vocab];
    let tol = logits_abs_tol(&inferno_formats::DType::Q8_0);
    let max = logits
        .iter()
        .zip(want_last)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!("[artifact] compiled vs interp max |Δlogit| = {max:e} (tol {tol:e})");
    assert!(max <= tol, "max |Δ| {max} > {tol}");
}

#[test]
fn tampered_meta_is_rejected() {
    use_temp_cache();
    let model = model_path();
    let target = TargetDesc::detect().unwrap();
    // Distinct max_seq_len => distinct cache key/dir, isolating this test from
    // the differential test's cache entry.
    let seq = 96usize;

    // Populate the cache and capture the correct logits.
    let art = Artifact::load_or_compile(&model, &target, seq).unwrap();
    let desc = load_desc(&model).unwrap();
    let vocab = desc.hyperparams.vocab_size as usize;
    let tokens = vec![1u32, 4, 7, 2];
    let mut kv = vec![0f32; art.meta().kv_total_bytes / 4];
    let mut arena = vec![0f32; art.meta().arena_f32];
    let mut good = vec![0f32; vocab];
    art.prefill(&tokens, 0, &mut kv, &mut arena, &mut good);
    drop(art);

    // Tamper: flip one byte of the cached weights.bin.
    let dir = cache_dir(&cache_key(&model, &target, seq).unwrap());
    let wpath = dir.join("weights.bin");
    let mut bytes = std::fs::read(&wpath).unwrap();
    bytes[0] ^= 0xff;
    let tampered_hash = content_hash(&bytes);
    std::fs::write(&wpath, &bytes).unwrap();

    // The stored meta still records the ORIGINAL hash, which no longer matches
    // the tampered file — so verify_cache must reject the entry.
    let stored: Meta =
        serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    assert_ne!(
        stored.weights_hash, tampered_hash,
        "precondition: tampered file hash differs from meta's recorded hash"
    );
    assert!(
        matches!(verify_cache(&dir, &model), Err(CoreError::Verification(_))),
        "verify_cache must reject the tampered weights.bin"
    );

    // A reload must NOT dlopen the tampered artifact: it recompiles, restoring a
    // consistent weights.bin/meta pair and reproducing the exact logits.
    let art2 = Artifact::load_or_compile(&model, &target, seq).unwrap();
    let restored = std::fs::read(&wpath).unwrap();
    assert_eq!(
        content_hash(&restored),
        art2.meta().weights_hash,
        "reload must rewrite a consistent weights.bin/meta pair (recompiled)"
    );
    // And the recompiled artifact now verifies cleanly.
    assert!(
        verify_cache(&dir, &model).is_ok(),
        "recompiled cache entry must verify"
    );
    let mut got = vec![0f32; vocab];
    art2.prefill(&tokens, 0, &mut kv, &mut arena, &mut got);
    let max = good
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        max == 0.0,
        "recompiled artifact must reproduce original logits exactly, max |Δ| = {max}"
    );
}
