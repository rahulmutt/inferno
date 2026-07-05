//! Task 16 gate: `CompiledBackend` driven exactly the way `Generator` drives
//! any `Backend` — one multi-token `forward` (prefill), then single-token
//! `forward` calls (decode). Lives in `tests/` (not a `#[cfg(test)]` module)
//! so the crate's `build.rs` `cargo:rustc-link-arg-tests=-rdynamic` applies to
//! this binary, exporting the statically-linked kernel symbols the
//! `dlopen`ed `model.so` resolves against — see `tests/artifact.rs`.

use std::path::Path;

use inferno_core::{CompiledBackend, Engine};
use inferno_formats::load_desc;
use inferno_graph::tolerance::logits_abs_tol;
use inferno_graph::{Interpreter, KvCache, build_graph};
use inferno_runtime::Backend;

const MODEL: &str = "../inferno-formats/tests/fixtures/tiny.gguf";

fn use_temp_cache() {
    let dir = std::env::temp_dir().join("inferno-core-backend-tests");
    // SAFETY: every caller sets the identical value, so a racing
    // `set_var`/`var_os` from another test thread observes the same string
    // either way.
    unsafe { std::env::set_var("XDG_CACHE_HOME", &dir) };
}

fn backend(max_seq_len: usize) -> CompiledBackend {
    let model = Path::new(MODEL);
    let engine = Engine::load(model, max_seq_len).unwrap();
    engine.compiled_backend().unwrap()
}

// CompiledBackend.forward must match the interpreter's last-token logits,
// driven the same way `Generator` drives any `Backend`: one multi-token
// prefill call, then single-token decode calls.
#[test]
fn compiled_backend_matches_interpreter_prefill_then_decode() {
    use_temp_cache();
    let max_seq_len = 64;
    let mut backend = backend(max_seq_len);

    let prompt = vec![1u32, 4, 7];
    let got_prefill = backend.forward(&prompt).unwrap();
    let got_decode = backend.forward(&[2u32]).unwrap();

    let model = Path::new(MODEL);
    let desc = load_desc(model).unwrap();
    let graph = build_graph(&desc).unwrap();
    let vocab = desc.hyperparams.vocab_size as usize;
    let mut interp = Interpreter::new();
    let mut kv = KvCache::new(&graph, max_seq_len).unwrap();
    let full = vec![1u32, 4, 7, 2];
    let want = interp.run(&desc, &graph, &full, &mut kv).unwrap();
    let want_last = &want.data[(full.len() - 1) * vocab..][..vocab];

    let tol = logits_abs_tol(&inferno_formats::DType::Q8_0);
    let max = got_decode
        .iter()
        .zip(want_last)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(max <= tol, "max |Δ| {max} > {tol}");
    assert_eq!(got_prefill.len(), vocab);
}

#[test]
fn reset_allows_a_fresh_sequence() {
    use_temp_cache();
    let mut backend = backend(80);

    let a = backend.forward(&[1u32, 4, 7]).unwrap();
    backend.reset();
    let b = backend.forward(&[1u32, 4, 7]).unwrap();
    assert_eq!(a, b, "identical prompt after reset must reproduce logits");
}
