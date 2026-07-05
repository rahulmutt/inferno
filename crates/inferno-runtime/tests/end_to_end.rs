use std::path::Path;

use inferno_runtime::{Generator, Greedy};

fn fixture(p: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../inferno-formats/tests/fixtures")
        .join(p)
}

fn generate_ids(model: &Path) -> Vec<u32> {
    let mut g = Generator::load(model, 64).unwrap();
    let mut sink = Vec::new();
    let (ids, stats) = g
        .generate("the", 8, &mut Greedy, &mut |b| {
            sink.extend_from_slice(b);
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(stats.generated, ids.len());
    assert!(stats.prompt_tokens >= 1);
    ids
}

#[test]
fn gguf_fixture_generates_deterministic_tokens() {
    let a = generate_ids(&fixture("tiny.gguf"));
    let b = generate_ids(&fixture("tiny.gguf"));
    assert_eq!(a, b);
    assert!(!a.is_empty());
}

#[test]
fn gguf_and_mlx_generate_identical_tokens() {
    // Spec acceptance: the two formats of the same model produce the same
    // greedy tokens (same effective weights; see Task 9's logit differential).
    assert_eq!(
        generate_ids(&fixture("tiny.gguf")),
        generate_ids(&fixture("mlx"))
    );
}

#[test]
fn max_tokens_bounds_generation() {
    let mut g = Generator::load(&fixture("tiny.gguf"), 64).unwrap();
    let (ids, _) = g
        .generate("the", 3, &mut Greedy, &mut |_| {
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    assert!(ids.len() <= 3);
}

/// The decode loop must stop when the KV context fills — the in-loop
/// `seq_len + 1 > max_seq_len` break (Task 15 refactored this from the
/// backend's `kv.len()`). With a 1-token prompt, decoding fills the rest of
/// the context and then halts *exactly* at `max_seq_len`, well before the
/// much larger `max_tokens`. Asserted with `==` so a future off-by-one
/// (1-based `step`, or `>=` instead of `>`) fails here.
#[test]
fn decode_stops_at_context_full_boundary() {
    for max_seq_len in [2usize, 3] {
        let mut g = Generator::load(&fixture("tiny.gguf"), max_seq_len).unwrap();
        let (ids, stats) = g
            // 1-token prompt "the"; max_tokens=10 >> the context allows.
            .generate("the", 10, &mut Greedy, &mut |_| {
                std::ops::ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(
            ids.len(),
            max_seq_len,
            "generation must stop at the context bound (max_seq_len={max_seq_len}), \
             not at max_tokens",
        );
        assert_eq!(stats.generated, ids.len());
    }
}

#[test]
fn prompt_longer_than_max_seq_len_is_typed_error() {
    let mut g = Generator::load(&fixture("tiny.gguf"), 2).unwrap();
    let err = g.generate("the cat sat on the mat", 4, &mut Greedy, &mut |_| {
        std::ops::ControlFlow::Continue(())
    });
    assert!(matches!(
        err,
        Err(inferno_runtime::RuntimeError::PromptTooLong { .. })
    ));
}
