use std::path::Path;

use inferno_graph::tolerance::LOGIT_TIE_EPSILON;
use inferno_runtime::{Generator, Greedy, teacher_forced};

fn generator() -> Generator {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferno-formats/tests/fixtures/tiny.gguf");
    Generator::load(&p, 64).unwrap()
}

#[test]
fn own_greedy_output_agrees_perfectly() {
    // Feed our own greedy generation back as the forced sequence: every
    // position must match (same weights, same math).
    let mut g = generator();
    let prompt = g.encode("the").unwrap();
    let (ids, _) = g
        .generate("the", 6, &mut Greedy, &mut |_| {
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    let out = teacher_forced(&mut g, &prompt, &ids).unwrap();
    assert!(out.passed(), "mismatches: {:?}", out.mismatches);
    assert_eq!(out.checked, ids.len());
    assert_eq!(out.matched + out.ties, out.checked);
}

#[test]
fn wrong_forced_token_is_reported_with_position_and_top5() {
    // The fixture's weights are drawn from a narrow range ([-0.125, 0.125),
    // see inferno-formats/src/fixtures.rs), so for the prompt "the" every
    // forced position's own top-1/top-2 gap stays under LOGIT_TIE_EPSILON
    // (verified empirically up to 60 generated tokens: gap plateaus ~0.04).
    // Any corruption there would legitimately be tolerated as a tie, not
    // reported as a mismatch. "cat dog" does produce a position with a
    // genuine (non-tie) gap, so locate it dynamically rather than
    // hardcoding a position number that depends on fixture internals.
    let mut g = generator();
    let prompt = g.encode("cat dog").unwrap();
    let (mut ids, _) = g
        .generate("cat dog", 6, &mut Greedy, &mut |_| {
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    let logits = g
        .full_logits(&[prompt.clone(), ids.clone()].concat())
        .unwrap();
    let vocab = g.vocab_size();
    let top2_gap = |row: &[f32]| {
        let mut top2 = [f32::NEG_INFINITY; 2];
        for &v in row {
            if v > top2[0] {
                top2[1] = top2[0];
                top2[0] = v;
            } else if v > top2[1] {
                top2[1] = v;
            }
        }
        top2[0] - top2[1]
    };
    let corrupt_at = (0..ids.len())
        .find(|&i| {
            let row_idx = prompt.len() + i - 1;
            let row = &logits.data[row_idx * vocab..(row_idx + 1) * vocab];
            top2_gap(row) > LOGIT_TIE_EPSILON
        })
        .expect("fixture should contain a position with a non-tie top-2 gap");
    // Corrupt that position with a token that is definitely not the argmax
    // AND whose gap exceeds the tie epsilon (pick the argmin instead).
    let row_idx = prompt.len() + corrupt_at - 1;
    let row = &logits.data[row_idx * vocab..(row_idx + 1) * vocab];
    let worst = row
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;
    ids[corrupt_at] = worst;
    let out = teacher_forced(&mut g, &prompt, &ids).unwrap();
    // Positions after the corruption legitimately diverge (different
    // history); the FIRST mismatch must be exactly the corrupted position.
    assert!(!out.passed());
    let first = &out.mismatches[0];
    assert_eq!(first.position, corrupt_at);
    assert_eq!(first.expected, worst);
    assert_eq!(first.top.len(), 5);
}

#[test]
fn vocab_size_one_is_a_typed_error_not_a_panic() {
    // Hostile model: build_graph only rejects vocab_size == 0, and a
    // 1-token tokenizer parses cleanly, so `inferno diff` would otherwise
    // reach `top[1]` in diff.rs's top-2 gap computation with a vocab of
    // size 1 and panic. teacher_forced() must reject it up front instead.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("hostile.gguf");
    std::fs::write(&path, inferno_formats::fixtures::hostile_vocab1_gguf()).unwrap();
    let mut g = Generator::load(&path, 8).unwrap();
    assert_eq!(g.vocab_size(), 1);

    let err = teacher_forced(&mut g, &[0], &[0]);
    assert!(
        matches!(err, Err(inferno_runtime::RuntimeError::VocabTooSmall(1))),
        "expected a typed VocabTooSmall error, got {err:?}"
    );
}
