use std::path::Path;

use inferno_formats::load_desc;
use inferno_graph::{Interpreter, KvCache, build_graph};

fn fixture(p: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../inferno-formats/tests/fixtures")
        .join(p)
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}

#[test]
fn prefill_then_decode_matches_single_prefill() {
    let desc = load_desc(&fixture("tiny.gguf")).unwrap();
    let graph = build_graph(&desc).unwrap();
    let mut interp = Interpreter::new();
    let toks = [3u32, 5, 250, 42];

    let mut kv_a = KvCache::new(&graph, 16).unwrap();
    let all = interp.run(&desc, &graph, &toks, &mut kv_a).unwrap();

    let mut kv_b = KvCache::new(&graph, 16).unwrap();
    let _ = interp.run(&desc, &graph, &toks[..3], &mut kv_b).unwrap();
    let last = interp.run(&desc, &graph, &toks[3..], &mut kv_b).unwrap();

    // Same final-position logits whether the last token came via prefill or decode.
    let vocab = desc.hyperparams.vocab_size as usize;
    let a = &all.data[3 * vocab..4 * vocab];
    let b = &last.data[..vocab];
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
}

#[test]
fn gguf_and_mlx_fixtures_agree() {
    // THE two-formats boundary test: same effective weights via different
    // formats, names, rope styles, and Q/K permutation must produce the
    // same logits (within float tolerance) and the same argmax chain.
    let dg = load_desc(&fixture("tiny.gguf")).unwrap();
    let dm = load_desc(&fixture("mlx")).unwrap();
    let gg = build_graph(&dg).unwrap();
    let gm = build_graph(&dm).unwrap();
    let toks = [1u32, 200, 116, 104, 101];
    let mut ig = Interpreter::new();
    let mut im = Interpreter::new();
    let mut kg = KvCache::new(&gg, 16).unwrap();
    let mut km = KvCache::new(&gm, 16).unwrap();
    let lg = ig.run(&dg, &gg, &toks, &mut kg).unwrap();
    let lm = im.run(&dm, &gm, &toks, &mut km).unwrap();
    assert_eq!(lg.shape, lm.shape);
    let widest = inferno_formats::DType::Q4_K; // widest dtype in the fixture
    let tol = inferno_graph::tolerance::logits_abs_tol(&widest);
    for (i, (a, b)) in lg.data.iter().zip(&lm.data).enumerate() {
        assert!((a - b).abs() <= tol, "logit {i}: {a} vs {b}");
    }
    let vocab = dg.hyperparams.vocab_size as usize;
    for p in 0..toks.len() {
        assert_eq!(
            argmax(&lg.data[p * vocab..(p + 1) * vocab]),
            argmax(&lm.data[p * vocab..(p + 1) * vocab]),
            "argmax diverged at position {p}"
        );
    }
}

#[test]
fn run_is_deterministic() {
    let desc = load_desc(&fixture("tiny.gguf")).unwrap();
    let graph = build_graph(&desc).unwrap();
    let toks = [7u32, 8, 9];
    let mut i1 = Interpreter::new();
    let mut i2 = Interpreter::new();
    let mut k1 = KvCache::new(&graph, 8).unwrap();
    let mut k2 = KvCache::new(&graph, 8).unwrap();
    let a = i1.run(&desc, &graph, &toks, &mut k1).unwrap();
    let b = i2.run(&desc, &graph, &toks, &mut k2).unwrap();
    assert_eq!(a.data, b.data); // bitwise: scalar f32, fixed order
}

#[test]
fn seq_overflow_and_hostile_kv_are_errors() {
    let desc = load_desc(&fixture("tiny.gguf")).unwrap();
    let graph = build_graph(&desc).unwrap();
    let mut kv = KvCache::new(&graph, 2).unwrap();
    let mut interp = Interpreter::new();
    assert!(matches!(
        interp.run(&desc, &graph, &[1, 2, 3], &mut kv),
        Err(inferno_graph::GraphError::SeqTooLong { got: 3, max: 2 })
    ));
    // Hostile hyperparams cannot make KvCache::new allocate unboundedly.
    let mut big = graph.clone();
    big.n_layers = u64::MAX / 4;
    assert!(KvCache::new(&big, 1 << 20).is_err());
}
