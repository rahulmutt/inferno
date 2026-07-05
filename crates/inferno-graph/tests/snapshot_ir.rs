use std::path::Path;

use inferno_formats::load_desc;
use inferno_graph::build_graph;

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferno-formats/tests/fixtures")
}

#[test]
fn gguf_fixture_graph_snapshot() {
    let desc = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    let g = build_graph(&desc).unwrap();
    insta::assert_snapshot!("tiny_gguf_ir", g.dump(&desc));
}

#[test]
fn mlx_fixture_graph_snapshot() {
    let desc = load_desc(&fixture_dir().join("mlx")).unwrap();
    let g = build_graph(&desc).unwrap();
    insta::assert_snapshot!("tiny_mlx_ir", g.dump(&desc));
}

#[test]
fn tied_embeddings_reuse_token_embed() {
    let desc = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    let g = build_graph(&desc).unwrap();
    let embed_idx = desc
        .tensors
        .iter()
        .position(|t| t.name == "token_embed.weight")
        .unwrap();
    // Final matmul's weight must be the embedding table (no lm_head in fixture).
    let last = g.nodes.last().unwrap();
    match &last.op {
        inferno_graph::Op::MatMul { weight, .. } => assert_eq!(weight.0, embed_idx),
        other => panic!("expected final MatMul, got {other:?}"),
    }
}

#[test]
fn missing_tensor_is_typed_error() {
    let mut desc = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    desc.tensors
        .retain(|t| t.name != "layers.0.ffn.gate_proj.weight");
    assert!(matches!(
        build_graph(&desc),
        Err(inferno_graph::GraphError::MissingTensor(name)) if name.contains("gate_proj")
    ));
}

#[test]
fn unknown_arch_is_typed_error() {
    let mut desc = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    desc.architecture = inferno_formats::Architecture::Unknown("mamba".into());
    assert!(matches!(
        build_graph(&desc),
        Err(inferno_graph::GraphError::UnsupportedArch(_))
    ));
}

#[test]
fn hostile_hyperparams_are_typed_errors() {
    let base = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    for mutate in [
        (|d: &mut inferno_formats::ModelDesc| d.hyperparams.n_heads = 0)
            as fn(&mut inferno_formats::ModelDesc),
        |d| d.hyperparams.n_heads = 3,           // hidden 64 % 3 != 0
        |d| d.hyperparams.n_kv_heads = 5,        // heads % kv != 0
        |d| d.hyperparams.hidden_size = 1 << 30, // allocation guard
        |d| d.hyperparams.vocab_size = u64::MAX,
    ] {
        let mut d = base.clone();
        mutate(&mut d);
        assert!(
            matches!(
                build_graph(&d),
                Err(inferno_graph::GraphError::BadHyperParams(_))
            ),
            "hyperparam mutation not rejected"
        );
    }
}

#[test]
fn qwen2_biases_and_qwen3_qk_norm_are_wired() {
    // Synthesize: take the fixture desc, relabel arch, add bias/qk-norm
    // tensor descs (data never read at build time).
    let mut desc = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    desc.architecture = inferno_formats::Architecture::Qwen3;
    let template = desc.tensors[0].clone();
    for i in 0..2 {
        for (name, shape) in [
            (format!("layers.{i}.attn.q_proj.bias"), vec![64u64]),
            (format!("layers.{i}.attn.k_proj.bias"), vec![32]),
            (format!("layers.{i}.attn.v_proj.bias"), vec![32]),
            (format!("layers.{i}.attn.q_norm.weight"), vec![32]),
            (format!("layers.{i}.attn.k_norm.weight"), vec![32]),
        ] {
            let mut t = template.clone();
            t.name = name;
            t.shape = shape;
            t.dtype = inferno_formats::DType::F32;
            desc.tensors.push(t);
        }
    }
    let g = build_graph(&desc).unwrap();
    let dump = g.dump(&desc);
    assert!(dump.contains("bias=@layers.0.attn.q_proj.bias"));
    assert!(dump.contains("rmsnorm_per_head"));
}
