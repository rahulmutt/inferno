//! Regenerates the committed test fixtures and fuzz corpus seeds.
//! Run: cargo run -p inferno-formats --example gen_fixtures

use std::fs;
use std::path::Path;

use inferno_formats::fixtures;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fix = root.join("tests/fixtures");
    fs::create_dir_all(fix.join("mlx")).unwrap();
    fs::write(fix.join("tiny.gguf"), fixtures::tiny_llama_gguf()).unwrap();
    fs::write(
        fix.join("mlx/config.json"),
        fixtures::tiny_llama_config_json(),
    )
    .unwrap();
    fs::write(
        fix.join("mlx/model.safetensors"),
        fixtures::tiny_llama_safetensors(),
    )
    .unwrap();

    // Fuzz corpus seeds (fuzz/ is created in the fuzz task; ignore if absent).
    let corpus = root.join("../../fuzz/corpus");
    if corpus
        .parent()
        .is_some_and(|p| p.join("Cargo.toml").exists())
    {
        fs::create_dir_all(corpus.join("gguf_parse")).unwrap();
        fs::create_dir_all(corpus.join("safetensors_parse")).unwrap();
        fs::write(
            corpus.join("gguf_parse/tiny.gguf"),
            fixtures::tiny_llama_gguf(),
        )
        .unwrap();
        fs::write(
            corpus.join("safetensors_parse/tiny.safetensors"),
            fixtures::tiny_llama_safetensors(),
        )
        .unwrap();
    }
    println!("fixtures written under {}", fix.display());
}
