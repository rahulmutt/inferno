//! MLX model directories: HF-style config.json + one or more .safetensors.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{Architecture, FormatError, HyperParams, ModelDesc, Result, RopeStyle, safetensors};

#[derive(Deserialize)]
struct MlxConfig {
    model_type: String,
    hidden_size: u64,
    num_hidden_layers: u64,
    num_attention_heads: u64,
    num_key_value_heads: Option<u64>,
    intermediate_size: u64,
    vocab_size: u64,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_norm_eps")]
    rms_norm_eps: f32,
    #[serde(default)]
    max_position_embeddings: u64,
}

fn default_rope_theta() -> f32 {
    10000.0
}
fn default_norm_eps() -> f32 {
    1e-5
}

pub fn load_dir(dir: &Path) -> Result<ModelDesc> {
    let config_path = dir.join("config.json");
    let config_file = File::open(&config_path).map_err(|e| FormatError::Malformed {
        context: "mlx model directory",
        detail: format!("cannot open {}: {e}", config_path.display()),
    })?;
    let config: MlxConfig = serde_json::from_reader(BufReader::new(config_file))?;

    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shard_paths.sort();
    if shard_paths.is_empty() {
        return Err(FormatError::Malformed {
            context: "mlx model directory",
            detail: format!("no .safetensors files in {}", dir.display()),
        });
    }

    let mut tensors = Vec::new();
    let mut data_section_offsets = Vec::new();
    for (i, path) in shard_paths.iter().enumerate() {
        let mut reader = BufReader::new(File::open(path)?);
        let (mut shard_tensors, data_off) = safetensors::parse(&mut reader, i as u32)?;
        tensors.append(&mut shard_tensors);
        data_section_offsets.push(data_off);
    }

    let tokenizer_json = dir.join("tokenizer.json");
    let tokenizer = tokenizer_json
        .is_file()
        .then_some(crate::desc::TokenizerSpec::HfJson {
            path: tokenizer_json,
        });

    Ok(ModelDesc {
        architecture: Architecture::from_id(&config.model_type),
        name: dir.file_name().map(|n| n.to_string_lossy().into_owned()),
        hyperparams: HyperParams {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            n_layers: config.num_hidden_layers,
            n_heads: config.num_attention_heads,
            n_kv_heads: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            ffn_hidden_size: config.intermediate_size,
            rope_theta: config.rope_theta,
            norm_eps: config.rms_norm_eps,
            context_length: config.max_position_embeddings,
            rope_style: RopeStyle::HalfSplit,
        },
        tensors,
        weight_files: shard_paths,
        data_section_offsets,
        tokenizer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Architecture, fixtures};

    // tempfile::tempdir() creates a securely-permissioned, uniquely-named
    // directory (not a predictable pid-based path in the shared system temp
    // dir) and removes it on drop.
    fn write_tiny_mlx_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            fixtures::tiny_llama_config_json(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors"),
            fixtures::tiny_llama_safetensors(),
        )
        .unwrap();
        dir
    }

    #[test]
    fn loads_tiny_mlx_dir() {
        let dir = write_tiny_mlx_dir();
        let desc = load_dir(dir.path()).unwrap();
        assert_eq!(desc.architecture, Architecture::Llama);
        // MLX models are always HF half-split rope layout, unlike the
        // GGUF-oriented fixtures::tiny_hyperparams() default.
        assert_eq!(
            desc.hyperparams,
            HyperParams {
                rope_style: RopeStyle::HalfSplit,
                ..fixtures::tiny_hyperparams()
            }
        );
        assert_eq!(desc.tensors.len(), fixtures::tiny_tensor_shapes().len());
        assert_eq!(
            desc.weight_files,
            vec![dir.path().join("model.safetensors")]
        );
        assert_eq!(desc.data_section_offsets.len(), 1);
    }

    #[test]
    fn missing_config_is_clear_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("model.safetensors"),
            fixtures::tiny_llama_safetensors(),
        )
        .unwrap();
        let err = load_dir(dir.path()).unwrap_err().to_string();
        assert!(err.contains("config.json"), "unhelpful error: {err}");
    }

    #[test]
    fn no_safetensors_is_clear_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            fixtures::tiny_llama_config_json(),
        )
        .unwrap();
        assert!(load_dir(dir.path()).is_err());
    }

    #[test]
    fn minimal_config_applies_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let config = r#"{
  "model_type": "llama",
  "hidden_size": 8,
  "num_hidden_layers": 2,
  "num_attention_heads": 2,
  "intermediate_size": 16,
  "vocab_size": 32
}"#;
        std::fs::write(dir.path().join("config.json"), config).unwrap();
        std::fs::write(
            dir.path().join("model.safetensors"),
            fixtures::tiny_llama_safetensors(),
        )
        .unwrap();
        let desc = load_dir(dir.path()).unwrap();
        assert_eq!(desc.hyperparams.n_kv_heads, desc.hyperparams.n_heads);
        assert_eq!(desc.hyperparams.rope_theta, 10000.0);
        assert_eq!(desc.hyperparams.norm_eps, 1e-5);
        assert_eq!(desc.hyperparams.context_length, 0);
    }

    #[test]
    fn detects_tokenizer_json_and_halfsplit_rope() {
        let dir = write_tiny_mlx_dir();
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        let desc = load_dir(dir.path()).unwrap();
        assert_eq!(desc.hyperparams.rope_style, crate::RopeStyle::HalfSplit);
        match desc.tokenizer {
            Some(crate::desc::TokenizerSpec::HfJson { path }) => {
                assert_eq!(path, dir.path().join("tokenizer.json"));
            }
            other => panic!("expected HfJson, got {other:?}"),
        }
    }

    #[test]
    fn no_tokenizer_json_means_none() {
        let dir = write_tiny_mlx_dir();
        assert!(load_dir(dir.path()).unwrap().tokenizer.is_none());
    }

    #[test]
    fn multi_shard_dir_orders_files_and_tracks_file_index() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            fixtures::tiny_llama_config_json(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model-00001-of-00002.safetensors"),
            fixtures::tiny_llama_safetensors(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model-00002-of-00002.safetensors"),
            fixtures::tiny_llama_safetensors(),
        )
        .unwrap();
        let desc = load_dir(dir.path()).unwrap();
        assert_eq!(
            desc.weight_files,
            vec![
                dir.path().join("model-00001-of-00002.safetensors"),
                dir.path().join("model-00002-of-00002.safetensors"),
            ]
        );
        assert_eq!(desc.data_section_offsets.len(), 2);
        assert!(desc.tensors.iter().filter(|t| t.file_index == 0).count() > 0);
        assert!(desc.tensors.iter().filter(|t| t.file_index == 1).count() > 0);
    }
}
