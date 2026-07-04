//! Tiny in-memory models for tests, fuzz corpus seeds, and CLI snapshots.
//! Also consumed by later milestones (M1 interpreter tests). Not a public
//! stability surface.

use crate::HyperParams;

pub fn tiny_hyperparams() -> HyperParams {
    HyperParams {
        vocab_size: 32,
        hidden_size: 8,
        n_layers: 2,
        n_heads: 2,
        n_kv_heads: 1,
        ffn_hidden_size: 16,
        rope_theta: 10000.0,
        norm_eps: 1e-5,
        context_length: 128,
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_kv_u32(out: &mut Vec<u8>, key: &str, v: u32) {
    put_str(out, key);
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_kv_f32(out: &mut Vec<u8>, key: &str, v: f32) {
    put_str(out, key);
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_kv_str(out: &mut Vec<u8>, key: &str, v: &str) {
    put_str(out, key);
    out.extend_from_slice(&8u32.to_le_bytes());
    put_str(out, v);
}

/// Tensor list for the tiny llama: (name, row-major shape).
/// GGUF stores dims fastest-first, so the writer reverses these.
pub fn tiny_tensor_shapes() -> Vec<(String, Vec<u64>)> {
    let hp = tiny_hyperparams();
    let (v, h, f) = (hp.vocab_size, hp.hidden_size, hp.ffn_hidden_size);
    let head_dim = h / hp.n_heads; // 4
    let kv_dim = head_dim * hp.n_kv_heads; // 4
    let mut t = vec![
        ("token_embd.weight".into(), vec![v, h]),
        ("output_norm.weight".into(), vec![h]),
        ("output.weight".into(), vec![v, h]),
    ];
    for i in 0..hp.n_layers {
        for (suffix, shape) in [
            ("attn_norm.weight", vec![h]),
            ("attn_q.weight", vec![h, h]),
            ("attn_k.weight", vec![kv_dim, h]),
            ("attn_v.weight", vec![kv_dim, h]),
            ("attn_output.weight", vec![h, h]),
            ("ffn_norm.weight", vec![h]),
            ("ffn_gate.weight", vec![f, h]),
            ("ffn_up.weight", vec![f, h]),
            ("ffn_down.weight", vec![h, f]),
        ] {
            t.push((format!("blk.{i}.{suffix}"), shape));
        }
    }
    t
}

/// A complete, valid GGUF v3 file (F32 tensors, data zero-filled).
pub fn tiny_llama_gguf() -> Vec<u8> {
    let hp = tiny_hyperparams();
    let tensors = tiny_tensor_shapes();
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes()); // version
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&10u64.to_le_bytes()); // kv count — keep in sync below!

    put_kv_str(&mut out, "general.architecture", "llama");
    put_kv_str(&mut out, "general.name", "tiny-llama-test");
    put_kv_u32(&mut out, "general.alignment", 32);
    put_kv_u32(&mut out, "llama.block_count", hp.n_layers as u32);
    put_kv_u32(&mut out, "llama.embedding_length", hp.hidden_size as u32);
    put_kv_u32(&mut out, "llama.attention.head_count", hp.n_heads as u32);
    put_kv_u32(
        &mut out,
        "llama.attention.head_count_kv",
        hp.n_kv_heads as u32,
    );
    put_kv_u32(
        &mut out,
        "llama.feed_forward_length",
        hp.ffn_hidden_size as u32,
    );
    put_kv_u32(&mut out, "llama.context_length", hp.context_length as u32);
    put_kv_f32(
        &mut out,
        "llama.attention.layer_norm_rms_epsilon",
        hp.norm_eps,
    );
    // vocab_size key deliberately omitted: exercises the token_embd fallback.

    // Tensor infos. Offsets are relative to the (32-aligned) data section.
    let mut offset = 0u64;
    for (name, shape) in &tensors {
        put_str(&mut out, name);
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in shape.iter().rev() {
            // fastest-first on disk
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes()); // ggml type 0 = F32
        out.extend_from_slice(&offset.to_le_bytes());
        let n: u64 = shape.iter().product();
        offset += (n * 4).next_multiple_of(32);
    }

    // Data section: align, then zero-fill.
    while out.len() % 32 != 0 {
        out.push(0);
    }
    out.resize(out.len() + offset as usize, 0);
    out
}

/// The tiny llama as a single MLX-style safetensors file (F32, zero data).
pub fn tiny_llama_safetensors() -> Vec<u8> {
    let mut entries = Vec::new();
    let mut offset = 0u64;
    for (name, shape) in tiny_tensor_shapes() {
        // HF/MLX naming differs from GGUF naming; that mapping is M1's
        // problem (graph builder). M0 records names verbatim.
        let n: u64 = shape.iter().product();
        let end = offset + n * 4;
        entries.push(format!(
            r#""{name}": {{"dtype":"F32","shape":[{}],"data_offsets":[{offset},{end}]}}"#,
            shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ));
        offset = end;
    }
    let json = format!("{{{}}}", entries.join(","));
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(json.as_bytes());
    out.resize(out.len() + offset as usize, 0);
    out
}

/// Matching MLX config.json (HF-style keys).
pub fn tiny_llama_config_json() -> String {
    let hp = tiny_hyperparams();
    format!(
        r#"{{
  "model_type": "llama",
  "hidden_size": {},
  "num_hidden_layers": {},
  "num_attention_heads": {},
  "num_key_value_heads": {},
  "intermediate_size": {},
  "vocab_size": {},
  "rope_theta": {},
  "rms_norm_eps": {},
  "max_position_embeddings": {}
}}"#,
        hp.hidden_size,
        hp.n_layers,
        hp.n_heads,
        hp.n_kv_heads,
        hp.ffn_hidden_size,
        hp.vocab_size,
        hp.rope_theta,
        hp.norm_eps,
        hp.context_length
    )
}
