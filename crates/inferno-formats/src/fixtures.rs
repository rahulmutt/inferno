//! Tiny in-memory models for tests, fuzz corpus seeds, and CLI snapshots.
//! Also consumed by later milestones (M1 interpreter tests). Not a public
//! stability surface.

use crate::{DType, HyperParams, quant};

pub fn tiny_hyperparams() -> HyperParams {
    HyperParams {
        vocab_size: 260,
        hidden_size: 64,
        n_layers: 2,
        n_heads: 2,
        n_kv_heads: 1,
        ffn_hidden_size: 256,
        rope_theta: 10000.0,
        norm_eps: 1e-5,
        context_length: 128,
        rope_style: crate::RopeStyle::Interleaved,
    }
}

pub(crate) fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

pub(crate) fn put_kv_u32(out: &mut Vec<u8>, key: &str, v: u32) {
    put_str(out, key);
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn put_kv_f32(out: &mut Vec<u8>, key: &str, v: f32) {
    put_str(out, key);
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&v.to_le_bytes());
}

pub(crate) fn put_kv_str(out: &mut Vec<u8>, key: &str, v: &str) {
    put_str(out, key);
    out.extend_from_slice(&8u32.to_le_bytes());
    put_str(out, v);
}

pub(crate) fn put_kv_str_array(out: &mut Vec<u8>, key: &str, items: &[String]) {
    put_str(out, key);
    out.extend_from_slice(&9u32.to_le_bytes()); // array
    out.extend_from_slice(&8u32.to_le_bytes()); // elem: string
    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for s in items {
        put_str(out, s);
    }
}

pub(crate) fn put_kv_i32_array(out: &mut Vec<u8>, key: &str, items: &[i32]) {
    put_str(out, key);
    out.extend_from_slice(&9u32.to_le_bytes());
    out.extend_from_slice(&5u32.to_le_bytes()); // elem: i32
    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for v in items {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

pub(crate) fn put_kv_bool(out: &mut Vec<u8>, key: &str, v: bool) {
    put_str(out, key);
    out.extend_from_slice(&7u32.to_le_bytes());
    out.push(u8::from(v));
}

/// Deterministic xorshift64* stream; weights in [-0.125, 0.125).
fn weight_stream(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((r >> 40) as f32 / 16_777_216.0 - 0.5) * 0.25
        })
        .collect()
}

/// HF half-split → GGML interleaved row order for rope'd projections
/// (convert_hf_to_gguf.py LlamaModel.permute): within each head, source row
/// s*half+j2 (s ∈ {0,1}) moves to row 2*j2+s.
fn permute_rows(w: &[f32], rows: usize, cols: usize, n_head: usize) -> Vec<f32> {
    let hd = rows / n_head;
    let half = hd / 2;
    let mut out = vec![0.0; w.len()];
    for h in 0..n_head {
        for j in 0..hd {
            let dst = h * hd + if j < half { 2 * j } else { 2 * (j - half) + 1 };
            let src = h * hd + j;
            out[dst * cols..(dst + 1) * cols].copy_from_slice(&w[src * cols..(src + 1) * cols]);
        }
    }
    out
}

pub struct FixtureTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

/// Table: (gguf name, hf name, shape, gguf dtype, permute heads (0 = no)).
fn tensor_table() -> Vec<(String, String, Vec<u64>, DType, usize)> {
    let hp = tiny_hyperparams();
    let (v, h, f) = (hp.vocab_size, hp.hidden_size, hp.ffn_hidden_size);
    let kv = h / hp.n_heads * hp.n_kv_heads; // 32
    let mut t: Vec<(String, String, Vec<u64>, DType, usize)> = vec![
        (
            "token_embd.weight".into(),
            "model.embed_tokens.weight".into(),
            vec![v, h],
            DType::F32,
            0,
        ),
        (
            "output_norm.weight".into(),
            "model.norm.weight".into(),
            vec![h],
            DType::F32,
            0,
        ),
        // NOTE: no output.weight / lm_head.weight — embeddings are tied.
    ];
    for i in 0..hp.n_layers {
        let g = |s: &str| format!("blk.{i}.{s}");
        let m = |s: &str| format!("model.layers.{i}.{s}");
        t.extend([
            (
                g("attn_norm.weight"),
                m("input_layernorm.weight"),
                vec![h],
                DType::F32,
                0,
            ),
            (
                g("attn_q.weight"),
                m("self_attn.q_proj.weight"),
                vec![h, h],
                DType::Q8_0,
                hp.n_heads as usize,
            ),
            (
                g("attn_k.weight"),
                m("self_attn.k_proj.weight"),
                vec![kv, h],
                DType::Q8_0,
                hp.n_kv_heads as usize,
            ),
            (
                g("attn_v.weight"),
                m("self_attn.v_proj.weight"),
                vec![kv, h],
                DType::F16,
                0,
            ),
            (
                g("attn_output.weight"),
                m("self_attn.o_proj.weight"),
                vec![h, h],
                DType::Q8_0,
                0,
            ),
            (
                g("ffn_norm.weight"),
                m("post_attention_layernorm.weight"),
                vec![h],
                DType::F32,
                0,
            ),
            (
                g("ffn_gate.weight"),
                m("mlp.gate_proj.weight"),
                vec![f, h],
                DType::F16,
                0,
            ),
            (
                g("ffn_up.weight"),
                m("mlp.up_proj.weight"),
                vec![f, h],
                DType::BF16,
                0,
            ),
            (
                g("ffn_down.weight"),
                m("mlp.down_proj.weight"),
                vec![h, f],
                DType::Q4_K,
                0,
            ),
        ]);
    }
    t
}

/// GGUF-side tensors: packed in `dtype`, Q/K rows permuted (Interleaved rope).
pub fn tiny_tensors_gguf() -> Vec<FixtureTensor> {
    tensor_table()
        .into_iter()
        .enumerate()
        .map(|(seed, (gname, _, shape, dtype, permute_heads))| {
            let n: usize = shape.iter().product::<u64>() as usize;
            let mut w = weight_stream(0xF17E + seed as u64, n);
            if permute_heads > 0 {
                let cols = *shape.last().unwrap() as usize;
                w = permute_rows(&w, n / cols, cols, permute_heads);
            }
            let data = quant::pack(&dtype, &w).unwrap();
            FixtureTensor {
                name: gname,
                shape,
                dtype,
                data,
            }
        })
        .collect()
}

/// MLX-side tensors: same effective values, HF names, unpermuted, quantized
/// dtypes materialized as F32 (safetensors has no Q8_0/Q4_K).
pub fn tiny_tensors_hf() -> Vec<FixtureTensor> {
    tensor_table()
        .into_iter()
        .enumerate()
        .map(|(seed, (_, hname, shape, dtype, _))| {
            let n: usize = shape.iter().product::<u64>() as usize;
            let w = weight_stream(0xF17E + seed as u64, n);
            // Effective value = dequant(pack(w)); per-row blocks make this
            // independent of the GGUF-side row permutation.
            let eff = quant::dequant(&dtype, &quant::pack(&dtype, &w).unwrap(), n).unwrap();
            let (dtype, data) = match dtype {
                DType::F16 | DType::BF16 => (dtype.clone(), quant::pack(&dtype, &eff).unwrap()),
                _ => (DType::F32, quant::pack(&DType::F32, &eff).unwrap()),
            };
            FixtureTensor {
                name: hname,
                shape,
                dtype,
                data,
            }
        })
        .collect()
}

/// GPT-2 byte↔unicode table (duplicated in inferno-runtime's BPE tokenizer;
/// kept private here — fixtures are not a stability surface).
fn byte_unicode(b: u8) -> char {
    let printable = (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
    if printable {
        char::from_u32(u32::from(b)).unwrap()
    } else {
        // Non-printables map to 256+n in first-seen order, matching GPT-2.
        let mut n = 0;
        for x in 0u16..u16::from(b) {
            let x8 = x as u8;
            let p = (b'!'..=b'~').contains(&x8) || (0xA1..=0xAC).contains(&x8) || x8 >= 0xAE;
            if x < 256 && !p {
                n += 1;
            }
        }
        char::from_u32(256 + n).unwrap()
    }
}

/// (tokens, merges): 256 byte tokens, <|bos|>=256, <|eos|>=257, "th"=258, "the"=259.
pub fn tiny_vocab() -> (Vec<String>, Vec<String>) {
    let mut tokens: Vec<String> = (0u16..256)
        .map(|b| byte_unicode(b as u8).to_string())
        .collect();
    tokens.push("<|bos|>".into());
    tokens.push("<|eos|>".into());
    tokens.push("th".into());
    tokens.push("the".into());
    (tokens, vec!["t h".into(), "th e".into()])
}

fn ggml_dtype_id(d: &DType) -> u32 {
    match d {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::Q8_0 => 8,
        DType::Q4_K => 12,
        DType::BF16 => 30,
        DType::Unsupported(_) => unreachable!("fixtures use supported dtypes"),
    }
}

/// The tiny-llama KV metadata block (architecture, hyperparams, tokenizer).
/// Each entry is one serialized KV pair; the header count is `kvs.len()` by
/// construction, so it can never drift out of sync.
fn tiny_kvs() -> Vec<Vec<u8>> {
    let hp = tiny_hyperparams();
    let (tokens, merges) = tiny_vocab();
    let mut token_types = vec![1i32; 256];
    token_types.extend([3, 3, 1, 1]); // bos/eos control, merged tokens normal
    let one = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = Vec::new();
        f(&mut b);
        b
    };
    vec![
        one(&|o| put_kv_str(o, "general.architecture", "llama")),
        one(&|o| put_kv_str(o, "general.name", "tiny-llama-test")),
        one(&|o| put_kv_u32(o, "general.alignment", 32)),
        one(&|o| put_kv_u32(o, "llama.block_count", hp.n_layers as u32)),
        one(&|o| put_kv_u32(o, "llama.embedding_length", hp.hidden_size as u32)),
        one(&|o| put_kv_u32(o, "llama.attention.head_count", hp.n_heads as u32)),
        one(&|o| put_kv_u32(o, "llama.attention.head_count_kv", hp.n_kv_heads as u32)),
        one(&|o| put_kv_u32(o, "llama.feed_forward_length", hp.ffn_hidden_size as u32)),
        one(&|o| put_kv_u32(o, "llama.context_length", hp.context_length as u32)),
        one(&|o| put_kv_f32(o, "llama.attention.layer_norm_rms_epsilon", hp.norm_eps)),
        one(&|o| put_kv_str(o, "tokenizer.ggml.model", "gpt2")),
        one(&|o| put_kv_str(o, "tokenizer.ggml.pre", "default")),
        one(&|o| put_kv_str_array(o, "tokenizer.ggml.tokens", &tokens)),
        one(&|o| put_kv_str_array(o, "tokenizer.ggml.merges", &merges)),
        one(&|o| put_kv_i32_array(o, "tokenizer.ggml.token_type", &token_types)),
        one(&|o| put_kv_u32(o, "tokenizer.ggml.bos_token_id", 256)),
        one(&|o| put_kv_u32(o, "tokenizer.ggml.eos_token_id", 257)),
        one(&|o| put_kv_bool(o, "tokenizer.ggml.add_bos_token", false)),
    ]
}

/// Serialize a GGUF v3 image from `tensors` + a pre-built `kvs` block. Tensor
/// data lives in a 32-aligned section; each info's offset is relative to it.
fn assemble_gguf(tensors: &[FixtureTensor], kvs: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for kv in kvs {
        out.extend_from_slice(kv);
    }

    let mut offset = 0u64;
    for t in tensors {
        put_str(&mut out, &t.name);
        out.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
        for d in t.shape.iter().rev() {
            out.extend_from_slice(&d.to_le_bytes()); // fastest-first on disk
        }
        out.extend_from_slice(&ggml_dtype_id(&t.dtype).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += (t.data.len() as u64).next_multiple_of(32);
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    for t in tensors {
        out.extend_from_slice(&t.data);
        while out.len() % 32 != 0 {
            out.push(0);
        }
    }
    out
}

pub fn tiny_llama_gguf() -> Vec<u8> {
    assemble_gguf(&tiny_tensors_gguf(), &tiny_kvs())
}

/// Like `tiny_llama_gguf()`, but every attention projection also carries a
/// q/k/v bias (as Qwen2/Qwen2.5 attention does). `build_graph` picks these up
/// (`blk.{i}.attn_q.bias` → `layers.{i}.attn.q_proj.bias`, etc.) and emits
/// `Op::MatMul { bias: Some(_) }`, so this fixture puts the compiled
/// `Step::Bias` lowering under the compiled-vs-interpreter differential gate.
/// Biases are plain F32; the differential compares the SAME GGUF through both
/// paths, so no rope row-permutation of the bias is needed here.
pub fn tiny_bias_llama_gguf() -> Vec<u8> {
    let hp = tiny_hyperparams();
    let kv = hp.hidden_size / hp.n_heads * hp.n_kv_heads; // 32 (kv_dim)
    let mut tensors = tiny_tensors_gguf();
    for i in 0..hp.n_layers {
        let g = |s: &str| format!("blk.{i}.{s}");
        // q bias spans all heads (h); k/v biases span the kv projection (kv_dim).
        for (name, rows, seed) in [
            (g("attn_q.bias"), hp.hidden_size, 0xB1A5u64),
            (g("attn_k.bias"), kv, 0xB1A6),
            (g("attn_v.bias"), kv, 0xB1A7),
        ] {
            let n = rows as usize;
            tensors.push(FixtureTensor {
                name,
                shape: vec![rows],
                dtype: DType::F32,
                data: quant::pack(&DType::F32, &weight_stream(seed + i, n)).unwrap(),
            });
        }
    }
    assemble_gguf(&tensors, &tiny_kvs())
}

/// A hostile-but-structurally-valid model: `vocab_size = 1`, a single layer,
/// and a 1-token tokenizer. `build_graph` only rejects `vocab_size == 0`
/// (spec's allocation guard), and a 1-token BPE tokenizer parses cleanly, so
/// this is accepted all the way through `Generator::load` — exactly the
/// hostile input the diff harness's `top[1]` indexing must reject with a
/// typed error instead of panicking (there is no top-2 with 1 token).
pub fn hostile_vocab1_gguf() -> Vec<u8> {
    let h: u64 = 2; // hidden_size: smallest even value (rope needs head_dim even)
    let v: u64 = 1; // vocab_size: the hostile part
    let f: u64 = 2; // ffn_hidden_size
    let n_layers: u64 = 1;

    let named = |name: &str, shape: Vec<u64>, seed: u64| -> FixtureTensor {
        let n: usize = shape.iter().product::<u64>() as usize;
        FixtureTensor {
            name: name.into(),
            shape,
            dtype: DType::F32,
            data: quant::pack(&DType::F32, &weight_stream(seed, n)).unwrap(),
        }
    };

    let mut tensors = vec![
        named("token_embd.weight", vec![v, h], 1),
        named("output_norm.weight", vec![h], 2),
    ];
    for i in 0..n_layers {
        let g = |s: &str| format!("blk.{i}.{s}");
        tensors.extend([
            named(&g("attn_norm.weight"), vec![h], 10),
            named(&g("attn_q.weight"), vec![h, h], 11),
            named(&g("attn_k.weight"), vec![h, h], 12),
            named(&g("attn_v.weight"), vec![h, h], 13),
            named(&g("attn_output.weight"), vec![h, h], 14),
            named(&g("ffn_norm.weight"), vec![h], 15),
            named(&g("ffn_gate.weight"), vec![f, h], 16),
            named(&g("ffn_up.weight"), vec![f, h], 17),
            named(&g("ffn_down.weight"), vec![h, f], 18),
        ]);
    }

    let one = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = Vec::new();
        f(&mut b);
        b
    };
    let tokens = vec!["a".to_string()];
    let token_types = vec![1i32];
    let kvs: Vec<Vec<u8>> = vec![
        one(&|o| put_kv_str(o, "general.architecture", "llama")),
        one(&|o| put_kv_str(o, "general.name", "hostile-vocab1-test")),
        one(&|o| put_kv_u32(o, "general.alignment", 32)),
        one(&|o| put_kv_u32(o, "llama.block_count", n_layers as u32)),
        one(&|o| put_kv_u32(o, "llama.embedding_length", h as u32)),
        one(&|o| put_kv_u32(o, "llama.attention.head_count", 1)),
        one(&|o| put_kv_u32(o, "llama.attention.head_count_kv", 1)),
        one(&|o| put_kv_u32(o, "llama.feed_forward_length", f as u32)),
        one(&|o| put_kv_u32(o, "llama.context_length", 8)),
        one(&|o| put_kv_f32(o, "llama.attention.layer_norm_rms_epsilon", 1e-5)),
        one(&|o| put_kv_str(o, "tokenizer.ggml.model", "gpt2")),
        one(&|o| put_kv_str(o, "tokenizer.ggml.pre", "default")),
        one(&|o| put_kv_str_array(o, "tokenizer.ggml.tokens", &tokens)),
        one(&|o| put_kv_str_array(o, "tokenizer.ggml.merges", &[])),
        one(&|o| put_kv_i32_array(o, "tokenizer.ggml.token_type", &token_types)),
        one(&|o| put_kv_bool(o, "tokenizer.ggml.add_bos_token", false)),
    ];

    assemble_gguf(&tensors, &kvs)
}

/// The tiny llama as a single MLX-style safetensors file: same effective
/// weights as `tiny_llama_gguf()`, HF names, unpermuted, tied embeddings.
pub fn tiny_llama_safetensors() -> Vec<u8> {
    let tensors = tiny_tensors_hf();
    let mut entries = Vec::new();
    let mut offset = 0u64;
    for t in &tensors {
        let end = offset + t.data.len() as u64;
        let dtype = match t.dtype {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            _ => unreachable!("hf fixture tensors are float dtypes"),
        };
        entries.push(format!(
            r#""{}": {{"dtype":"{dtype}","shape":[{}],"data_offsets":[{offset},{end}]}}"#,
            t.name,
            t.shape
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
    for t in &tensors {
        out.extend_from_slice(&t.data);
    }
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

/// HF tokenizer.json equivalent of the embedded GGUF vocab (ByteLevel BPE).
pub fn tiny_tokenizer_json() -> String {
    let (tokens, merges) = tiny_vocab();
    let vocab: Vec<String> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| format!(r#""{}": {i}"#, t.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect();
    let merges: Vec<String> = merges.iter().map(|m| format!(r#""{m}""#)).collect();
    format!(
        r#"{{
  "version": "1.0",
  "added_tokens": [
    {{"id": 256, "content": "<|bos|>", "single_word": false, "lstrip": false,
      "rstrip": false, "normalized": false, "special": true}},
    {{"id": 257, "content": "<|eos|>", "single_word": false, "lstrip": false,
      "rstrip": false, "normalized": false, "special": true}}
  ],
  "normalizer": null,
  "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}},
  "post_processor": null,
  "decoder": {{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}},
  "model": {{
    "type": "BPE",
    "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
    "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
    "vocab": {{{vocab}}},
    "merges": [{merges}]
  }}
}}"#,
        vocab = vocab.join(", "),
        merges = merges.join(", ")
    )
}

#[cfg(test)]
mod task5_tests {
    use super::*;
    use crate::{DType, load_desc, quant};
    use std::io::Cursor;

    #[test]
    fn gguf_fixture_is_tied_quantized_and_tokenized() {
        let desc = crate::gguf::parse(&mut Cursor::new(&tiny_llama_gguf())).unwrap();
        assert!(desc.tensors.iter().all(|t| t.name != "lm_head.weight")); // tied
        let down = desc
            .tensors
            .iter()
            .find(|t| t.name == "layers.0.ffn.down_proj.weight")
            .unwrap();
        assert_eq!(down.dtype, DType::Q4_K);
        assert!(desc.tokenizer.is_some());
    }

    #[test]
    fn gguf_and_mlx_effective_weights_match() {
        // Same value stream: GGUF stores packed (and Q/K-permuted) weights,
        // MLX stores the dequantized (unpermuted) values. Dequantizing the
        // GGUF v_proj (F16, never permuted) must equal the MLX v_proj bytes.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiny.gguf"), tiny_llama_gguf()).unwrap();
        std::fs::write(dir.path().join("config.json"), tiny_llama_config_json()).unwrap();
        std::fs::write(
            dir.path().join("model.safetensors"),
            tiny_llama_safetensors(),
        )
        .unwrap();
        let g = load_desc(&dir.path().join("tiny.gguf")).unwrap();
        let m = load_desc(dir.path()).unwrap();
        for name in ["layers.0.attn.v_proj.weight", "layers.1.ffn.up_proj.weight"] {
            let gt = g.tensors.iter().find(|t| t.name == name).unwrap();
            let mt = m.tensors.iter().find(|t| t.name == name).unwrap();
            let gv = quant::dequant(
                &gt.dtype,
                &crate::read_tensor_bytes(&g, gt).unwrap(),
                gt.shape.iter().product::<u64>() as usize,
            )
            .unwrap();
            let mv = quant::dequant(
                &mt.dtype,
                &crate::read_tensor_bytes(&m, mt).unwrap(),
                mt.shape.iter().product::<u64>() as usize,
            )
            .unwrap();
            assert_eq!(gv, mv, "{name}");
        }
    }

    #[test]
    fn hostile_vocab1_gguf_parses_with_vocab_size_one() {
        // Regression guard for the inferno-runtime diff harness panic: this
        // fixture must remain parseable (and vocab_size must stay 1) so the
        // downstream typed-error guard has something hostile to reject.
        let desc = crate::gguf::parse(&mut Cursor::new(&hostile_vocab1_gguf())).unwrap();
        assert_eq!(desc.hyperparams.vocab_size, 1);
        assert!(desc.tokenizer.is_some());
    }

    #[test]
    fn weights_are_not_degenerate() {
        let desc = crate::gguf::parse(&mut Cursor::new(&tiny_llama_gguf())).unwrap();
        let embd = desc
            .tensors
            .iter()
            .find(|t| t.name == "token_embed.weight")
            .unwrap();
        // Data written into the in-memory image, non-zero and deterministic.
        let bytes = tiny_llama_gguf();
        let start = desc.data_section_offsets[0] + embd.data_offset;
        let b = &bytes[start as usize..(start + 16) as usize];
        assert_ne!(b, &[0u8; 16]);
    }
}
