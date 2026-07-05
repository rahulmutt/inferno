//! Canonical tensor names. Parsers map format-specific names to one scheme at
//! the edge so nothing downstream can tell which file format a model came
//! from (ARCHITECTURE.md boundary rule). Unmapped names pass through raw.

/// GGUF suffix (after `blk.{i}.`) → canonical suffix (after `layers.{i}.`).
const GGUF_LAYER: &[(&str, &str)] = &[
    ("attn_norm", "attn_norm"),
    ("attn_q_norm", "attn.q_norm"),
    ("attn_k_norm", "attn.k_norm"),
    ("attn_q", "attn.q_proj"),
    ("attn_k", "attn.k_proj"),
    ("attn_v", "attn.v_proj"),
    ("attn_output", "attn.o_proj"),
    ("ffn_norm", "ffn_norm"),
    ("ffn_gate", "ffn.gate_proj"),
    ("ffn_up", "ffn.up_proj"),
    ("ffn_down", "ffn.down_proj"),
];

/// HF infix (after `model.layers.{i}.`) → canonical suffix.
const HF_LAYER: &[(&str, &str)] = &[
    ("input_layernorm", "attn_norm"),
    ("self_attn.q_norm", "attn.q_norm"),
    ("self_attn.k_norm", "attn.k_norm"),
    ("self_attn.q_proj", "attn.q_proj"),
    ("self_attn.k_proj", "attn.k_proj"),
    ("self_attn.v_proj", "attn.v_proj"),
    ("self_attn.o_proj", "attn.o_proj"),
    ("post_attention_layernorm", "ffn_norm"),
    ("mlp.gate_proj", "ffn.gate_proj"),
    ("mlp.up_proj", "ffn.up_proj"),
    ("mlp.down_proj", "ffn.down_proj"),
];

/// Split "name.weight" / "name.bias" → (name, param). `map_layer` below does
/// exact-match lookup against the tables above, so their ordering is
/// irrelevant (e.g. "attn_q" and "attn_q_norm" are distinct exact keys, not
/// competing prefixes).
fn split_param(raw: &str) -> Option<(&str, &str)> {
    raw.rsplit_once('.')
        .filter(|(_, p)| *p == "weight" || *p == "bias")
}

fn map_layer(table: &[(&str, &'static str)], stem: &str) -> Option<&'static str> {
    table
        .iter()
        .find(|(from, _)| *from == stem)
        .map(|(_, to)| *to)
}

pub(crate) fn canonical_gguf(raw: &str) -> Option<String> {
    let (stem, param) = split_param(raw)?;
    match stem {
        "token_embd" => return Some(format!("token_embed.{param}")),
        "output" => return Some(format!("lm_head.{param}")),
        "output_norm" => return Some(format!("output_norm.{param}")),
        _ => {}
    }
    let rest = stem.strip_prefix("blk.")?;
    let (idx, suffix) = rest.split_once('.')?;
    let idx: u64 = idx.parse().ok()?;
    let mapped = map_layer(GGUF_LAYER, suffix)?;
    Some(format!("layers.{idx}.{mapped}.{param}"))
}

pub(crate) fn canonical_hf(raw: &str) -> Option<String> {
    let (stem, param) = split_param(raw)?;
    match stem {
        "model.embed_tokens" => return Some(format!("token_embed.{param}")),
        "model.norm" => return Some(format!("output_norm.{param}")),
        "lm_head" => return Some(format!("lm_head.{param}")),
        _ => {}
    }
    let rest = stem.strip_prefix("model.layers.")?;
    let (idx, suffix) = rest.split_once('.')?;
    let idx: u64 = idx.parse().ok()?;
    let mapped = map_layer(HF_LAYER, suffix)?;
    Some(format!("layers.{idx}.{mapped}.{param}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_names_map() {
        assert_eq!(
            canonical_gguf("token_embd.weight").as_deref(),
            Some("token_embed.weight")
        );
        assert_eq!(
            canonical_gguf("output.weight").as_deref(),
            Some("lm_head.weight")
        );
        assert_eq!(
            canonical_gguf("output_norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            canonical_gguf("blk.3.attn_q.weight").as_deref(),
            Some("layers.3.attn.q_proj.weight")
        );
        assert_eq!(
            canonical_gguf("blk.0.attn_k.bias").as_deref(),
            Some("layers.0.attn.k_proj.bias")
        );
        assert_eq!(
            canonical_gguf("blk.12.ffn_down.weight").as_deref(),
            Some("layers.12.ffn.down_proj.weight")
        );
        assert_eq!(
            canonical_gguf("blk.0.attn_q_norm.weight").as_deref(),
            Some("layers.0.attn.q_norm.weight")
        );
        assert_eq!(canonical_gguf("rope_freqs.weight"), None); // unmapped → raw
    }

    #[test]
    fn hf_names_map() {
        assert_eq!(
            canonical_hf("model.embed_tokens.weight").as_deref(),
            Some("token_embed.weight")
        );
        assert_eq!(
            canonical_hf("model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            canonical_hf("lm_head.weight").as_deref(),
            Some("lm_head.weight")
        );
        assert_eq!(
            canonical_hf("model.layers.3.self_attn.q_proj.weight").as_deref(),
            Some("layers.3.attn.q_proj.weight")
        );
        assert_eq!(
            canonical_hf("model.layers.0.input_layernorm.weight").as_deref(),
            Some("layers.0.attn_norm.weight")
        );
        assert_eq!(
            canonical_hf("model.layers.0.post_attention_layernorm.weight").as_deref(),
            Some("layers.0.ffn_norm.weight")
        );
        assert_eq!(
            canonical_hf("model.layers.1.mlp.gate_proj.weight").as_deref(),
            Some("layers.1.ffn.gate_proj.weight")
        );
        assert_eq!(
            canonical_hf("model.layers.1.self_attn.q_norm.weight").as_deref(),
            Some("layers.1.attn.q_norm.weight")
        );
        assert_eq!(canonical_hf("model.rotary_emb.inv_freq"), None);
    }
}
