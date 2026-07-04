use inferno_formats::{Architecture, ModelDesc};

fn arch_label(a: &Architecture) -> String {
    match a {
        Architecture::Llama => "llama".into(),
        Architecture::Qwen2 => "qwen2".into(),
        Architecture::Qwen3 => "qwen3".into(),
        Architecture::Mistral => "mistral".into(),
        Architecture::Unknown(s) => format!("unknown ({s})"),
    }
}

pub fn render(desc: &ModelDesc, max_tensors: usize) -> String {
    let hp = &desc.hyperparams;
    let mut out = String::new();
    if let Some(name) = &desc.name {
        out.push_str(&format!("model: {name}\n"));
    }
    out.push_str(&format!(
        "architecture: {}\n",
        arch_label(&desc.architecture)
    ));
    out.push_str(&format!(
        "hyperparams: layers={} hidden={} heads={} kv_heads={} ffn={} vocab={} ctx={} rope_theta={} norm_eps={}\n",
        hp.n_layers, hp.hidden_size, hp.n_heads, hp.n_kv_heads,
        hp.ffn_hidden_size, hp.vocab_size, hp.context_length, hp.rope_theta, hp.norm_eps,
    ));
    out.push_str(&format!("tensors: {}\n", desc.tensors.len()));
    for t in desc.tensors.iter().take(max_tensors) {
        let shape = t
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join("x");
        out.push_str(&format!("  {:<40} {:>12} {:?}\n", t.name, shape, t.dtype));
    }
    if desc.tensors.len() > max_tensors && max_tensors > 0 {
        out.push_str(&format!(
            "  … and {} more\n",
            desc.tensors.len() - max_tensors
        ));
    }
    out
}
