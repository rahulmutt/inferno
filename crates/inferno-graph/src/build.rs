//! ModelDesc → Graph. One data-driven builder covers the Llama family;
//! presence/absence of canonical tensors drives structure (biases, q/k
//! norms, tied embeddings). Everything reachable from a model file is a
//! typed error, never a panic.

use std::collections::HashMap;

use inferno_formats::{Architecture, ModelDesc};

use crate::ir::{Dim, Graph, Node, Op, Shape, TensorRef, ValueId};
use crate::{GraphError, Result};

/// Allocation guard for hostile hyperparams (spec §Error handling): caps the
/// largest single dimension a model file can request the interpreter allocate.
const MAX_DIM: u64 = 1 << 20;

struct Builder<'d> {
    desc: &'d ModelDesc,
    by_name: HashMap<&'d str, usize>,
    nodes: Vec<Node>,
}

impl<'d> Builder<'d> {
    fn get(&self, name: &str) -> Option<TensorRef> {
        self.by_name.get(name).copied().map(TensorRef)
    }

    fn require(&self, name: &str, expected: &[u64]) -> Result<TensorRef> {
        let r = self
            .get(name)
            .ok_or_else(|| GraphError::MissingTensor(name.into()))?;
        let got = &self.desc.tensors[r.0].shape;
        if got != expected {
            return Err(GraphError::ShapeMismatch {
                name: name.into(),
                expected: expected.to_vec(),
                got: got.clone(),
            });
        }
        Ok(r)
    }

    /// Optional tensor: absent → None, present with wrong shape → error.
    fn optional(&self, name: &str, expected: &[u64]) -> Result<Option<TensorRef>> {
        match self.get(name) {
            None => Ok(None),
            Some(_) => self.require(name, expected).map(Some),
        }
    }

    fn push(&mut self, op: Op, inputs: Vec<ValueId>, out_shape: Shape, label: String) -> ValueId {
        self.nodes.push(Node {
            op,
            inputs,
            out_shape,
            label,
        });
        self.nodes.len() // node i outputs value i+1; value 0 is the tokens input
    }
}

pub fn build_graph(desc: &ModelDesc) -> Result<Graph> {
    if let Architecture::Unknown(id) = &desc.architecture {
        return Err(GraphError::UnsupportedArch(id.clone()));
    }
    let hp = &desc.hyperparams;
    let bad = |msg: String| Err(GraphError::BadHyperParams(msg));
    if hp.n_heads == 0 || hp.n_kv_heads == 0 || hp.hidden_size == 0 || hp.n_layers == 0 {
        return bad("zero-valued hyperparameter".into());
    }
    if !hp.hidden_size.is_multiple_of(hp.n_heads) {
        return bad(format!(
            "hidden {} not divisible by heads {}",
            hp.hidden_size, hp.n_heads
        ));
    }
    if !hp.n_heads.is_multiple_of(hp.n_kv_heads) {
        return bad(format!(
            "heads {} not divisible by kv heads {}",
            hp.n_heads, hp.n_kv_heads
        ));
    }
    let head_dim = hp.hidden_size / hp.n_heads;
    if !head_dim.is_multiple_of(2) {
        return bad(format!("head_dim {head_dim} must be even for rope"));
    }
    for (what, v) in [
        ("hidden_size", hp.hidden_size),
        ("ffn_hidden_size", hp.ffn_hidden_size),
        ("vocab_size", hp.vocab_size),
        ("n_layers", hp.n_layers),
    ] {
        if v == 0 || v > MAX_DIM {
            return bad(format!("{what} = {v} outside 1..={MAX_DIM}"));
        }
    }

    let (h, f, v) = (hp.hidden_size, hp.ffn_hidden_size, hp.vocab_size);
    let kv_dim = head_dim * hp.n_kv_heads;
    let mut b = Builder {
        desc,
        by_name: desc
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.as_str(), i))
            .collect(),
        nodes: Vec::new(),
    };

    let seq_h = Shape(vec![Dim::Seq, Dim::Const(h)]);
    let embed_w = b.require("token_embed.weight", &[v, h])?;
    let mut x = b.push(
        Op::Embed { weight: embed_w },
        vec![0],
        seq_h.clone(),
        "embed".into(),
    );

    for i in 0..hp.n_layers {
        let l = |s: &str| format!("layers.{i}.{s}");
        let li = i as usize;

        let norm_w = b.require(&l("attn_norm.weight"), &[h])?;
        let hn = b.push(
            Op::RmsNorm {
                weight: norm_w,
                eps: hp.norm_eps,
                head_dim: None,
            },
            vec![x],
            seq_h.clone(),
            l("attn_norm"),
        );
        let proj = |b: &mut Builder, name: &str, rows: u64| -> Result<ValueId> {
            let w = b.require(&l(&format!("attn.{name}.weight")), &[rows, h])?;
            let bias = b.optional(&l(&format!("attn.{name}.bias")), &[rows])?;
            Ok(b.push(
                Op::MatMul { weight: w, bias },
                vec![hn],
                Shape(vec![Dim::Seq, Dim::Const(rows)]),
                l(&format!("attn.{name}")),
            ))
        };
        let mut q = proj(&mut b, "q_proj", h)?;
        let mut k = proj(&mut b, "k_proj", kv_dim)?;
        let vv = proj(&mut b, "v_proj", kv_dim)?;

        // Qwen3 per-head q/k rmsnorm, before rope.
        if let Some(qn) = b.optional(&l("attn.q_norm.weight"), &[head_dim])? {
            q = b.push(
                Op::RmsNorm {
                    weight: qn,
                    eps: hp.norm_eps,
                    head_dim: Some(head_dim),
                },
                vec![q],
                Shape(vec![Dim::Seq, Dim::Const(h)]),
                l("attn.q_norm"),
            );
        }
        if let Some(kn) = b.optional(&l("attn.k_norm.weight"), &[head_dim])? {
            k = b.push(
                Op::RmsNorm {
                    weight: kn,
                    eps: hp.norm_eps,
                    head_dim: Some(head_dim),
                },
                vec![k],
                Shape(vec![Dim::Seq, Dim::Const(kv_dim)]),
                l("attn.k_norm"),
            );
        }

        let rope = |b: &mut Builder, x: ValueId, heads: u64, width: u64, label: String| {
            b.push(
                Op::Rope {
                    theta: hp.rope_theta,
                    style: hp.rope_style,
                    n_heads: heads,
                    head_dim,
                },
                vec![x],
                Shape(vec![Dim::Seq, Dim::Const(width)]),
                label,
            )
        };
        let q = rope(&mut b, q, hp.n_heads, h, l("rope_q"));
        let k = rope(&mut b, k, hp.n_kv_heads, kv_dim, l("rope_k"));

        let att = b.push(
            Op::Attention {
                layer: li,
                n_heads: hp.n_heads,
                n_kv_heads: hp.n_kv_heads,
                head_dim,
            },
            vec![q, k, vv],
            seq_h.clone(),
            l("attention"),
        );
        let ow = b.require(&l("attn.o_proj.weight"), &[h, h])?;
        let o = b.push(
            Op::MatMul {
                weight: ow,
                bias: None,
            },
            vec![att],
            seq_h.clone(),
            l("attn.o_proj"),
        );
        x = b.push(Op::Add, vec![x, o], seq_h.clone(), l("residual_attn"));

        let fnw = b.require(&l("ffn_norm.weight"), &[h])?;
        let hf = b.push(
            Op::RmsNorm {
                weight: fnw,
                eps: hp.norm_eps,
                head_dim: None,
            },
            vec![x],
            seq_h.clone(),
            l("ffn_norm"),
        );
        let gw = b.require(&l("ffn.gate_proj.weight"), &[f, h])?;
        let uw = b.require(&l("ffn.up_proj.weight"), &[f, h])?;
        let dw = b.require(&l("ffn.down_proj.weight"), &[h, f])?;
        let seq_f = Shape(vec![Dim::Seq, Dim::Const(f)]);
        let g = b.push(
            Op::MatMul {
                weight: gw,
                bias: None,
            },
            vec![hf],
            seq_f.clone(),
            l("ffn.gate"),
        );
        let u = b.push(
            Op::MatMul {
                weight: uw,
                bias: None,
            },
            vec![hf],
            seq_f.clone(),
            l("ffn.up"),
        );
        let s = b.push(Op::SwiGlu, vec![g, u], seq_f, l("swiglu"));
        let d = b.push(
            Op::MatMul {
                weight: dw,
                bias: None,
            },
            vec![s],
            seq_h.clone(),
            l("ffn.down"),
        );
        x = b.push(Op::Add, vec![x, d], seq_h.clone(), l("residual_ffn"));
    }

    let onw = b.require("output_norm.weight", &[h])?;
    let x = b.push(
        Op::RmsNorm {
            weight: onw,
            eps: hp.norm_eps,
            head_dim: None,
        },
        vec![x],
        seq_h,
        "output_norm".into(),
    );
    // Tied embeddings: no lm_head.weight → project with the embedding table.
    let lm = match b.optional("lm_head.weight", &[v, h])? {
        Some(w) => w,
        None => embed_w,
    };
    let out = b.push(
        Op::MatMul {
            weight: lm,
            bias: None,
        },
        vec![x],
        Shape(vec![Dim::Seq, Dim::Const(v)]),
        "lm_head".into(),
    );

    Ok(Graph {
        nodes: b.nodes,
        output: out,
        n_layers: hp.n_layers,
        n_kv_heads: hp.n_kv_heads,
        head_dim,
    })
}
