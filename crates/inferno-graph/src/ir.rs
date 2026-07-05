//! The graph IR (spec §Graph IR): a flat, SSA-style list of ops over
//! tensors named by index into `ModelDesc::tensors`. Builders (Task 7+)
//! produce a `Graph`; the scalar interpreter (Task 8+) executes it.

use std::fmt::Write as _;

use inferno_formats::{ModelDesc, RopeStyle};

/// Index of a node's output value. `0` is the reserved token-input value;
/// node `i` (0-indexed in `Graph::nodes`) produces value `i + 1`.
pub type ValueId = usize;

/// A single axis of a [`Shape`]: either a fixed size or the runtime
/// sequence length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    Const(u64),
    Seq,
}

/// The shape of a node's output value, outermost dimension first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<Dim>);

/// Index into `ModelDesc::tensors` identifying a weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRef(pub usize);

/// One operation in the graph IR.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Embed {
        weight: TensorRef,
    },
    MatMul {
        weight: TensorRef,
        bias: Option<TensorRef>,
    },
    /// `head_dim: Some(_)` selects per-head normalization (Qwen3 q/k norm).
    RmsNorm {
        weight: TensorRef,
        eps: f32,
        head_dim: Option<u64>,
    },
    Rope {
        theta: f32,
        style: RopeStyle,
        n_heads: u64,
        head_dim: u64,
    },
    Attention {
        layer: usize,
        n_heads: u64,
        n_kv_heads: u64,
        head_dim: u64,
    },
    SwiGlu,
    Add,
}

/// One node in the graph: an op, its input values, and its output shape.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<ValueId>,
    pub out_shape: Shape,
    pub label: String,
}

/// A Llama-family forward-pass graph: a flat op list plus the hyperparameters
/// the scalar interpreter needs (KV cache shape, layer count) but that don't
/// belong on any single node.
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    /// Output value id: logits `[Seq, vocab]`.
    pub output: ValueId,
    pub n_layers: u64,
    pub n_kv_heads: u64,
    pub head_dim: u64,
}

impl Graph {
    /// Stable text dump for snapshot tests. Weight refs render as
    /// @name[shape:dtype] so builder regressions show up in review.
    pub fn dump(&self, desc: &ModelDesc) -> String {
        let dim = |d: &Dim| match d {
            Dim::Const(c) => c.to_string(),
            Dim::Seq => "seq".into(),
        };
        let shape = |s: &Shape| format!("[{}]", s.0.iter().map(dim).collect::<Vec<_>>().join(","));
        let wref = |t: &TensorRef| {
            let td = &desc.tensors[t.0];
            format!(
                "@{}[{}:{:?}]",
                td.name,
                td.shape
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join("x"),
                td.dtype
            )
        };
        let mut out = format!(
            "graph (layers={}, kv_heads={}, head_dim={})\n  %0 = tokens : [seq]\n",
            self.n_layers, self.n_kv_heads, self.head_dim
        );
        for (i, n) in self.nodes.iter().enumerate() {
            let id = i + 1;
            let ins = |sep: &str| {
                n.inputs
                    .iter()
                    .map(|v| format!("%{v}"))
                    .collect::<Vec<_>>()
                    .join(sep)
            };
            let body = match &n.op {
                Op::Embed { weight } => format!("embed({}, {})", ins(", "), wref(weight)),
                Op::MatMul { weight, bias } => match bias {
                    Some(b) => {
                        format!("matmul({}, {}, bias={})", ins(", "), wref(weight), wref(b))
                    }
                    None => format!("matmul({}, {})", ins(", "), wref(weight)),
                },
                Op::RmsNorm {
                    weight,
                    eps,
                    head_dim,
                } => match head_dim {
                    Some(hd) => format!(
                        "rmsnorm_per_head({}, {}, eps={eps}, head_dim={hd})",
                        ins(", "),
                        wref(weight)
                    ),
                    None => format!("rmsnorm({}, {}, eps={eps})", ins(", "), wref(weight)),
                },
                Op::Rope {
                    theta,
                    style,
                    n_heads,
                    head_dim,
                } => format!(
                    "rope({}, theta={theta}, style={style:?}, heads={n_heads}, head_dim={head_dim})",
                    ins(", ")
                ),
                Op::Attention {
                    layer,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                } => format!(
                    "attention({}, layer={layer}, heads={n_heads}, kv_heads={n_kv_heads}, head_dim={head_dim})",
                    ins(", ")
                ),
                Op::SwiGlu => format!("swiglu({})", ins(", ")),
                Op::Add => format!("add({})", ins(", ")),
            };
            let _ = writeln!(
                out,
                "  %{id} = {body} : {}  ; {}",
                shape(&n.out_shape),
                n.label
            );
        }
        let _ = writeln!(out, "  output %{}", self.output);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dump_is_stable_and_readable() {
        use inferno_formats::fixtures;
        let desc =
            inferno_formats::gguf::parse(&mut std::io::Cursor::new(&fixtures::tiny_llama_gguf()))
                .unwrap();
        let embed_idx = desc
            .tensors
            .iter()
            .position(|t| t.name == "token_embed.weight")
            .unwrap();
        let g = Graph {
            nodes: vec![Node {
                op: Op::Embed {
                    weight: TensorRef(embed_idx),
                },
                inputs: vec![0],
                out_shape: Shape(vec![Dim::Seq, Dim::Const(64)]),
                label: "embed".into(),
            }],
            output: 1,
            n_layers: 2,
            n_kv_heads: 1,
            head_dim: 32,
        };
        let dump = g.dump(&desc);
        assert!(dump.contains("%1 = embed(%0, @token_embed.weight[260x64:F32]) : [seq,64]"));
        let _ = RopeStyle::HalfSplit; // silence unused import if assertions change
    }
}
