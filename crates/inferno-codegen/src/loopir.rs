//! Plan -> Loop IR: translate each fusion island's graph nodes into a flat
//! list of [`Step`]s, one step per op (plus the `Quantize`/`Gemv`/`Bias`
//! expansion of a `MatMul`). Pure data — no LLVM here (see lib.rs doc).

use inferno_formats::{DType, ModelDesc, RopeStyle};
use inferno_graph::{Graph, Op};
use inferno_plan::Plan;
use inferno_plan::island::IslandKind;

/// One executable step in an island's body. Value ids (`out`/`src`/…) are
/// graph value ids (node `i` produces value `i + 1`); resolving them to
/// arena byte offsets happens at LLVM-emission time, not here.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    Embed {
        table: usize,
        out: usize,
    },
    /// Quantize an f32 activation into the act scratch region ahead of a
    /// quantized-weight `Gemv`. Never emitted for F32 (native or widened
    /// F16/BF16) weights.
    Quantize {
        dtype: DType,
        src: usize,
        k: usize,
    },
    Gemv {
        symbol: String,
        /// Index into `Plan.weights.weights`.
        weight: usize,
        out: usize,
        rows: usize,
        k: usize,
    },
    Bias {
        bias_tensor: usize,
        out: usize,
        rows: usize,
    },
    RmsNorm {
        src: usize,
        weight: usize,
        eps: f32,
        out: usize,
        head_dim: Option<usize>,
    },
    Rope {
        src: usize,
        out: usize,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        style: RopeStyle,
    },
    Attention {
        q: usize,
        layer: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        out: usize,
    },
    SwiGlu {
        gate: usize,
        up: usize,
        out: usize,
    },
    Add {
        a: usize,
        b: usize,
        out: usize,
    },
}

/// One fusion island's translated body.
#[derive(Debug, Clone, PartialEq)]
pub struct IslandCode {
    pub kind: IslandKind,
    pub steps: Vec<Step>,
}

/// The full program: one [`IslandCode`] per fusion island, in execution
/// order (mirrors `Plan.islands`).
#[derive(Debug, Clone, PartialEq)]
pub struct LoopIr {
    pub islands: Vec<IslandCode>,
}

/// `inferno_gemv_{dtype}_rs8_{isa}`: the packed-weight kernel entry point
/// selected for a MatMul's *stored* dtype (widened F16/BF16 weights are
/// stored as F32 — see `weights::pack_weight` — so they correctly resolve to
/// the f32 kernel, not a quantized one) and the target ISA.
fn gemv_symbol(dtype: &DType, isa: inferno_kernels::KernelIsa) -> String {
    let d = match dtype {
        DType::F32 => "f32",
        DType::Q8_0 => "q8_0",
        DType::Q4_K => "q4_k",
        _ => "f32",
    };
    let i = match isa {
        inferno_kernels::KernelIsa::Scalar => "scalar",
        inferno_kernels::KernelIsa::Avx2 => "avx2",
    };
    format!("inferno_gemv_{d}_rs8_{i}")
}

/// Translate a [`Plan`]'s fusion islands into a [`LoopIr`]: one [`Step`] per
/// graph node (a `MatMul` expands to `Quantize?` + `Gemv` + `Bias?`).
pub fn build_loopir(plan: &Plan, graph: &Graph, _desc: &ModelDesc) -> LoopIr {
    let find_weight = |tensor_index: usize| {
        plan.weights
            .weights
            .iter()
            .position(|w| w.tensor_index == tensor_index)
            .expect("every MatMul weight was packed in Task 3")
    };

    let mut islands = Vec::new();
    for isl in &plan.islands {
        let mut steps = Vec::new();
        for n in isl.nodes.clone() {
            let node = &graph.nodes[n];
            let out = n + 1;
            match &node.op {
                Op::Embed { weight } => steps.push(Step::Embed {
                    table: weight.0,
                    out,
                }),
                Op::MatMul { weight, bias } => {
                    let wi = find_weight(weight.0);
                    let pw = &plan.weights.weights[wi];
                    if pw.dtype != DType::F32 {
                        steps.push(Step::Quantize {
                            dtype: pw.dtype.clone(),
                            src: node.inputs[0],
                            k: pw.k,
                        });
                    }
                    steps.push(Step::Gemv {
                        symbol: gemv_symbol(&pw.dtype, pw.isa),
                        weight: wi,
                        out,
                        rows: pw.rows,
                        k: pw.k,
                    });
                    if let Some(b) = bias {
                        steps.push(Step::Bias {
                            bias_tensor: b.0,
                            out,
                            rows: pw.rows,
                        });
                    }
                }
                Op::RmsNorm {
                    weight,
                    eps,
                    head_dim,
                } => steps.push(Step::RmsNorm {
                    src: node.inputs[0],
                    weight: weight.0,
                    eps: *eps,
                    out,
                    head_dim: head_dim.map(|d| d as usize),
                }),
                Op::Rope {
                    theta,
                    style,
                    n_heads,
                    head_dim,
                } => steps.push(Step::Rope {
                    src: node.inputs[0],
                    out,
                    n_heads: *n_heads as usize,
                    head_dim: *head_dim as usize,
                    theta: *theta,
                    style: *style,
                }),
                Op::Attention {
                    layer,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                } => steps.push(Step::Attention {
                    q: node.inputs[0],
                    layer: *layer,
                    n_heads: *n_heads as usize,
                    n_kv_heads: *n_kv_heads as usize,
                    head_dim: *head_dim as usize,
                    out,
                }),
                Op::SwiGlu => steps.push(Step::SwiGlu {
                    gate: node.inputs[0],
                    up: node.inputs[1],
                    out,
                }),
                Op::Add => steps.push(Step::Add {
                    a: node.inputs[0],
                    b: node.inputs[1],
                    out,
                }),
            }
        }
        islands.push(IslandCode {
            kind: isl.kind,
            steps,
        });
    }
    LoopIr { islands }
}

impl LoopIr {
    /// A stable, portable text dump: the `Gemv` kernel symbol has its
    /// `_scalar`/`_avx2` ISA suffix stripped back to `_rs8` so the snapshot
    /// doesn't depend on the host CPU running the test.
    pub fn dump(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        for isl in &self.islands {
            writeln!(s, "island {:?}", isl.kind).unwrap();
            for step in &isl.steps {
                match step {
                    Step::Gemv {
                        symbol,
                        weight,
                        out,
                        rows,
                        k,
                    } => {
                        let base = symbol
                            .rsplit_once("_rs8_")
                            .map(|(b, _)| b)
                            .unwrap_or(symbol);
                        writeln!(s, "  gemv {base}_rs8 w{weight} -> %{out} rows={rows} k={k}")
                            .unwrap();
                    }
                    other => writeln!(s, "  {other:?}").unwrap(),
                }
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferno_formats::load_desc;
    use inferno_graph::build_graph;
    use inferno_target::TargetDesc;
    use std::path::Path;

    #[test]
    fn loopir_dump_gguf() {
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let graph = build_graph(&desc).unwrap();
        let target = TargetDesc::detect().unwrap();
        let plan = inferno_plan::plan(&desc, &graph, &target, 64).unwrap();
        let lir = build_loopir(&plan, &graph, &desc);
        // Structure is ISA-independent; symbol suffix is not, so dump the
        // symbol base without the _scalar/_avx2 suffix (see dump()).
        insta::assert_snapshot!(lir.dump());
    }
}
