//! Graph-walking scalar interpreter. Weights dequantize to f32 lazily on
//! first use and stay cached (fine for the ≤~1B-param models the oracle
//! targets — ~4 bytes/param). KV cache is allocated once up front.

use std::collections::HashMap;

use inferno_formats::{ModelDesc, quant, read_tensor_bytes};

use crate::ir::{Graph, Op};
use crate::ops::{self, Tensor};
use crate::{GraphError, Result};

/// Hostile-input guard: max total KV allocation (spec §Error handling).
const MAX_KV_BYTES: u64 = 8 << 30;

pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    kv_dim: usize,
    max_seq_len: usize,
    len: usize,
}

impl KvCache {
    pub fn new(graph: &Graph, max_seq_len: usize) -> Result<KvCache> {
        let kv_dim = (graph.n_kv_heads * graph.head_dim) as usize;
        let per_layer = (kv_dim as u64)
            .checked_mul(max_seq_len as u64)
            .and_then(|n| n.checked_mul(8)) // k + v, 4 bytes each
            .ok_or_else(|| GraphError::BadHyperParams("kv size overflow".into()))?;
        let total = per_layer
            .checked_mul(graph.n_layers)
            .ok_or_else(|| GraphError::BadHyperParams("kv size overflow".into()))?;
        if total > MAX_KV_BYTES {
            return Err(GraphError::BadHyperParams(format!(
                "kv cache would need {total} bytes (limit {MAX_KV_BYTES})"
            )));
        }
        let layers = graph.n_layers as usize;
        let mk = || {
            let mut v = Vec::new();
            v.reserve_exact(kv_dim * max_seq_len);
            v
        };
        Ok(KvCache {
            k: (0..layers).map(|_| mk()).collect(),
            v: (0..layers).map(|_| mk()).collect(),
            kv_dim,
            max_seq_len,
            len: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[derive(Default)]
pub struct Interpreter {
    weights: HashMap<usize, Vec<f32>>,
}

impl Interpreter {
    pub fn new() -> Interpreter {
        Interpreter::default()
    }

    fn weight(&mut self, desc: &ModelDesc, idx: usize) -> Result<&[f32]> {
        if let std::collections::hash_map::Entry::Vacant(e) = self.weights.entry(idx) {
            let t = &desc.tensors[idx];
            let n: u64 = t.shape.iter().product();
            let bytes = read_tensor_bytes(desc, t)?;
            let vals = quant::dequant(&t.dtype, &bytes, n as usize)?;
            e.insert(vals);
        }
        Ok(self.weights.get(&idx).unwrap().as_slice())
    }

    pub fn run(
        &mut self,
        desc: &ModelDesc,
        graph: &Graph,
        tokens: &[u32],
        kv: &mut KvCache,
    ) -> Result<Tensor> {
        if kv.len + tokens.len() > kv.max_seq_len {
            return Err(GraphError::SeqTooLong {
                got: kv.len + tokens.len(),
                max: kv.max_seq_len,
            });
        }
        let pos_off = kv.len;
        let hp = &desc.hyperparams;
        // env[v]: value v's tensor. env[0] unused (tokens are read directly).
        let mut env: Vec<Option<Tensor>> = vec![None; graph.nodes.len() + 1];
        for (i, node) in graph.nodes.iter().enumerate() {
            let id = i + 1;
            let arg = |v: usize, env: &[Option<Tensor>]| -> Tensor {
                env[v].clone().expect("graph is topologically ordered")
            };
            let out = match &node.op {
                Op::Embed { weight } => {
                    let table = self.weight(desc, weight.0)?;
                    ops::embed(
                        tokens,
                        table,
                        hp.vocab_size as usize,
                        hp.hidden_size as usize,
                    )?
                }
                Op::MatMul { weight, bias } => {
                    let x = arg(node.inputs[0], &env);
                    let wt = &desc.tensors[weight.0];
                    let (n_out, k) = (wt.shape[0] as usize, wt.shape[1] as usize);
                    let b = match bias {
                        Some(br) => Some(self.weight(desc, br.0)?.to_vec()),
                        None => None,
                    };
                    let w = self.weight(desc, weight.0)?;
                    ops::matmul(&x, w, n_out, k, b.as_deref())
                }
                Op::RmsNorm {
                    weight,
                    eps,
                    head_dim,
                } => {
                    let x = arg(node.inputs[0], &env);
                    let w = self.weight(desc, weight.0)?;
                    ops::rmsnorm(&x, w, *eps, head_dim.map(|d| d as usize))
                }
                Op::Rope {
                    theta,
                    style,
                    n_heads,
                    head_dim,
                } => {
                    let x = arg(node.inputs[0], &env);
                    ops::rope(
                        &x,
                        *n_heads as usize,
                        *head_dim as usize,
                        *theta,
                        *style,
                        pos_off,
                    )
                }
                Op::Attention {
                    layer,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                } => {
                    let q = arg(node.inputs[0], &env);
                    let k = arg(node.inputs[1], &env);
                    let v = arg(node.inputs[2], &env);
                    kv.k[*layer].extend_from_slice(&k.data);
                    kv.v[*layer].extend_from_slice(&v.data);
                    let total = kv.k[*layer].len() / kv.kv_dim;
                    ops::attention(
                        &q,
                        &kv.k[*layer],
                        &kv.v[*layer],
                        total,
                        *n_heads as usize,
                        *n_kv_heads as usize,
                        *head_dim as usize,
                        pos_off,
                    )
                }
                Op::SwiGlu => ops::swiglu(&arg(node.inputs[0], &env), &arg(node.inputs[1], &env)),
                Op::Add => ops::add(&arg(node.inputs[0], &env), &arg(node.inputs[1], &env)),
            };
            env[id] = Some(out);
        }
        kv.len += tokens.len();
        Ok(env[graph.output].take().expect("output value produced"))
    }
}
