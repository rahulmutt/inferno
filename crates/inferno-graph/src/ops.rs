//! Obviously-correct scalar implementations of every graph op. No SIMD, no
//! rayon, no blocking — this is the oracle the compiled paths are compared
//! against (spec §Scalar interpreter). f32 accumulation throughout.

use inferno_formats::RopeStyle;

use crate::{GraphError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn rows(&self) -> usize {
        self.shape[0]
    }
    pub fn cols(&self) -> usize {
        self.shape[1]
    }
}

pub fn embed(ids: &[u32], table: &[f32], vocab: usize, hidden: usize) -> Result<Tensor> {
    let mut data = Vec::with_capacity(ids.len() * hidden);
    for &id in ids {
        let i = id as usize;
        if i >= vocab {
            return Err(GraphError::TokenOutOfRange { id, vocab });
        }
        data.extend_from_slice(&table[i * hidden..(i + 1) * hidden]);
    }
    Ok(Tensor {
        shape: vec![ids.len(), hidden],
        data,
    })
}

/// x [seq, k] · wᵀ, w row-major [n_out, k] (file order) → [seq, n_out].
pub fn matmul(x: &Tensor, w: &[f32], n_out: usize, k: usize, bias: Option<&[f32]>) -> Tensor {
    let seq = x.rows();
    let mut data = vec![0f32; seq * n_out];
    for s in 0..seq {
        let xr = &x.data[s * k..(s + 1) * k];
        for n in 0..n_out {
            let wr = &w[n * k..(n + 1) * k];
            let mut acc = 0f32;
            for j in 0..k {
                acc += xr[j] * wr[j];
            }
            data[s * n_out + n] = acc + bias.map_or(0.0, |b| b[n]);
        }
    }
    Tensor {
        shape: vec![seq, n_out],
        data,
    }
}

/// head_dim = None: normalize each row. Some(hd): normalize each hd-slice of
/// each row independently, cycling the weight (Qwen3 per-head q/k norm).
pub fn rmsnorm(x: &Tensor, w: &[f32], eps: f32, head_dim: Option<usize>) -> Tensor {
    let cols = x.cols();
    let unit = head_dim.unwrap_or(cols);
    let mut data = Vec::with_capacity(x.data.len());
    for chunk in x.data.chunks_exact(unit) {
        let ms = chunk.iter().map(|v| v * v).sum::<f32>() / unit as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (j, v) in chunk.iter().enumerate() {
            data.push(v * inv * w[j]);
        }
    }
    Tensor {
        shape: x.shape.clone(),
        data,
    }
}

pub fn rope(
    x: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    style: RopeStyle,
    pos_off: usize,
) -> Tensor {
    let mut out = x.clone();
    let half = head_dim / 2;
    for s in 0..x.rows() {
        let pos = (pos_off + s) as f32;
        for h in 0..n_heads {
            let base = s * x.cols() + h * head_dim;
            for i in 0..half {
                let freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = pos * freq;
                let (sin, cos) = angle.sin_cos();
                let (a, b) = match style {
                    RopeStyle::Interleaved => (base + 2 * i, base + 2 * i + 1),
                    RopeStyle::HalfSplit => (base + i, base + i + half),
                };
                let (x0, x1) = (out.data[a], out.data[b]);
                out.data[a] = x0 * cos - x1 * sin;
                out.data[b] = x0 * sin + x1 * cos;
            }
        }
    }
    out
}

pub fn swiglu(gate: &Tensor, up: &Tensor) -> Tensor {
    let data = gate
        .data
        .iter()
        .zip(&up.data)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    Tensor {
        shape: gate.shape.clone(),
        data,
    }
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Tensor {
        shape: a.shape.clone(),
        data,
    }
}

/// Causal GQA attention. q [seq, n_heads*head_dim]; kcache/vcache hold
/// `total` rows of n_kv_heads*head_dim (this batch already appended).
/// Query row s has absolute position pos_off + s and attends to keys
/// 0..=pos_off+s. Softmax with max-subtraction, all f32.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &Tensor,
    kcache: &[f32],
    vcache: &[f32],
    total: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos_off: usize,
) -> Tensor {
    let seq = q.rows();
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let group = n_heads / n_kv_heads;
    let mut data = vec![0f32; seq * n_heads * head_dim];
    let mut scores = vec![0f32; total];
    for s in 0..seq {
        let visible = pos_off + s + 1; // causal horizon
        for h in 0..n_heads {
            let g = h / group;
            let qv = &q.data[s * n_heads * head_dim + h * head_dim..][..head_dim];
            for (t, sc) in scores[..visible].iter_mut().enumerate() {
                let kv = &kcache[t * kv_dim + g * head_dim..][..head_dim];
                *sc = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let max = scores[..visible]
                .iter()
                .fold(f32::NEG_INFINITY, |m, v| m.max(*v));
            let mut denom = 0f32;
            for sc in &mut scores[..visible] {
                *sc = (*sc - max).exp();
                denom += *sc;
            }
            let out = &mut data[s * n_heads * head_dim + h * head_dim..][..head_dim];
            for (t, &w) in scores[..visible].iter().enumerate() {
                let vv = &vcache[t * kv_dim + g * head_dim..][..head_dim];
                let w = w / denom;
                for (o, v) in out.iter_mut().zip(vv) {
                    *o += w * v;
                }
            }
        }
    }
    Tensor {
        shape: vec![seq, n_heads * head_dim],
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferno_formats::RopeStyle;

    fn t(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor {
            shape: shape.to_vec(),
            data: data.to_vec(),
        }
    }

    #[test]
    fn embed_looks_up_rows_and_bounds_checks() {
        let table = [0.0, 1.0, 2.0, 3.0]; // vocab 2, hidden 2
        let e = embed(&[1, 0], &table, 2, 2).unwrap();
        assert_eq!(e.data, vec![2.0, 3.0, 0.0, 1.0]);
        assert!(matches!(
            embed(&[2], &table, 2, 2),
            Err(crate::GraphError::TokenOutOfRange { id: 2, vocab: 2 })
        ));
    }

    #[test]
    fn matmul_hand_computed() {
        // x = [[1,2]], w rows: [3,4], [5,6]  → [1*3+2*4, 1*5+2*6] = [11, 17]
        let out = matmul(&t(&[1, 2], &[1.0, 2.0]), &[3.0, 4.0, 5.0, 6.0], 2, 2, None);
        assert_eq!(out.data, vec![11.0, 17.0]);
        let out = matmul(
            &t(&[1, 2], &[1.0, 2.0]),
            &[3.0, 4.0, 5.0, 6.0],
            2,
            2,
            Some(&[10.0, 20.0]),
        );
        assert_eq!(out.data, vec![21.0, 37.0]);
    }

    #[test]
    fn rmsnorm_hand_computed() {
        // x=[3,4]: rms = sqrt((9+16)/2 + 0) = 3.5355339; w=[2,2] → [1.6970563, 2.2627417]
        let out = rmsnorm(&t(&[1, 2], &[3.0, 4.0]), &[2.0, 2.0], 0.0, None);
        assert!((out.data[0] - 1.697_056_3).abs() < 1e-5);
        assert!((out.data[1] - 2.262_741_7).abs() < 1e-5);
    }

    #[test]
    fn rmsnorm_large_eps_is_inside_sqrt() {
        // x=[1], w=[1], eps=3: ms = 1*1/1 = 1; inv = 1/sqrt(1 + 3) = 0.5
        // → out = 1 * 0.5 * 1 = 0.5 exactly. Distinguishes eps-inside-sqrt
        // from 1/sqrt(ms)+eps (=1+3=4) or 1/(sqrt(ms)+eps) (=1/(1+3)=0.25).
        let out = rmsnorm(&t(&[1, 1], &[1.0]), &[1.0], 3.0, None);
        assert_eq!(out.data, vec![0.5]);
    }

    #[test]
    fn rmsnorm_per_head_normalizes_each_head() {
        // 2 heads of dim 2; second head huge — must not affect first head's norm.
        let x = t(&[1, 4], &[3.0, 4.0, 300.0, 400.0]);
        let out = rmsnorm(&x, &[1.0, 1.0], 0.0, Some(2));
        assert!((out.data[0] - 3.0 / 3.535_534).abs() < 1e-5);
        assert!((out.data[2] - 300.0 / 353.553_39).abs() < 1e-3);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let x = t(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);
        for style in [RopeStyle::Interleaved, RopeStyle::HalfSplit] {
            let out = rope(&x, 1, 4, 10000.0, style, 0);
            assert_eq!(out.data, x.data, "{style:?}");
        }
    }

    #[test]
    fn rope_hand_computed_position_one() {
        // head_dim 2, one pair, freq = theta^0 = 1 → angle = pos = 1 rad.
        // [1, 0] at pos 1 → [cos1, sin1] in both styles (pair (0,1)).
        let x = t(&[1, 2], &[1.0, 0.0]);
        for style in [RopeStyle::Interleaved, RopeStyle::HalfSplit] {
            let out = rope(&x, 1, 2, 10000.0, style, 1);
            assert!((out.data[0] - 0.540_302_3).abs() < 1e-6, "{style:?}");
            assert!((out.data[1] - 0.841_470_96).abs() < 1e-6, "{style:?}");
        }
    }

    #[test]
    fn rope_hand_computed_position_one_nonzero_cross_term() {
        // head_dim 2, one pair, freq = theta^0 = 1 → angle = pos = 1 rad.
        // x1 nonzero here, so both cross-terms (the -sin and +sin ones) are
        // exercised: out0 = x0 cos - x1 sin, out1 = x0 sin + x1 cos.
        // x=[1,2] at pos 1: cos1=0.5403023, sin1=0.8414710
        //   out0 = 1*0.5403023 - 2*0.8414710 = -1.1426397
        //   out1 = 1*0.8414710 + 2*0.5403023 =  1.9220756
        let x = t(&[1, 2], &[1.0, 2.0]);
        for style in [RopeStyle::Interleaved, RopeStyle::HalfSplit] {
            let out = rope(&x, 1, 2, 10000.0, style, 1);
            assert!((out.data[0] - (-1.142_639_7)).abs() < 1e-5, "{style:?}");
            assert!((out.data[1] - 1.922_075_6).abs() < 1e-5, "{style:?}");
        }
    }

    #[test]
    fn rope_styles_pair_differently_at_dim4() {
        // head_dim 4: Interleaved pairs (0,1),(2,3); HalfSplit pairs (0,2),(1,3).
        let x = t(&[1, 4], &[1.0, 0.0, 1.0, 0.0]);
        let inter = rope(&x, 1, 4, 10000.0, RopeStyle::Interleaved, 1);
        let half = rope(&x, 1, 4, 10000.0, RopeStyle::HalfSplit, 1);
        assert_ne!(inter.data, half.data);
    }

    #[test]
    fn swiglu_hand_computed() {
        // silu(1) = 1/(1+e^-1) = 0.731058578; × up 2 = 1.462117
        let out = swiglu(&t(&[1, 1], &[1.0]), &t(&[1, 1], &[2.0]));
        assert!((out.data[0] - 1.462_117_2).abs() < 1e-6);
    }

    #[test]
    fn attention_single_head_hand_computed() {
        // 1 head, head_dim 1, scale 1. Cache: k=[1, 2], v=[10, 20], q for the
        // 2nd position (pos_off 1... here: q is the second row, total=2).
        // q=1: scores [1, 2] → softmax [e1,e2]/Σ → out = (10 e1 + 20 e2)/(e1+e2)
        let q = t(&[1, 1], &[1.0]);
        let e1 = 1f32.exp();
        let e2 = 2f32.exp();
        let expect = (10.0 * e1 + 20.0 * e2) / (e1 + e2);
        let out = attention(&q, &[1.0, 2.0], &[10.0, 20.0], 2, 1, 1, 1, 1);
        assert!((out.data[0] - expect).abs() < 1e-4);
    }

    #[test]
    fn attention_is_causal() {
        // 2 query rows over a 2-row cache with pos_off 0: row 0 may only see
        // key 0; making key 1 enormous must not change row 0's output.
        let q = t(&[2, 1], &[1.0, 1.0]);
        let a = attention(&q, &[1.0, 1.0], &[10.0, 999.0], 2, 1, 1, 1, 0);
        assert!((a.data[0] - 10.0).abs() < 1e-5); // only v[0] visible
    }

    #[test]
    fn attention_gqa_maps_query_heads_to_shared_kv_head() {
        // 2 query heads share 1 kv head (head_dim 1): both heads read the
        // same cache but with their own q values.
        let q = t(&[1, 2], &[1.0, 3.0]);
        let out = attention(&q, &[1.0], &[7.0], 1, 2, 1, 1, 0);
        assert_eq!(out.data, vec![7.0, 7.0]); // single key → output is v
    }

    #[test]
    fn attention_gqa_group_mapping_is_contiguous_not_interleaved() {
        // 4 query heads, 2 kv heads (group=2), head_dim 1, single cached
        // position → softmax weight is trivially 1, so each head's output
        // is exactly its mapped kv head's v. Correct mapping g = h / group:
        // heads 0,1 → kv0 (v=100), heads 2,3 → kv1 (v=200) → [100,100,200,200].
        // The interleaved alternative g = h % n_kv_heads would instead give
        // [100,200,100,200], which fails this assertion.
        let q = t(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let out = attention(&q, &[1.0, 2.0], &[100.0, 200.0], 1, 4, 2, 1, 0);
        assert_eq!(out.data, vec![100.0, 100.0, 200.0, 200.0]);
    }
}
