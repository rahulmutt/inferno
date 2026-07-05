use crate::{PlanError, Result};
use inferno_graph::Graph;

/// KV cache physical layout: one contiguous region per layer holding both
/// the K and V tensors for the compile-time sequence bound, stored as F16.
#[derive(Debug, Clone, PartialEq)]
pub struct KvLayout {
    pub n_layers: usize,
    /// `n_kv_heads * head_dim`.
    pub kv_dim: usize,
    pub max_seq_len: usize,
    /// `kv_dim * max_seq_len * 2 (f16 bytes) * 2 (k + v)`.
    pub bytes_per_layer: usize,
    /// `bytes_per_layer * n_layers`.
    pub total_bytes: usize,
}

/// Size the KV cache: F16 storage for both K and V, at the compile-time
/// `max_seq_len` bound, one region per layer.
pub fn plan_kv(graph: &Graph, max_seq_len: usize) -> Result<KvLayout> {
    let kv_dim = graph
        .n_kv_heads
        .checked_mul(graph.head_dim)
        .ok_or(PlanError::Overflow("kv_dim"))? as usize;
    // F16 KV, k and v: 2 bytes/elem * 2 tensors.
    let bytes_per_layer = kv_dim
        .checked_mul(max_seq_len)
        .and_then(|x| x.checked_mul(4))
        .ok_or(PlanError::Overflow("kv per-layer"))?;
    let total_bytes = bytes_per_layer
        .checked_mul(graph.n_layers as usize)
        .ok_or(PlanError::Overflow("kv total"))?;
    Ok(KvLayout {
        n_layers: graph.n_layers as usize,
        kv_dim,
        max_seq_len,
        bytes_per_layer,
        total_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferno_formats::load_desc;
    use inferno_graph::build_graph;
    use std::path::Path;

    #[test]
    fn kv_bytes_are_f16_k_plus_v() {
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let g = build_graph(&desc).unwrap();
        let kv = plan_kv(&g, 64).unwrap();
        let kv_dim = (g.n_kv_heads * g.head_dim) as usize;
        assert_eq!(kv.kv_dim, kv_dim);
        assert_eq!(kv.bytes_per_layer, kv_dim * 64 * 2 * 2);
        assert_eq!(kv.total_bytes, kv.bytes_per_layer * g.n_layers as usize);
    }
}
