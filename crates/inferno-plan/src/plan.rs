use crate::island::Island;
use crate::kv::KvLayout;
use crate::memory::ArenaLayout;
use crate::weights::WeightImageLayout;

/// Everything codegen needs, as pure data. No LLVM here.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// Fusion islands, in execution order.
    pub islands: Vec<Island>,
    /// Where each MatMul weight lives in the packed weight image.
    pub weights: WeightImageLayout,
    /// Activation arena: per-value byte offsets (liveness-packed) + scratch.
    pub arena: ArenaLayout,
    /// KV cache physical layout.
    pub kv: KvLayout,
    /// Compile-time sequence bound the arena/KV were sized for.
    pub max_seq_len: usize,
    /// Prefill tile length (tokens per batched forward pass); sizes the GEMM
    /// activation panel and the codegen tile loop.
    pub prefill_tile: usize,
}

impl Plan {
    /// A stable, human-readable text form of the plan (the "IR after
    /// planning" golden). Deterministic ordering only: islands and weights
    /// are already in graph order from Tasks 2-4, so no HashMap iteration
    /// is involved.
    pub fn dump(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        writeln!(
            s,
            "plan (max_seq_len={} prefill_tile={})",
            self.max_seq_len, self.prefill_tile
        )
        .unwrap();
        writeln!(s, "islands:").unwrap();
        for isl in &self.islands {
            writeln!(
                s,
                "  {:?} nodes[{}..{}] {}",
                isl.kind, isl.nodes.start, isl.nodes.end, isl.label
            )
            .unwrap();
        }
        writeln!(s, "weights: image_bytes={}", self.weights.image.len()).unwrap();
        for w in &self.weights.weights {
            writeln!(
                s,
                "  t{} {:?}/{} rows={} k={} @{}+{}",
                w.tensor_index, w.dtype, w.layout, w.rows, w.k, w.offset, w.len
            )
            .unwrap();
        }
        writeln!(
            s,
            "arena: total_f32={} act_scratch=@{}+{}",
            self.arena.total_f32, self.arena.act_scratch_off, self.arena.act_scratch_bytes
        )
        .unwrap();
        writeln!(
            s,
            "kv: layers={} kv_dim={} per_layer={} total={}",
            self.kv.n_layers, self.kv.kv_dim, self.kv.bytes_per_layer, self.kv.total_bytes
        )
        .unwrap();
        s
    }
}

#[cfg(test)]
mod tests {
    use inferno_formats::load_desc;
    use inferno_graph::build_graph;
    use inferno_target::TargetDesc;
    use std::path::Path;

    #[test]
    fn plan_dump_gguf() {
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let graph = build_graph(&desc).unwrap();
        let target = TargetDesc::detect().unwrap();
        let plan = crate::plan(&desc, &graph, &target, 64, 64).unwrap();
        insta::assert_snapshot!(plan.dump());
    }
}
