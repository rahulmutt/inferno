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
}
