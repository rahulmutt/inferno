//! The `Backend` seam: decouples the generation loop from the execution
//! engine. M1 ships `InterpBackend` (the scalar interpreter); M3's compiled
//! path plugs in a `CompiledBackend` behind the same trait without touching
//! `Generator`'s sampling/streaming code.

use inferno_formats::ModelDesc;
use inferno_graph::{Graph, Interpreter, KvCache};

use crate::Result;

/// A token-generation execution engine: appends tokens to a running
/// sequence (advancing its own KV cache / arena) and returns logits for the
/// last position only. Teacher-forcing (all-position logits) is out of
/// scope — that stays on the interpreter directly (see `Generator::full_logits`).
pub trait Backend {
    /// Append `tokens`, advancing the KV cache; return last-token logits.
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;
    /// Reset KV/arena for a new sequence.
    fn reset(&mut self);
}

/// `Backend` implementation that runs the graph-walking scalar interpreter
/// (`inferno_graph::Interpreter`) — the M1 execution path. `forward` drives
/// `Interpreter::run` exactly as `Generator::generate` used to do directly,
/// threading the same `KvCache` across calls, and returns only the last
/// position's logits (a full generation step never needs more).
pub struct InterpBackend {
    desc: ModelDesc,
    graph: Graph,
    interp: Interpreter,
    kv: KvCache,
    vocab: usize,
    max_seq_len: usize,
}

impl InterpBackend {
    pub fn new(desc: ModelDesc, graph: Graph, max_seq_len: usize) -> Result<InterpBackend> {
        let kv = KvCache::new(&graph, max_seq_len)?;
        let vocab = desc.hyperparams.vocab_size as usize;
        Ok(InterpBackend {
            desc,
            graph,
            interp: Interpreter::new(),
            kv,
            vocab,
            max_seq_len,
        })
    }
}

impl Backend for InterpBackend {
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let result = self
            .interp
            .run(&self.desc, &self.graph, tokens, &mut self.kv)?;
        let last_start = (tokens.len() - 1) * self.vocab;
        Ok(result.data[last_start..last_start + self.vocab].to_vec())
    }

    fn reset(&mut self) {
        // A fresh sequence needs a fresh KV cache; graph/max_seq_len are
        // fixed for the backend's lifetime, so `KvCache::new` cannot fail
        // here if it did not fail in `InterpBackend::new`.
        self.kv = KvCache::new(&self.graph, self.max_seq_len)
            .expect("KvCache::new succeeded once at construction, so it must succeed again");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // InterpBackend.forward(prompt) must equal Interpreter.run(prompt) last row.
    #[test]
    fn interp_backend_matches_direct_interpreter() {
        use inferno_formats::load_desc;
        use inferno_graph::{Interpreter, KvCache, build_graph};
        use std::path::Path;
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let graph = build_graph(&desc).unwrap();
        let vocab = desc.hyperparams.vocab_size as usize;
        let tokens = vec![1u32, 2, 3];

        let mut be = InterpBackend::new(desc.clone(), graph.clone(), 64).unwrap();
        let got = be.forward(&tokens).unwrap();

        let mut interp = Interpreter::new();
        let mut kv = KvCache::new(&graph, 64).unwrap();
        let want = interp.run(&desc, &graph, &tokens, &mut kv).unwrap();
        let want_last = want.data[(tokens.len() - 1) * vocab..][..vocab].to_vec();
        assert_eq!(got, want_last);
    }
}
