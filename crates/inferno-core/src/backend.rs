//! [`inferno_runtime::Backend`] implementation over a compiled [`Artifact`]:
//! the M3 execution path plugged into `Generator` exactly like `InterpBackend`
//! (Task 15) — the generation loop / sampling / streaming code never changes.

use inferno_runtime::{Backend, Result as RuntimeResult};

use crate::artifact::{Artifact, ensure_kernels_linked};

/// Drives a compiled [`Artifact`] as a [`Backend`]: owns the KV cache, the
/// scratch arena, and the last-token logits buffer, all sized from the
/// artifact's [`Meta`](crate::Meta) at construction.
pub struct CompiledBackend {
    artifact: Artifact,
    kv: Vec<f32>,
    arena: Vec<f32>,
    logits: Vec<f32>,
    pos: usize,
    vocab: usize,
}

impl CompiledBackend {
    /// Allocate `kv`/`arena`/`logits` from `artifact.meta()` and wrap it as a
    /// fresh (`pos == 0`) backend.
    ///
    /// Calls [`ensure_kernels_linked`] so the host binary retains (and, with
    /// `-rdynamic`, exports) the kernel symbols the compiled `model.so`
    /// resolves against at `dlopen` time — every construction path for a
    /// `CompiledBackend` goes through here.
    pub fn new(artifact: Artifact) -> CompiledBackend {
        ensure_kernels_linked();
        let vocab = artifact.meta().vocab;
        let kv = vec![0.0; artifact.meta().kv_total_bytes / 4];
        let arena = vec![0.0; artifact.meta().arena_f32];
        let logits = vec![0.0; vocab];
        CompiledBackend {
            artifact,
            kv,
            arena,
            logits,
            pos: 0,
            vocab,
        }
    }
}

impl Backend for CompiledBackend {
    fn forward(&mut self, tokens: &[u32]) -> RuntimeResult<Vec<f32>> {
        debug_assert_eq!(
            self.logits.len(),
            self.vocab,
            "logits buffer must stay vocab-sized"
        );
        if self.pos == 0 {
            self.artifact
                .prefill(tokens, 0, &mut self.kv, &mut self.arena, &mut self.logits);
            self.pos += tokens.len();
        } else {
            for &t in tokens {
                self.artifact.decode_step(
                    t,
                    self.pos,
                    &mut self.kv,
                    &mut self.arena,
                    &mut self.logits,
                );
                self.pos += 1;
            }
        }
        Ok(self.logits.clone())
    }

    fn reset(&mut self) {
        self.pos = 0;
        self.kv.fill(0.0);
        self.arena.fill(0.0);
    }
}

// No `#[cfg(test)]` unit tests here: exercising `CompiledBackend::forward`
// means `dlopen`ing a real compiled `model.so`, which needs `-rdynamic` —
// applied (by `build.rs`) only to `tests/*.rs` integration-test binaries, not
// the `src/lib.rs` unit-test harness. See `tests/backend.rs`.
