//! `inferno-core`: the embeddable engine. Task 13 adds the content-addressed
//! cache key + cache directory; Task 14 the dlopen loader; Task 16 the
//! `CompiledBackend` + `Engine` seam the CLI drives. See
//! docs/superpowers/specs/2026-07-05-m3-compiler-design.md.

pub mod artifact;
pub mod backend;
pub mod cache;
pub mod error;

use std::path::{Path, PathBuf};

use inferno_target::TargetDesc;

pub use artifact::{Artifact, ensure_kernels_linked};
pub use backend::CompiledBackend;
pub use cache::{cache_dir, cache_key, content_hash};
pub use error::{CoreError, Result};
// Re-export codegen's `Meta`: it is written by codegen and read/finalized here.
pub use inferno_codegen::Meta;

/// The embeddable entry point: detects the host target once, then compiles
/// (or loads a cached compile of) a model on demand.
///
/// `max_seq_len` here MUST be the same value the caller ultimately hands to
/// `inferno_runtime::Generator` (`Generator::load_with_backend`/`load`): the
/// Generator uses it for its own context-full bookkeeping, and a mismatch
/// would desync that check from the `CompiledBackend`'s real KV capacity
/// (which is sized off this `Engine`'s `max_seq_len`).
pub struct Engine {
    model: PathBuf,
    target: TargetDesc,
    max_seq_len: usize,
}

impl Engine {
    /// Detect the host target and record `model`/`max_seq_len`. Does not
    /// compile anything yet — that happens lazily in
    /// [`compiled_backend`](Self::compiled_backend).
    pub fn load(model: &Path, max_seq_len: usize) -> Result<Engine> {
        let target = TargetDesc::detect()?;
        Ok(Engine {
            model: model.to_path_buf(),
            target,
            max_seq_len,
        })
    }

    /// Compile (or load a verified cached compile of) the model for this
    /// engine's target/`max_seq_len`, and build a ready-to-use
    /// [`CompiledBackend`] over it.
    pub fn compiled_backend(&self) -> Result<CompiledBackend> {
        let artifact = Artifact::load_or_compile(&self.model, &self.target, self.max_seq_len)?;
        Ok(CompiledBackend::new(artifact))
    }

    /// The on-disk cache directory this engine's `model`/target/`max_seq_len`
    /// resolve to (`model.so`/`weights.bin`/`meta.json`), whether or not a
    /// compile has happened yet. Used by `inferno compile` to report where
    /// the artifact landed.
    pub fn cache_dir(&self) -> Result<PathBuf> {
        let key = cache_key(&self.model, &self.target, self.max_seq_len)?;
        Ok(cache::cache_dir(&key))
    }
}
