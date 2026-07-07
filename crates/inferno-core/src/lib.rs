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
    threads: usize,
    opts: inferno_codegen::CompileOptions,
}

impl Engine {
    /// Detect the host target and record `model`/`max_seq_len`. Does not
    /// compile anything yet — that happens lazily in
    /// [`compiled_backend`](Self::compiled_backend).
    pub fn load(model: &Path, max_seq_len: usize) -> Result<Engine> {
        let target = TargetDesc::detect()?;
        let threads = target.topology.physical_cores.max(1) as usize;
        Ok(Engine {
            model: model.to_path_buf(),
            target,
            max_seq_len,
            threads,
            opts: inferno_codegen::CompileOptions::default(),
        })
    }

    /// Compiled-path thread count for backends built by this engine.
    /// Defaults to the target's physical cores; clamped to
    /// `1..=logical_cores` (the pool's spec bounds).
    pub fn set_threads(&mut self, threads: usize) {
        let max = self.target.topology.logical_cores.max(1) as usize;
        self.threads = threads.clamp(1, max);
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Enable per-op profiling for artifacts this engine builds (distinct
    /// cache entry). Off by default.
    pub fn set_profile(&mut self, on: bool) {
        self.opts.profile = on;
    }

    /// Prefill tile length for artifacts this engine builds.
    pub fn set_prefill_tile(&mut self, t: usize) {
        self.opts.prefill_tile = t.max(1);
    }

    /// Compile (or load a verified cached compile of) the model for this
    /// engine's target/`max_seq_len`, and build a ready-to-use
    /// [`CompiledBackend`] over it. Also sizes the process-global
    /// `inferno-pool` thread pool to `self.threads` (initializing it on
    /// first use, loud error on a mismatched re-init) and caps active
    /// parallelism to that count before the backend runs any GEMVs.
    pub fn compiled_backend(&self) -> Result<CompiledBackend> {
        // Size the process-global pool once (loud error on a mismatched
        // re-init — spec), then cap active parallelism to this engine's
        // count so bench's t=1 diagnostics can vary it per run.
        inferno_pool::init_global(self.threads)?;
        inferno_pool::set_global_active_threads(self.threads);
        let artifact =
            Artifact::load_or_compile(&self.model, &self.target, self.max_seq_len, &self.opts)?;
        Ok(CompiledBackend::new(artifact))
    }

    /// The on-disk cache directory this engine's `model`/target/`max_seq_len`
    /// resolve to (`model.so`/`weights.bin`/`meta.json`), whether or not a
    /// compile has happened yet. Used by `inferno compile` to report where
    /// the artifact landed.
    pub fn cache_dir(&self) -> Result<PathBuf> {
        let key = cache_key(&self.model, &self.target, self.max_seq_len, &self.opts)?;
        Ok(cache::cache_dir(&key))
    }

    /// Per-slot weight bytes touched by ONE forward-pass invocation of each
    /// profiler slot in `slots` (`0` for non-matmul slots).
    ///
    /// The profiler aggregates every layer sharing a `matmul:<normalized
    /// name>` label into a single counter (see `inferno_codegen::profile`),
    /// so this sums the packed byte length of every weight tensor whose
    /// normalized name matches that label — i.e. the bytes read from weights
    /// by one token's worth of work across all layers, exactly mirroring
    /// what one accumulated cycle count already covers.
    ///
    /// This re-derives the (pure, LLVM-free) `inferno_plan::Plan` from the
    /// model rather than caching one from compile time — `Engine` doesn't
    /// keep it around, and building it is cheap relative to the model run
    /// this is used alongside (`inferno run --profile`).
    ///
    /// The CLI multiplies the result by each phase's per-token invocation
    /// count (prompt tokens for prefill, generated tokens for decode) to
    /// approximate total phase bytes for the `--profile` GB/s column — this
    /// assumes one full weight read per token, which is only exact for
    /// decode (prefill batches tokens through fewer, larger calls); it is a
    /// diagnostic approximation, not a contract.
    pub fn profile_matmul_bytes(&self, slots: &[String]) -> Result<Vec<u64>> {
        let desc = inferno_formats::load_desc(&self.model)?;
        let graph = inferno_graph::build_graph(&desc)?;
        let plan = inferno_plan::plan(
            &desc,
            &graph,
            &self.target,
            self.max_seq_len,
            self.opts.prefill_tile,
        )?;
        let slot_index: std::collections::HashMap<&str, usize> = slots
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        let mut bytes = vec![0u64; slots.len()];
        for w in &plan.weights.weights {
            let name = &desc.tensors[w.tensor_index].name;
            let label = format!(
                "matmul:{}",
                inferno_codegen::profile::normalize_weight_name(name)
            );
            if let Some(&i) = slot_index.get(label.as_str()) {
                bytes[i] += w.len as u64;
            }
        }
        Ok(bytes)
    }
}
