//! Object emission + link + artifact write: turns a verified [`LlvmModule`]
//! into an on-disk `model.so` + `weights.bin` + `meta.json` triple.
//!
//! Hashing layering: `Meta`'s `model_hash`/`target_hash`/`weights_hash` are
//! left as empty strings here. `inferno-core`'s `cache` module (Task 13/14)
//! owns content hashing end to end (`cache_key`) and rewrites `meta.json`
//! with the real hashes after compile; this crate does no hashing and takes
//! no hashing dependency.

use std::path::{Path, PathBuf};
use std::process::Command;

use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use serde::{Deserialize, Serialize};

use crate::{CodegenError, Result};

/// Compile-time options that change the emitted artifact (and therefore its
/// cache identity). Both fields are folded into `inferno-core`'s cache key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileOptions {
    /// Emit per-op `readcyclecounter` instrumentation + the
    /// `inferno_prof_counters` global (Task 3). A profiled `model.so` is a
    /// distinct artifact; logits are bit-identical to the unprofiled build.
    pub profile: bool,
    /// Prefill tile length (tokens per batched forward pass); sizes the
    /// GEMM activation panel (`act_scratch`) and the codegen tile loop.
    pub prefill_tile: usize,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            profile: false,
            prefill_tile: crate::PREFILL_TILE,
        }
    }
}

/// A compiled model on disk: `dir` contains `model.so`, `weights.bin`, and
/// `meta.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    pub dir: PathBuf,
}

/// Sidecar metadata written alongside the compiled shared object. Hash
/// fields are placeholders (empty strings) here; `inferno-core` (Task 13/14)
/// recomputes and rewrites them with real content hashes after compile.
///
/// `Deserialize` is derived so `inferno-core` (Task 14) can read `meta.json`
/// back to verify hashes and size the KV / arena / logits buffers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Meta {
    pub model_hash: String,
    pub target_hash: String,
    pub weights_hash: String,
    pub inferno_version: String,
    pub vocab: usize,
    pub n_layers: usize,
    pub arena_f32: usize,
    pub kv_total_bytes: usize,
    pub max_seq_len: usize,
    pub entry_prefill: String,
    pub entry_decode: String,
    /// Prefill tile length this artifact was compiled for (Task 7).
    pub prefill_tile: usize,
    /// Per-op profiler slot labels, slot index = position (empty if this
    /// artifact was compiled without `profile`). Task 3 populates it.
    #[serde(default)]
    pub profile_slots: Vec<String>,
}

/// Plan -> Loop IR -> LLVM IR -> object -> `model.so`, plus the `weights.bin`
/// / `meta.json` sidecars, all written under `out_dir`.
pub fn compile(
    desc: &inferno_formats::ModelDesc,
    graph: &inferno_graph::Graph,
    target: &inferno_target::TargetDesc,
    max_seq_len: usize,
    opts: &CompileOptions,
    out_dir: &Path,
) -> Result<Artifact> {
    let plan = inferno_plan::plan(desc, graph, target, max_seq_len, opts.prefill_tile)?;
    let ctx = Context::create();
    let module = crate::llvm::build_full_module(&ctx, &plan, graph, desc)?;
    module.verify()?;

    Target::initialize_x86(&InitializationConfig::default());
    let triple = TargetMachine::get_default_triple();
    let tgt = Target::from_triple(&triple).map_err(|e| CodegenError::Emit(e.to_string()))?;
    // AllowFPOpFusion defaults to Standard (fuses only flagged ops); the
    // lowered IR carries no fast-math flags, so this cannot contract
    // fmul+fadd into fma and stays bit-parity with the interpreter oracle.
    let tm = tgt
        .create_target_machine(
            &triple,
            &TargetMachine::get_host_cpu_name().to_string(),
            &TargetMachine::get_host_cpu_features().to_string(),
            OptimizationLevel::Aggressive,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| CodegenError::Emit("no target machine".into()))?;

    std::fs::create_dir_all(out_dir)?;
    let obj = out_dir.join("model.o");
    tm.write_to_file(module.raw_module(), FileType::Object, &obj)
        .map_err(|e| CodegenError::Emit(e.to_string()))?;

    let so = out_dir.join("model.so");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(cc)
        .arg("-shared")
        .arg("-o")
        .arg(&so)
        .arg(&obj)
        .status()?;
    if !status.success() {
        return Err(CodegenError::Link(format!("linker exited {status}")));
    }

    std::fs::write(out_dir.join("weights.bin"), &plan.weights.image)?;
    let meta = build_meta(desc, &plan, opts, Vec::new());
    std::fs::write(out_dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;

    Ok(Artifact {
        dir: out_dir.to_path_buf(),
    })
}

/// Assemble the sidecar [`Meta`] from the model description and plan. Hash
/// fields are intentionally left empty — see the module doc comment.
fn build_meta(
    desc: &inferno_formats::ModelDesc,
    plan: &inferno_plan::Plan,
    opts: &CompileOptions,
    profile_slots: Vec<String>,
) -> Meta {
    Meta {
        model_hash: String::new(),
        target_hash: String::new(),
        weights_hash: String::new(),
        inferno_version: env!("CARGO_PKG_VERSION").to_string(),
        vocab: desc.hyperparams.vocab_size as usize,
        n_layers: desc.hyperparams.n_layers as usize,
        // The quantized-activation scratch region lives *inside* the arena
        // buffer, immediately after the f32 arena (`act_scratch_off ==
        // total_f32 * 4` bytes). The caller allocates a single arena of
        // `arena_f32` f32s, so it must cover both regions or the GEMV
        // activation-quantize writes out of bounds.
        arena_f32: plan.arena.total_f32 + plan.arena.act_scratch_bytes.div_ceil(4),
        kv_total_bytes: plan.kv.total_bytes,
        max_seq_len: plan.max_seq_len,
        entry_prefill: "prefill".to_string(),
        entry_decode: "decode_step".to_string(),
        prefill_tile: opts.prefill_tile,
        profile_slots,
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
    fn compile_writes_three_artifact_files() {
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let graph = build_graph(&desc).unwrap();
        let target = TargetDesc::detect().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let art = compile(
            &desc,
            &graph,
            &target,
            64,
            &CompileOptions::default(),
            tmp.path(),
        )
        .unwrap();
        assert!(art.dir.join("model.so").exists());
        assert!(art.dir.join("weights.bin").exists());
        assert!(art.dir.join("meta.json").exists());
        // weights.bin matches the plan image length.
        let wb = std::fs::metadata(art.dir.join("weights.bin"))
            .unwrap()
            .len() as usize;
        let plan = inferno_plan::plan(
            &desc,
            &graph,
            &target,
            64,
            CompileOptions::default().prefill_tile,
        )
        .unwrap();
        assert_eq!(wb, plan.weights.image.len());
    }
}
