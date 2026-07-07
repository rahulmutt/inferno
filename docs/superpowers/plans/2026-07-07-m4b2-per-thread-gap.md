# M4b.2 — Per-Thread Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a built-in per-op profiler to the compiled path, then close the single-thread prefill gap vs llama.cpp by batching prefill into GEMM tiles (M-loop kernels that read each weight strip once per tile instead of once per token), keeping output bit-identical.

**Architecture:** Two workstreams behind one compile-time options struct. (1) A profiler: when compiled with `profile`, codegen wraps each lowered op with `@llvm.readcyclecounter()` reads that accumulate into a module-global `[N x i64]` counter array, resolved by the host after `dlopen`; the CLI prints per-op time / wall-share / GB/s. (2) Prefill tiles: three new batched `gemm_*_rs8` kernels (scalar + AVX2) whose per-output-element dot order is identical to the GEMV kernels, dispatched through a new `inferno_par_gemm` pool wrapper; `lower_prefill` runs the forward pass a tile of `T` tokens at a time, calling GEMM once per matmul while elementwise/attention ops loop over the tile's rows. Decode is untouched.

**Tech Stack:** Rust (edition 2024, workspace), std-only threading (no rayon), inkwell/LLVM 18 for codegen, cargo-nextest + insta + proptest for tests, criterion for kernel benches.

**Spec:** `docs/superpowers/specs/2026-07-07-m4b2-per-thread-gap-design.md` — read it before starting.

## Global Constraints

- **Bit-identity is a hard contract.** `gemm(m=1)` must equal `gemv` bitwise (`to_bits()`); scalar and AVX2 GEMM must be bitwise-equal; GEMM row-range partitioning must be bit-stable; output must not change with thread count or tile size `T`. Every such test asserts **exact** equality, never tolerance.
- **The compiled-vs-interpreter differential tolerances are frozen.** `inferno-graph/src/tolerance.rs` (`logits_abs_tol`, `gemv_rel_tol`, `LOGIT_TIE_EPSILON`) must not change in this plan. If a differential test goes red, the code is wrong — never loosen a tolerance to make it green. (F16 KV, the one lever that would touch tolerances, is out of scope — it is a later spec amendment.)
- **Kernels stay single-threaded; the caller partitions.** New GEMM kernels take a row range and never spawn threads. `inferno-pool` (`inferno_par_gemm`) is the only partitioning caller. `inferno-kernels`, `inferno-core`, and `inferno-pool` are the only crates allowed `unsafe`; each opts out of the workspace `unsafe_code = "deny"` with its own `[lints.rust]` table and denies `unsafe_op_in_unsafe_fn`. Every `unsafe` block needs a `// SAFETY:` comment.
- **Profiling is a compile-time variant.** A profiled `model.so` is a distinct artifact: `profile` (and `prefill_tile`) are part of the cache key via `CompileOptions`. `--profile` never gates CI; it is a self-measurement tool. Logits must be **bit-identical** with and without `--profile` (the instrumentation only reads clocks).
- **Prefill tile size** is a planner constant `PREFILL_TILE` (default **64**), carried in `CompileOptions.prefill_tile` and the cache key. The final ragged tile runs the same code with a runtime `m ≤ T`.
- **No new external dependencies.** std-only throughout; `@llvm.readcyclecounter` is an LLVM intrinsic (no crate).
- **No CI perf gates** (AGENTS.md). End-to-end numbers come only from the manual `mise run bench` protocol on quiet hardware inside the devenv shell; record them in the spec's Amendments and never edit a recorded data point. The nightly `bench-compiled` speedup gate stays pinned at `--threads 1`.
- **`HOST_ABI_VERSION`** (`inferno-codegen/src/lib.rs`) must be bumped when the emitted host-call shape changes (this plan adds `inferno_par_gemm` and the profiler global), so stale cached `model.so`s recompile.
- Run `mise run lint && mise run test` before every commit; both must be clean. Review any insta snapshot with `cargo insta review` — never blind-accept.

## File Structure

- `crates/inferno-codegen/src/emit.rs` — new `CompileOptions { profile, prefill_tile }`; `compile()` takes it; `Meta` gains `prefill_tile` + `profile_slots`.
- `crates/inferno-codegen/src/profile.rs` *(new)* — pure slot-assignment pass: `LoopIr` → ordered `Vec<String>` op-slot labels + `HashMap<String,usize>`.
- `crates/inferno-codegen/src/llvm/mod.rs` — declare `@llvm.readcyclecounter`, the `inferno_prof_counters` global, and `inferno_par_gemm`.
- `crates/inferno-codegen/src/llvm/ops.rs` — profiler wrapping; `lower_prefill` tiling; `lower_gemv` → batched GEMM emission via `inferno_par_gemm`.
- `crates/inferno-codegen/src/loopir.rs` — `gemm_symbol()` alongside `gemv_symbol()`.
- `crates/inferno-kernels/src/{q8_0,f32k,q4_k}.rs` — `gemm_*_rs8` (scalar + AVX2) siblings of the GEMV kernels.
- `crates/inferno-kernels/src/registry.rs` — `KernelSet::gemm()` safe wrapper; GEMM fn pointers.
- `crates/inferno-kernels/src/lib.rs` — re-export the six GEMM symbols.
- `crates/inferno-pool/src/pool.rs`, `.../lib.rs` — `GemmFn`, `Pool::par_gemm`, `inferno_par_gemm`.
- `crates/inferno-plan/src/memory.rs`, `.../lib.rs`, `.../plan.rs` — `act_scratch` ×`prefill_tile`; thread `prefill_tile` through `plan()`.
- `crates/inferno-core/src/{cache,artifact,lib,backend}.rs` — `CompileOptions` in the cache key + loader + `Engine`; profiled-artifact counter access; retain `inferno_par_gemm`.
- `crates/inferno-kernels/tests/rig.rs` — GEMM rig (gemm≡gemv, ISA bit-equal, range partition).
- `crates/inferno-pool/tests/par_rig.rs` — `par_gemm` bit-identity across thread counts.
- `crates/inferno-codegen/tests/differential.rs` — tiling + tile-size-invariance under the correctness gate.
- `crates/inferno-core/tests/artifact.rs` — profiled artifact loads and counters populate.
- `cli/src/{main,run,bench,profile.rs}` — `--profile` flag + table output.
- `crates/inferno-kernels/benches/gemv.rs` — GEMM criterion benches vs ggml.

---

## Phase 1 — Profiler

### Task 1: `CompileOptions` + thread it through the cache key and compile

**Files:**
- Modify: `crates/inferno-codegen/src/emit.rs`
- Modify: `crates/inferno-codegen/src/lib.rs`
- Modify: `crates/inferno-codegen/src/llvm/mod.rs` (signature of `build_full_module` call is in `ops.rs`; only `emit.rs` calls it — see Task 3)
- Modify: `crates/inferno-plan/src/lib.rs`, `crates/inferno-plan/src/memory.rs`, `crates/inferno-plan/src/plan.rs`
- Modify: `crates/inferno-core/src/cache.rs`, `crates/inferno-core/src/artifact.rs`, `crates/inferno-core/src/lib.rs`

**Interfaces:**
- Produces: `inferno_codegen::CompileOptions { pub profile: bool, pub prefill_tile: usize }` with `Default` (`profile: false`, `prefill_tile: PREFILL_TILE`); `pub const inferno_codegen::PREFILL_TILE: usize = 64`.
- Produces: `compile(desc, graph, target, max_seq_len, opts: &CompileOptions, out_dir) -> Result<Artifact>` (new `opts` param before `out_dir`).
- Produces: `inferno_plan::plan(desc, graph, target, max_seq_len, prefill_tile: usize) -> Result<Plan>` (new trailing param).
- Produces: `inferno_core::cache_key(model, target, max_seq_len, opts: &CompileOptions) -> Result<String>`; `Artifact::load_or_compile(model, target, max_seq_len, opts: &CompileOptions)`.

This task is a mechanical widening: introduce the options struct and thread it, planning `act_scratch` unchanged for now (Task 8 multiplies it by `prefill_tile`) and codegen ignoring `profile` for now (Task 3 consumes it). Keeping the full struct in one task avoids editing every call site twice.

- [ ] **Step 1: Add `PREFILL_TILE` and `CompileOptions`**

In `crates/inferno-codegen/src/lib.rs`, below `HOST_ABI_VERSION`, bump the version and add the constant:

```rust
/// Version of the host-symbol surface generated code links against
/// (kernel symbols + `inferno_par_gemv` + `inferno_par_gemm` + the profiler
/// global). Folded into `inferno-core`'s artifact cache key. "3" = M4b.2's
/// GEMM dispatch + optional profiling (v2 was M4b.1's `inferno_par_gemv`).
pub const HOST_ABI_VERSION: &str = "3";

/// Default prefill tile length (tokens per batched forward pass). Part of
/// `CompileOptions` and the artifact cache key.
pub const PREFILL_TILE: usize = 64;
```

In `crates/inferno-codegen/src/emit.rs`, add near the top (after the `use` block):

```rust
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
```

Re-export from `lib.rs`:

```rust
pub use emit::{Artifact, CompileOptions, Meta, compile};
```

- [ ] **Step 2: Add `prefill_tile` + `profile_slots` to `Meta`; thread `opts` into `compile()`**

In `emit.rs`, extend `Meta` (add the two fields at the end so existing field order is preserved):

```rust
    pub entry_prefill: String,
    pub entry_decode: String,
    /// Prefill tile length this artifact was compiled for (Task 7).
    pub prefill_tile: usize,
    /// Per-op profiler slot labels, slot index = position (empty if this
    /// artifact was compiled without `profile`). Task 3 populates it.
    #[serde(default)]
    pub profile_slots: Vec<String>,
```

Change `compile()`'s signature and body:

```rust
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
    // Task 3 makes build_full_module consume `opts` + return slot labels;
    // for now it ignores profiling and slots stay empty.
    let module = crate::llvm::build_full_module(&ctx, &plan, graph, desc)?;
    module.verify()?;
    // ... unchanged object emit/link ...
    std::fs::write(out_dir.join("weights.bin"), &plan.weights.image)?;
    let meta = build_meta(desc, &plan, opts, Vec::new());
    std::fs::write(out_dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
    Ok(Artifact { dir: out_dir.to_path_buf() })
}
```

Update `build_meta` to take `opts` + slots and set the new fields:

```rust
fn build_meta(
    desc: &inferno_formats::ModelDesc,
    plan: &inferno_plan::Plan,
    opts: &CompileOptions,
    profile_slots: Vec<String>,
) -> Meta {
    Meta {
        // ... unchanged fields ...
        entry_prefill: "prefill".to_string(),
        entry_decode: "decode_step".to_string(),
        prefill_tile: opts.prefill_tile,
        profile_slots,
    }
}
```

Update the `compile_writes_three_artifact_files` test in this file to pass `&CompileOptions::default()` and the two `inferno_plan::plan(...)` calls to pass `CompileOptions::default().prefill_tile`.

- [ ] **Step 3: Thread `prefill_tile` through `inferno_plan::plan`**

In `crates/inferno-plan/src/lib.rs`:

```rust
pub fn plan(
    desc: &ModelDesc,
    graph: &Graph,
    target: &TargetDesc,
    max_seq_len: usize,
    prefill_tile: usize,
) -> Result<Plan> {
    let islands = island::partition(graph);
    let weights = weights::build_weight_image(desc, graph, target)?;
    let arena = memory::plan_arena(graph, &weights, max_seq_len, prefill_tile)?;
    let kv = kv::plan_kv(graph, max_seq_len)?;
    Ok(Plan { islands, weights, arena, kv, max_seq_len })
}
```

In `crates/inferno-plan/src/memory.rs`, change `plan_arena`'s signature to accept `prefill_tile: usize` and (for now) keep behavior identical — the ×T multiply is Task 8. Add the param and thread it unused with a `let _ = prefill_tile;` **only if** Task 8 is not done in the same session; otherwise implement Task 8's multiply here. To avoid an unused-param lint in the interim, prefer doing Step 3 and Task 8's Step together. Update the three `plan_arena(&graph, &weights, 128)` test calls in `memory.rs` and the `plan(...)` calls in `plan.rs` (add `, 64`) and `crate::plan(&desc, &graph, &target, 64, 64)` where they appear.

- [ ] **Step 4: Thread `opts` into the cache key and loader**

In `crates/inferno-core/src/cache.rs`, change `cache_key` to fold the options in:

```rust
pub fn cache_key(
    model_path: &Path,
    target: &TargetDesc,
    max_seq_len: usize,
    opts: &inferno_codegen::CompileOptions,
) -> Result<String> {
    let model_bytes = read_model_bytes(model_path)?;
    let mut h = Sha256::new();
    h.update(content_hash(&model_bytes).as_bytes());
    h.update(format!("{target:?}").as_bytes());
    h.update((max_seq_len as u64).to_le_bytes());
    h.update(env!("CARGO_PKG_VERSION").as_bytes());
    h.update(inferno_codegen::HOST_ABI_VERSION.as_bytes());
    // Profiling and tile size change the emitted artifact.
    h.update([opts.profile as u8]);
    h.update((opts.prefill_tile as u64).to_le_bytes());
    Ok(format!("{:x}", h.finalize()))
}
```

Update `key_is_stable_and_input_sensitive` to pass `&CompileOptions::default()`, and add an assertion that toggling `profile` changes the key:

```rust
    let k_prof = cache_key(m, &t, 64, &inferno_codegen::CompileOptions { profile: true, prefill_tile: 64 }).unwrap();
    assert_ne!(k1, k_prof); // profiling is part of the key
```

In `crates/inferno-core/src/artifact.rs`, change `load_or_compile` and `compile_and_publish` to take `opts: &CompileOptions`, pass it to `cache_key` and to `inferno_codegen::compile(&desc, &graph, target, max_seq_len, opts, &staging)`.

In `crates/inferno-core/src/lib.rs`, give `Engine` an `opts: CompileOptions` field, add a setter, and pass it through `compiled_backend()` and `cache_dir()`:

```rust
pub struct Engine {
    model: PathBuf,
    target: TargetDesc,
    max_seq_len: usize,
    threads: usize,
    opts: inferno_codegen::CompileOptions,
}
```

`Engine::load` initializes `opts: CompileOptions::default()`. Add:

```rust
    /// Enable per-op profiling for artifacts this engine builds (distinct
    /// cache entry). Off by default.
    pub fn set_profile(&mut self, on: bool) {
        self.opts.profile = on;
    }

    /// Prefill tile length for artifacts this engine builds.
    pub fn set_prefill_tile(&mut self, t: usize) {
        self.opts.prefill_tile = t.max(1);
    }
```

`compiled_backend()` calls `Artifact::load_or_compile(&self.model, &self.target, self.max_seq_len, &self.opts)`; `cache_dir()` calls `cache_key(&self.model, &self.target, self.max_seq_len, &self.opts)`.

- [ ] **Step 5: Fix remaining call sites, then lint + test**

Update every other `plan(...)` / `compile(...)` / `cache_key(...)` call site the compiler flags: `crates/inferno-codegen/src/llvm/mod.rs` test (`inferno_plan::plan(&desc, &graph, &target, 64, 64)`), `crates/inferno-codegen/src/loopir.rs` test, `crates/inferno-codegen/tests/differential.rs` (`compile(&desc, &graph, &target, 64, &CompileOptions::default(), tmp.path())`), `crates/inferno-core/tests/artifact.rs` and `backend.rs`, and `cli/src/compile.rs`/`run.rs`/`diff.rs`/`bench.rs` (any `Engine`/`cache_key` use compiles unchanged since defaults are internal).

Run: `mise run lint && mise run test`
Expected: clean. The plan snapshot (`inferno_plan__plan__tests__plan_dump_gguf.snap`) is unchanged (act_scratch not yet scaled).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(codegen,core,plan): CompileOptions (profile + prefill_tile) threaded through cache key"
```

---

### Task 2: Profiler slot-assignment pass

**Files:**
- Create: `crates/inferno-codegen/src/profile.rs`
- Modify: `crates/inferno-codegen/src/lib.rs` (add `pub mod profile;`)

**Interfaces:**
- Consumes: `LoopIr` (Task's own crate), `Plan`, `ModelDesc`.
- Produces: `inferno_codegen::profile::assign_slots(loopir: &LoopIr, plan: &Plan, desc: &ModelDesc) -> ProfileSlots` where `ProfileSlots { pub labels: Vec<String>, index: HashMap<String, usize> }` with `pub fn slot(&self, label: &str) -> usize` and `pub fn len(&self) -> usize`.
- Produces: `profile::step_label(step: &Step, plan: &Plan, desc: &ModelDesc) -> String`.

- [ ] **Step 1: Write the failing test**

`crates/inferno-codegen/src/profile.rs`:

```rust
//! Per-op profiler slot assignment (pure, no LLVM). Each lowered `Step`
//! maps to a stable label; matmuls aggregate across layers by their weight
//! tensor's role (the numeric `blk.N.` segment normalized to `*`), so the
//! slot count is op-kind-sized, not per-layer (spec: op-kind totals).

use std::collections::HashMap;

use inferno_formats::ModelDesc;
use inferno_plan::Plan;

use crate::loopir::{LoopIr, Step};

/// Ordered, de-duplicated profiler slots. `labels[i]` names slot `i`.
#[derive(Debug, Clone, Default)]
pub struct ProfileSlots {
    pub labels: Vec<String>,
    index: HashMap<String, usize>,
}

impl ProfileSlots {
    pub fn slot(&self, label: &str) -> usize {
        self.index[label]
    }
    pub fn len(&self) -> usize {
        self.labels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
    fn intern(&mut self, label: String) {
        if !self.index.contains_key(&label) {
            self.index.insert(label.clone(), self.labels.len());
            self.labels.push(label);
        }
    }
}

/// Normalize a weight tensor name so all layers share one slot: any dotted
/// segment that is a bare integer becomes `*` (e.g. `blk.7.attn_q.weight`
/// -> `blk.*.attn_q.weight`).
fn normalize_weight_name(name: &str) -> String {
    name.split('.')
        .map(|seg| if seg.parse::<u64>().is_ok() { "*" } else { seg })
        .collect::<Vec<_>>()
        .join(".")
}

/// The profiler label for one step. Gemv is keyed by normalized weight name
/// (its matmul "site"); every other op by its kind.
pub fn step_label(step: &Step, plan: &Plan, desc: &ModelDesc) -> String {
    match step {
        Step::Gemv { weight, .. } => {
            let ti = plan.weights.weights[*weight].tensor_index;
            format!("matmul:{}", normalize_weight_name(&desc.tensors[ti].name))
        }
        Step::Quantize { .. } => "quantize".into(),
        Step::Bias { .. } => "bias".into(),
        Step::Embed { .. } => "embed".into(),
        Step::RmsNorm { .. } => "rmsnorm".into(),
        Step::Rope { .. } => "rope".into(),
        Step::SwiGlu { .. } => "swiglu".into(),
        Step::Add { .. } => "add".into(),
        Step::Attention { .. } => "attention".into(),
    }
}

/// Assign a slot to every distinct step label, in first-seen program order.
pub fn assign_slots(loopir: &LoopIr, plan: &Plan, desc: &ModelDesc) -> ProfileSlots {
    let mut slots = ProfileSlots::default();
    for island in &loopir.islands {
        for step in &island.steps {
            slots.intern(step_label(step, plan, desc));
        }
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loopir::build_loopir;
    use inferno_formats::load_desc;
    use inferno_graph::build_graph;
    use inferno_target::TargetDesc;
    use std::path::Path;

    #[test]
    fn slots_aggregate_matmuls_across_layers() {
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let graph = build_graph(&desc).unwrap();
        let target = TargetDesc::detect().unwrap();
        let plan = inferno_plan::plan(&desc, &graph, &target, 64, 64).unwrap();
        let lir = build_loopir(&plan, &graph, &desc);
        let slots = assign_slots(&lir, &plan, &desc);
        // Two transformer layers, but matmul slots must not double: every
        // label is unique, and no label contains a bare layer index.
        let unique: std::collections::HashSet<_> = slots.labels.iter().collect();
        assert_eq!(unique.len(), slots.labels.len(), "labels: {:?}", slots.labels);
        assert!(slots.labels.iter().all(|l| !l.contains(".0.") && !l.contains(".1.")));
        // Sanity: the elementwise kinds each collapse to one slot.
        for kind in ["rmsnorm", "rope", "swiglu", "add", "attention"] {
            assert_eq!(slots.labels.iter().filter(|l| *l == kind).count(), 1, "{kind}");
        }
    }

    #[test]
    fn normalize_strips_layer_index() {
        assert_eq!(normalize_weight_name("blk.7.attn_q.weight"), "blk.*.attn_q.weight");
        assert_eq!(normalize_weight_name("output.weight"), "output.weight");
    }
}
```

Add `pub mod profile;` to `crates/inferno-codegen/src/lib.rs`.

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `cargo nextest run -p inferno-codegen profile::`
Expected: PASS (the module compiles and both tests pass — this is pure data, written complete above).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(codegen): profiler slot-assignment pass (op-kind + matmul-site labels)"
```

---

### Task 3: Emit profiler instrumentation in codegen

**Files:**
- Modify: `crates/inferno-codegen/src/llvm/mod.rs`
- Modify: `crates/inferno-codegen/src/llvm/ops.rs`
- Modify: `crates/inferno-codegen/src/emit.rs`

**Interfaces:**
- Consumes: `ProfileSlots` (Task 2), `CompileOptions` (Task 1).
- Produces: `build_full_module(ctx, plan, graph, desc, opts: &CompileOptions, slots: &ProfileSlots) -> Result<LlvmModule>`; when `opts.profile`, the module exports a global `inferno_prof_counters : [slots.len() x i64]` (zero-init, external linkage) and each op accumulates `readcyclecounter` deltas into its slot.
- Produces: `compile()` now computes `assign_slots(...)`, passes it, and writes `slots.labels` into `Meta.profile_slots`.

- [ ] **Step 1: Declare the intrinsic + counters global**

In `crates/inferno-codegen/src/llvm/mod.rs`, add a method to `LlvmModule`:

```rust
    /// Emit the profiler counter global `inferno_prof_counters : [n x i64]`
    /// (zero-initialized, external linkage so the host resolves it after
    /// `dlopen`). No-op when `n == 0`. Returns the global's pointer.
    pub(crate) fn declare_prof_counters(&self, n: usize) -> Option<inkwell::values::PointerValue<'c>> {
        if n == 0 {
            return None;
        }
        let i64_t = self.ctx.i64_type();
        let arr = i64_t.array_type(n as u32);
        let g = self.module.add_global(arr, Some(AddressSpace::default()), "inferno_prof_counters");
        g.set_linkage(Linkage::External);
        g.set_initializer(&arr.const_zero());
        Some(g.as_pointer_value())
    }
```

- [ ] **Step 2: Thread profiling into `Codegen` and wrap each step**

In `ops.rs`, extend `Codegen` with three fields and populate them in `new`. Add to the struct:

```rust
    /// Profiler counter array base (`[N x i64]`), or None when not profiling.
    prof_counters: Option<PointerValue<'c>>,
    /// step-label -> slot index (empty when not profiling).
    prof_slots: HashMap<String, usize>,
    readcyc_fn: FunctionValue<'c>,
```

In `Codegen::new`, after the other intrinsic decls, add:

```rust
            readcyc_fn: Intrinsic::find("llvm.readcyclecounter")
                .expect("readcyclecounter intrinsic")
                .get_declaration(module, &[])
                .expect("readcyclecounter declaration"),
            prof_counters: None,
            prof_slots: HashMap::new(),
```

Change `build_full_module` to accept `opts` + `slots`, declare the global, and set the fields:

```rust
pub fn build_full_module<'c>(
    ctx: &'c Context,
    plan: &Plan,
    graph: &Graph,
    desc: &ModelDesc,
    opts: &crate::CompileOptions,
    slots: &crate::profile::ProfileSlots,
) -> Result<LlvmModule<'c>> {
    let lm = LlvmModule::new(ctx, "model");
    lm.declare_kernels();
    let prof = if opts.profile { lm.declare_prof_counters(slots.len()) } else { None };
    let (prefill, decode) = lm.declare_entry_points();

    let mut cg = Codegen::new(ctx, lm.module(), plan, graph, desc);
    cg.prof_counters = prof;
    if opts.profile {
        cg.prof_slots = slots.index_map();
    }
    let loopir = build_loopir(plan, graph, desc);
    cg.lower_prefill(prefill, &loopir);
    cg.lower_decode(decode, &loopir);
    Ok(lm)
}
```

(Add `pub(crate) fn index_map(&self) -> HashMap<String, usize> { self.index.clone() }` to `ProfileSlots` in `profile.rs`, and make its `index` field `pub(crate)`.) Note `Codegen::new` must return a value bound as `mut`.

Add i64 load/store helpers and the wrapper near the other helpers in `ops.rs`:

```rust
    fn load_i64(&self, ptr: PointerValue<'c>) -> IntValue<'c> {
        self.builder.build_load(self.i64_t, ptr, "ld64").unwrap().into_int_value()
    }
    fn readcyc(&self) -> IntValue<'c> {
        self.builder
            .build_call(self.readcyc_fn, &[], "rdtsc")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value()
    }

    /// Run `emit` (one op's lowering) bracketed by cycle-counter reads that
    /// accumulate `t1 - t0` into this op's profiler slot. When not profiling
    /// (`prof_counters` is None) it just runs `emit` — zero overhead, and the
    /// emitted math is byte-for-byte identical (readcyclecounter is pure and
    /// only reads a clock). The accumulation runs on the entry-point thread
    /// (GEMV/GEMM shards join before the wrapper returns), so the plain
    /// load/add/store needs no atomics; profiled artifacts are driven by the
    /// single-threaded CLI generate loop.
    fn profiled(&self, label: &str, emit: impl FnOnce(&Self)) {
        let Some(base) = self.prof_counters else {
            return emit(self);
        };
        let slot = self.prof_slots[label];
        let t0 = self.readcyc();
        emit(self);
        let t1 = self.readcyc();
        let delta = self.builder.build_int_sub(t1, t0, "cyc").unwrap();
        let p = self.byte_ptr(base, self.const_i64((slot * 8) as u64));
        let cur = self.load_i64(p);
        let next = self.builder.build_int_add(cur, delta, "acc64").unwrap();
        self.builder.build_store(p, next).unwrap();
    }
```

Change `lower_body` (and, after Task 7, `lower_tile`) to bracket each step. The label is computed with the same `step_label` used for slot assignment:

```rust
    fn lower_body(&self, loopir: &LoopIr, frame: &Frame<'c>) {
        for island in &loopir.islands {
            for step in &island.steps {
                let label = crate::profile::step_label(step, self.plan, self.desc);
                self.profiled(&label, |cg| cg.lower_step(frame, step));
            }
        }
    }
```

- [ ] **Step 3: Compute slots in `compile()` and record them in `Meta`**

In `emit.rs`, change `compile()`:

```rust
    let plan = inferno_plan::plan(desc, graph, target, max_seq_len, opts.prefill_tile)?;
    let loopir_slots = if opts.profile {
        let graph_lir = crate::loopir::build_loopir(&plan, graph, desc);
        crate::profile::assign_slots(&graph_lir, &plan, desc)
    } else {
        crate::profile::ProfileSlots::default()
    };
    let ctx = Context::create();
    let module = crate::llvm::build_full_module(&ctx, &plan, graph, desc, opts, &loopir_slots)?;
    module.verify()?;
    // ... object emit/link unchanged ...
    let meta = build_meta(desc, &plan, opts, loopir_slots.labels.clone());
```

- [ ] **Step 4: Test — the profiled module verifies and stays bit-identical**

Add to `crates/inferno-codegen/src/llvm/mod.rs` tests:

```rust
    #[test]
    fn profiled_module_verifies_and_exports_counters() {
        use inferno_formats::load_desc;
        use inferno_graph::build_graph;
        use inferno_target::TargetDesc;
        use std::path::Path;
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let graph = build_graph(&desc).unwrap();
        let target = TargetDesc::detect().unwrap();
        let plan = inferno_plan::plan(&desc, &graph, &target, 64, 64).unwrap();
        let lir = crate::loopir::build_loopir(&plan, &graph, &desc);
        let slots = crate::profile::assign_slots(&lir, &plan, &desc);
        let opts = crate::CompileOptions { profile: true, prefill_tile: 64 };
        let ctx = Context::create();
        let m = super::build_full_module(&ctx, &plan, &graph, &desc, &opts, &slots).unwrap();
        assert!(m.verify().is_ok(), "{}", m.print_to_string());
        let ir = m.print_to_string();
        assert!(ir.contains("inferno_prof_counters"));
        assert!(ir.contains("readcyclecounter"));
    }
```

Run: `cargo nextest run -p inferno-codegen`
Expected: PASS. The existing `lowered_module_verifies_on_tiny` (unprofiled) call site now passes `&CompileOptions::default(), &ProfileSlots::default()`.

- [ ] **Step 5: Bit-identity under the differential — add a profiled arm**

In `crates/inferno-codegen/tests/differential.rs`, add a test that compiles the same fixture with and without `profile` and asserts the last-token logits are **bitwise** equal (not just within tolerance). Reuse `run_compiled`; add a helper that compiles with a given `CompileOptions` into its own tempdir:

```rust
#[test]
fn profiling_does_not_change_logits() {
    use inferno_codegen::CompileOptions;
    let fixture = "../inferno-formats/tests/fixtures/tiny.gguf";
    let desc = load_desc(Path::new(fixture)).unwrap();
    let graph = build_graph(&desc).unwrap();
    let target = TargetDesc::detect().unwrap();
    let tokens: Vec<u32> = vec![1, 5, 9, 3];

    let plain_dir = tempfile::tempdir().unwrap();
    let a = compile(&desc, &graph, &target, 64, &CompileOptions::default(), plain_dir.path()).unwrap();
    let prof_dir = tempfile::tempdir().unwrap();
    let b = compile(&desc, &graph, &target, 64,
        &CompileOptions { profile: true, prefill_tile: 64 }, prof_dir.path()).unwrap();

    let read_meta = |art: &inferno_codegen::Artifact| -> Meta {
        serde_json::from_slice(&std::fs::read(art.dir.join("meta.json")).unwrap()).unwrap()
    };
    let la = unsafe { run_compiled(&a.dir, &tokens, &read_meta(&a)) };
    let lb = unsafe { run_compiled(&b.dir, &tokens, &read_meta(&b)) };
    for (i, (x, y)) in la.iter().zip(&lb).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "logit {i} differs with --profile");
    }
}
```

Run: `cargo nextest run -p inferno-codegen --test differential`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(codegen): emit per-op readcyclecounter profiler behind CompileOptions.profile"
```

---

### Task 4: `--profile` CLI + baseline profile recording

**Files:**
- Modify: `crates/inferno-core/src/artifact.rs` (counter access on a profiled artifact)
- Modify: `crates/inferno-core/src/backend.rs` (expose profile snapshot/reset)
- Create: `cli/src/profile.rs`
- Modify: `cli/src/main.rs`, `cli/src/run.rs`

**Interfaces:**
- Consumes: `Meta.profile_slots`, the `inferno_prof_counters` symbol.
- Produces: `Artifact::profile_snapshot(&self) -> Option<Vec<u64>>`, `Artifact::profile_reset(&self)`, `Artifact::profile_slots(&self) -> &[String]`.
- Produces: `CompiledBackend::profile_snapshot(&self)/profile_reset(&self)/profile_slots(&self)` delegating to the artifact.
- Produces: `cli::profile::render(slots, prefill_counts, decode_counts, matmul_bytes) -> String`.

- [ ] **Step 1: Resolve + read the counters symbol on the artifact**

In `crates/inferno-core/src/artifact.rs`, add a field to `Artifact`:

```rust
    /// Base of the profiled `model.so`'s `[N x i64]` counter array, resolved
    /// at load time when `meta.profile_slots` is non-empty; None otherwise.
    prof_counters: Option<NonNull<u64>>,
```

In `load_from`, after resolving the entry points, resolve the optional global:

```rust
        // SAFETY: a profiled artifact exports `inferno_prof_counters` as
        // `[N x i64]` with N == meta.profile_slots.len(); we copy out the raw
        // base pointer and keep `lib` alive in the returned Artifact.
        let prof_counters = if meta.profile_slots.is_empty() {
            None
        } else {
            let sym: libloading::Symbol<*mut u64> =
                unsafe { lib.get(b"inferno_prof_counters\0") }?;
            NonNull::new(unsafe { *sym })
        };
```

Add the field to the `Artifact { .. }` constructor. Add methods:

```rust
    /// Profiler slot labels (empty unless compiled with `profile`).
    pub fn profile_slots(&self) -> &[String] {
        &self.meta.profile_slots
    }

    /// Current per-slot cycle counters, or None if unprofiled. Reads the raw
    /// `[N x i64]` global the compiled code accumulates into.
    pub fn profile_snapshot(&self) -> Option<Vec<u64>> {
        let base = self.prof_counters?;
        let n = self.meta.profile_slots.len();
        // SAFETY: `base` points at the artifact's live `[n x i64]` global for
        // as long as `self._lib` is alive; we only read it.
        Some(unsafe { std::slice::from_raw_parts(base.as_ptr(), n).to_vec() })
    }

    /// Zero the counters (separates prefill vs decode measurement).
    pub fn profile_reset(&self) {
        if let Some(base) = self.prof_counters {
            let n = self.meta.profile_slots.len();
            // SAFETY: exclusive logical access — the CLI resets between phases
            // while no forward pass is running.
            unsafe { std::ptr::write_bytes(base.as_ptr(), 0, n) };
        }
    }
```

The `unsafe impl Send/Sync for Artifact` comment gains: the raw counters pointer is only touched by the single-threaded CLI profile path.

- [ ] **Step 2: Delegate from `CompiledBackend`**

In `crates/inferno-core/src/backend.rs`, add pass-throughs:

```rust
    pub fn profile_slots(&self) -> &[String] {
        self.artifact.profile_slots()
    }
    pub fn profile_snapshot(&self) -> Option<Vec<u64>> {
        self.artifact.profile_snapshot()
    }
    pub fn profile_reset(&self) {
        self.artifact.profile_reset();
    }
```

(`CompiledBackend.artifact` is private; these methods live in the same module.)

- [ ] **Step 3: The profile table renderer**

`cli/src/profile.rs`:

```rust
//! `--profile` output: per-op cycle totals, wall-clock share, and (for
//! matmul sites) achieved GB/s. Self-measurement only; never a CI gate.

/// Render a profile table. `counts[i]` is slot `i`'s accumulated cycles;
/// `bytes[i]` is the weight bytes touched per matmul slot invocation × the
/// invocation count (0 for non-matmul slots), used for the GB/s column.
/// `secs` is the measured wall-clock for this phase (prefill or decode),
/// used to convert the cycle share into GB/s without knowing the TSC rate.
pub fn render(phase: &str, slots: &[String], counts: &[u64], bytes: &[u64], secs: f64) -> String {
    use std::fmt::Write;
    let total: u64 = counts.iter().sum();
    let mut rows: Vec<usize> = (0..slots.len()).collect();
    rows.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
    let mut s = String::new();
    writeln!(s, "profile [{phase}] {secs:.3}s wall, {total} cyc total").unwrap();
    writeln!(s, "  {:<28} {:>14} {:>7}  {:>10}", "op", "cycles", "share", "GB/s").unwrap();
    for i in rows {
        let share = if total > 0 { counts[i] as f64 / total as f64 } else { 0.0 };
        // Time attributed to this op = its cycle share of the phase wall-clock.
        let op_secs = share * secs;
        let gbps = if bytes[i] > 0 && op_secs > 0.0 {
            bytes[i] as f64 / op_secs / 1e9
        } else {
            0.0
        };
        let gbps_col = if gbps > 0.0 { format!("{gbps:.1}") } else { "-".into() };
        writeln!(s, "  {:<28} {:>14} {:>6.1}%  {:>10}", slots[i], counts[i], share * 100.0, gbps_col).unwrap();
    }
    s
}

#[cfg(test)]
mod tests {
    #[test]
    fn render_sorts_and_computes_share() {
        let slots = vec!["matmul:blk.*.attn_q.weight".to_string(), "rmsnorm".to_string()];
        let out = super::render("decode", &slots, &[300, 100], &[6_000_000, 0], 0.5);
        assert!(out.contains("matmul:blk.*.attn_q.weight"));
        assert!(out.contains("75.0%")); // 300/400
        // matmul row shows a GB/s number; rmsnorm shows '-'.
        let mm_line = out.lines().find(|l| l.contains("attn_q")).unwrap();
        assert!(mm_line.contains('.') && !mm_line.contains(" - "));
    }
}
```

Add `mod profile;` to `cli/src/main.rs`.

- [ ] **Step 4: Wire `--profile` into `inferno run`**

In `cli/src/main.rs`, add `#[arg(long)] profile: bool` to the `Run` variant and pass it to `run::run`. In `cli/src/run.rs`, thread `profile: bool` into `load_compiled`, calling `engine.set_profile(true)` when set. After generation, if profiling, print two tables using `stats.prefill_secs` / `stats.decode_secs`, reading the backend's snapshot. Because `Generator` owns the backend, expose the snapshot through a `Generator` accessor or capture it via a closure; simplest is to compute the matmul-bytes vector from the plan's packed weights.

Concretely, in `run.rs` after a successful `generate`, when `profile` is set, call a helper on the generator's backend. Add to `inferno_runtime::Generator` a passthrough `pub fn backend_profile(&self) -> Option<(Vec<String>, Vec<u64>)>` **only if** the runtime exposes the backend; if not, drive profiling through a dedicated `inferno run --profile` code path that constructs the `CompiledBackend` directly (bypassing `Generator`) and runs one prefill + a fixed decode count, reading `profile_reset`/`profile_snapshot` around each phase. Prefer the dedicated path — it keeps `Generator` unchanged:

```rust
// In run.rs, when `profile` is set, take a measurement path instead of the
// normal streaming generate: prefill the prompt, snapshot, reset, decode
// `max_tokens` greedily, snapshot again, and print both tables.
```

Record the matmul-bytes vector by summing `plan.weights.weights[w].len` for the slot each matmul maps to, times its per-phase invocation count (prompt_len for prefill, generated for decode). Compute it from `Engine`'s plan via a new `Engine::profile_matmul_bytes(&self, slots) -> Vec<u64>` helper, or approximate with the weight image bytes per slot × invocations. Keep this helper small and documented.

- [ ] **Step 5: Manual — record the baseline profile in the spec**

This is the Phase-1 deliverable that unblocks the decode attribution fork. Inside the devenv shell on the quiet machine, against the pinned nightly model:

```bash
mise run -- # enter devenv, then:
cargo run --release -p inferno -- run \
  /home/dev/.cache/inferno-tests/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "$(head -c 2048 /dev/urandom | base64)" --max-tokens 64 --threads 1 --profile
```

Paste both tables (prefill + decode) verbatim into a new dated entry under **## Amendments** in `docs/superpowers/specs/2026-07-07-m4b2-per-thread-gap-design.md`, noting the commit, model, and that this is the pre-optimization baseline. Do not interpret it into a lever yet — that is the Phase-5 amendment.

- [ ] **Step 6: Lint, test, commit**

Run: `mise run lint && mise run test`

```bash
git add -A
git commit -m "feat(cli,core): inferno run --profile — per-op cycle/GB-s tables; record baseline"
```

---

## Phase 2 — GEMM kernels

### Task 5: Batched `gemm_*_rs8` kernels (scalar + AVX2, three dtypes)

**Files:**
- Modify: `crates/inferno-kernels/src/q8_0.rs`, `crates/inferno-kernels/src/f32k.rs`, `crates/inferno-kernels/src/q4_k.rs`
- Modify: `crates/inferno-kernels/src/lib.rs`
- Modify: `crates/inferno-kernels/tests/rig.rs`

**Interfaces:**
- Produces (per dtype `d` in {`f32`, `q8_0`, `q4_k`}, per isa in {`scalar`, `avx2`}): `unsafe extern "C" fn inferno_gemm_{d}_rs8_{isa}(y: *mut f32, xq: *const u8, w: *const u8, k: usize, m: usize, rows: usize, row_start: usize, row_end: usize)`.
- Semantics: for each token `t in 0..m` and output row `r in row_start..row_end`, `y[t*rows + r] = dot(W[r], activation_t)`, where activation `t` is at `xq + t*act_stride` (`act_stride = q8a_len(k)` / `q8k_len(k)` / `k*4`). The per-(t,r) dot uses the **same block/strip k-order** as the matching GEMV kernel — so `gemm(m=1, rows, 0, rows)` is bitwise-equal to `gemv(0, rows)`.
- Re-exported from `lib.rs`: the six symbols.

The transformation is identical for every kernel: take the GEMV body, add an outer `for b in 0..nb` (weight block) that reads the weight group **once**, and an inner `for t in 0..m` that accumulates each token against that group. Token `t`'s per-row accumulators persist across the `b` loop, so each token sees blocks in order `0..nb` — the GEMV accumulation order.

- [ ] **Step 1: Q8_0 GEMM (the pinned model's dtype) — write the failing rig test first**

In `crates/inferno-kernels/tests/rig.rs`, under the Q8_0 section, add a batched runner and the invariants:

```rust
#[allow(clippy::too_many_arguments)]
fn gemm_q8_0(
    isa: KernelIsa,
    w: &inferno_kernels::AlignedBuf,
    xq_panel: &[u8],
    k: usize,
    m: usize,
    rows: usize,
    range: (usize, usize),
    y: &mut [f32],
) {
    // SAFETY: w is a pack_q8_0_rs8 image for (rows, k); xq_panel is m q8a rows
    // for k, contiguous; y has m*rows elements; range within rows.
    unsafe {
        match isa {
            KernelIsa::Scalar => inferno_kernels::inferno_gemm_q8_0_rs8_scalar(
                y.as_mut_ptr(), xq_panel.as_ptr(), w.as_ptr(), k, m, rows, range.0, range.1),
            KernelIsa::Avx2 => inferno_kernels::inferno_gemm_q8_0_rs8_avx2(
                y.as_mut_ptr(), xq_panel.as_ptr(), w.as_ptr(), k, m, rows, range.0, range.1),
        }
    }
}

proptest! {
    /// gemm(m=1) is bit-identical to gemv over the same range.
    #[test]
    fn q8_0_gemm_m1_equals_gemv(seed in any::<u64>(), rows in 1usize..20, nb in 1usize..5) {
        let k = nb * 32;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 0xa1, k);
        let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
        let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
        for isa in KernelIsa::all_available() {
            let mut yv = vec![f32::NAN; rows];
            gemv_q8_0(isa, &w, &xq, rows, k, (0, rows), &mut yv);
            let mut yg = vec![f32::NAN; rows];
            gemm_q8_0(isa, &w, &xq, k, 1, rows, (0, rows), &mut yg);
            for (i, (a, b)) in yv.iter().zip(&yg).enumerate() {
                prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
            }
        }
    }

    /// Each token in an m-row panel matches an independent gemv on that token.
    #[test]
    fn q8_0_gemm_rows_match_per_token_gemv(seed in any::<u64>(), rows in 1usize..16, nb in 1usize..4, m in 1usize..6) {
        let k = nb * 32;
        let vals = pseudo(seed, rows * k);
        let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
        let mut panel = Vec::new();
        let mut per_token = Vec::new();
        for t in 0..m {
            let x = pseudo(seed ^ (0x100 + t as u64), k);
            let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
            panel.extend_from_slice(&xq);
            let mut yv = vec![f32::NAN; rows];
            gemv_q8_0(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut yv);
            per_token.push(yv);
        }
        for isa in KernelIsa::all_available() {
            let mut yg = vec![f32::NAN; m * rows];
            gemm_q8_0(isa, &w, &panel, k, m, rows, (0, rows), &mut yg);
            for t in 0..m {
                for r in 0..rows {
                    prop_assert_eq!(yg[t * rows + r].to_bits(), per_token[t][r].to_bits(), "t{} r{}", t, r);
                }
            }
        }
    }

    /// Row-range partitioning is bit-stable (the property par_gemm relies on).
    #[test]
    fn q8_0_gemm_range_partition_bitwise(seed in any::<u64>(), rows in 2usize..24, nb in 1usize..4, m in 1usize..4) {
        let k = nb * 32;
        let split = (seed % rows as u64) as usize;
        let vals = pseudo(seed, rows * k);
        let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
        let mut panel = Vec::new();
        for t in 0..m {
            let x = pseudo(seed ^ (0x200 + t as u64), k);
            panel.extend_from_slice(&act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap());
        }
        for isa in KernelIsa::all_available() {
            let mut full = vec![f32::NAN; m * rows];
            gemm_q8_0(isa, &w, &panel, k, m, rows, (0, rows), &mut full);
            let mut split_y = vec![f32::NAN; m * rows];
            gemm_q8_0(isa, &w, &panel, k, m, rows, (0, split), &mut split_y);
            gemm_q8_0(isa, &w, &panel, k, m, rows, (split, rows), &mut split_y);
            for i in 0..m * rows {
                prop_assert_eq!(full[i].to_bits(), split_y[i].to_bits(), "i {}", i);
            }
        }
    }
}
```

Run: `cargo nextest run -p inferno-kernels q8_0_gemm`
Expected: FAIL to compile (`inferno_gemm_q8_0_rs8_*` not found).

- [ ] **Step 2: Implement Q8_0 GEMM (scalar + AVX2)**

In `crates/inferno-kernels/src/q8_0.rs`, append. The scalar path is the GEMV per-row loop with an m-inner loop that shares the loaded weight group:

```rust
/// Batched Q8_0 GEMV (GEMM): `y[t*rows + r] = W[r] · dequant(xq_t)` for every
/// token `t in 0..m` and row `r in row_start..row_end`. Each weight block is
/// read once per batch (outer `b`, inner `t`); per (t,r) the block order is
/// `0..nb`, identical to `inferno_gemv_q8_0_rs8_*`, so `gemm(m=1)` is
/// bitwise-equal to `gemv`.
///
/// # Safety
/// As the GEMV symbol, with: `xq` valid for `m` contiguous q8a rows of `k`
/// (`m * q8a_len(k)` bytes); `y` valid for `m * rows` f32 writes; `m >= 1`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemm_q8_0_rs8_scalar(
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    m: usize,
    rows: usize,
    row_start: usize,
    row_end: usize,
) {
    let nb = k / WBLOCK;
    let act = Q8A_BLOCK_BYTES * nb; // per-token activation stride
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        // One accumulator per token; blocks visited in order → gemv order.
        let mut acc = vec![0f32; m];
        for b in 0..nb {
            let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let qw = unsafe { g.add(32 + lane * WBLOCK) };
            for (t, at) in acc.iter_mut().enumerate() {
                let xb = unsafe { xq.add(t * act + b * Q8A_BLOCK_BYTES) };
                let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                let qx = unsafe { xb.add(4) };
                let mut isum = 0i32;
                for i in 0..WBLOCK {
                    let a = i32::from(unsafe { qw.add(i).cast::<i8>().read() });
                    let bb = i32::from(unsafe { qx.add(i).cast::<i8>().read() });
                    isum += a * bb;
                }
                *at = (dw * dx).mul_add(isum as f32, *at);
            }
        }
        for (t, at) in acc.iter().enumerate() {
            unsafe { y.add(t * rows + r).write(*at) };
        }
    }
}

/// # Safety
/// As [`inferno_gemm_q8_0_rs8_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemm_q8_0_rs8_avx2(
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    m: usize,
    rows: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let nb = k / WBLOCK;
    let act = Q8A_BLOCK_BYTES * nb;
    let ones = _mm256_set1_epi16(1);
    let mut r = row_start;
    while r < row_end {
        let strip = r / STRIP;
        let lane0 = r - strip * STRIP;
        // Full-strip fast path: 8 rows lane-parallel, one acc per token.
        if lane0 == 0 && r + STRIP <= row_end {
            let mut acc = vec![_mm256_setzero_ps(); m];
            for b in 0..nb {
                let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                let qs = unsafe { g.add(32) };
                // Weight group's 8 per-row scales (lane = row), loaded once.
                let dw = unsafe { _mm256_load_ps(g.cast()) };
                for (t, at) in acc.iter_mut().enumerate() {
                    let xb = unsafe { xq.add(t * act + b * Q8A_BLOCK_BYTES) };
                    let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                    let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                    let mut p = [_mm256_setzero_si256(); STRIP];
                    for (lane, pl) in p.iter_mut().enumerate() {
                        let wv = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                        let aw = _mm256_sign_epi8(wv, wv);
                        let sx = _mm256_sign_epi8(xv, wv);
                        *pl = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                    }
                    let isum = _mm256_cvtepi32_ps(hsum8_i32(p));
                    let dwdx = _mm256_mul_ps(dw, _mm256_set1_ps(dx));
                    *at = _mm256_fmadd_ps(dwdx, isum, *at);
                }
            }
            for (t, at) in acc.iter().enumerate() {
                unsafe { _mm256_storeu_ps(y.add(t * rows + r), *at) };
            }
            r += STRIP;
            continue;
        }
        // Partial head/tail row: per-row path, one acc per token.
        let lane = lane0;
        let mut acc = vec![0f32; m];
        for b in 0..nb {
            let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let wv = unsafe { _mm256_load_si256(g.add(32 + lane * WBLOCK).cast()) };
            for (t, at) in acc.iter_mut().enumerate() {
                let xb = unsafe { xq.add(t * act + b * Q8A_BLOCK_BYTES) };
                let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                let aw = _mm256_sign_epi8(wv, wv);
                let sx = _mm256_sign_epi8(xv, wv);
                let isum = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones));
                *at = (dw * dx).mul_add(isum as f32, *at);
            }
        }
        for (t, at) in acc.iter().enumerate() {
            unsafe { y.add(t * rows + r).write(*at) };
        }
        r += 1;
    }
}
```

Run: `cargo nextest run -p inferno-kernels q8_0_gemm`
Expected: PASS.

- [ ] **Step 3: F32 GEMM — test then implement**

Add an `f32`-section rig test `f32_gemm_m1_equals_gemv` mirroring Step 1 (use `gemv_f32` / a new `gemm_f32` runner, activation is raw LE f32 so `act_stride = k*4`, panel = `bytemuck_free_cast` of `m*k` f32). Then in `crates/inferno-kernels/src/f32k.rs` append the two symbols; the scalar body wraps `gemv_rows` per token but with weight-reuse via a block-outer loop:

```rust
/// Batched F32 GEMM. Same per-(t,r) fma order as `inferno_gemv_f32_rs8_*`
/// (`gemm(m=1) ≡ gemv`). `xq` is `m` contiguous rows of `k` LE f32.
///
/// # Safety
/// As the F32 GEMV symbol, with `xq` valid for `m*k` f32, `y` for `m*rows`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemm_f32_rs8_scalar(
    y: *mut f32, xq: *const u8, w: *const u8, k: usize,
    m: usize, rows: usize, row_start: usize, row_end: usize,
) {
    let x = xq.cast::<f32>();
    let wf = w.cast::<f32>();
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let base = unsafe { wf.add(strip * k * STRIP + lane) };
        let mut acc = vec![0f32; m];
        for c in 0..k {
            let wv = unsafe { base.add(c * STRIP).read() };
            for (t, at) in acc.iter_mut().enumerate() {
                let xv = unsafe { x.add(t * k + c).read_unaligned() };
                *at = wv.mul_add(xv, *at);
            }
        }
        for (t, at) in acc.iter().enumerate() {
            unsafe { y.add(t * rows + r).write(*at) };
        }
    }
}

/// # Safety
/// As [`inferno_gemm_f32_rs8_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemm_f32_rs8_avx2(
    y: *mut f32, xq: *const u8, w: *const u8, k: usize,
    m: usize, rows: usize, row_start: usize, row_end: usize,
) {
    use std::arch::x86_64::*;
    let x = xq.cast::<f32>();
    let wf = w.cast::<f32>();
    let mut r = row_start;
    let head = row_start.next_multiple_of(STRIP).min(row_end);
    if head > r {
        // Partial head: scalar per-row (bit-identical), one acc per token.
        unsafe { inferno_gemm_f32_rs8_scalar(y, xq, w, k, m, rows, r, head) };
        r = head;
    }
    while r + STRIP <= row_end {
        let base = unsafe { wf.add((r / STRIP) * k * STRIP) };
        let mut acc = vec![_mm256_setzero_ps(); m];
        for c in 0..k {
            let wv = unsafe { _mm256_load_ps(base.add(c * STRIP)) };
            for (t, at) in acc.iter_mut().enumerate() {
                let xv = _mm256_set1_ps(unsafe { x.add(t * k + c).read_unaligned() });
                *at = _mm256_fmadd_ps(wv, xv, *at);
            }
        }
        for (t, at) in acc.iter().enumerate() {
            unsafe { _mm256_storeu_ps(y.add(t * rows + r), *at) };
        }
        r += STRIP;
    }
    if r < row_end {
        unsafe { inferno_gemm_f32_rs8_scalar(y, xq, w, k, m, rows, r, row_end) };
    }
}
```

(The head/tail scalar delegation writes `y[t*rows + r]` for the same `r` sub-range, so its output is disjoint from the strip path's — no double write.)

Run: `cargo nextest run -p inferno-kernels f32_gemm`
Expected: PASS.

- [ ] **Step 4: Q4_K GEMM — test then implement**

Add `q4_k_gemm_m1_equals_gemv` + `q4_k_gemm_rows_match_per_token_gemv` mirroring Step 1 (activation `q8k`, `act_stride = q8k_len(k)`). In `crates/inferno-kernels/src/q4_k.rs`, append `inferno_gemm_q4_k_rs8_{scalar,avx2}` by the same transformation: read the existing `inferno_gemv_q4_k_rs8_{scalar,avx2}` body, hoist the weight super-block loads to a block-outer loop, and add the inner `for t in 0..m` accumulating each token against the loaded super-block, storing `y[t*rows + r]`. Keep the per-(t,r) accumulation order identical to the GEMV kernel. (Q4_K is not the pinned model's dtype and is not on the measured path; correctness parity with GEMV is the bar, not new tuning.)

Run: `cargo nextest run -p inferno-kernels q4_k_gemm`
Expected: PASS.

- [ ] **Step 5: Re-export the six symbols**

In `crates/inferno-kernels/src/lib.rs`:

```rust
pub use f32k::{inferno_gemm_f32_rs8_avx2, inferno_gemm_f32_rs8_scalar, inferno_gemv_f32_rs8_avx2, inferno_gemv_f32_rs8_scalar};
pub use q4_k::{inferno_gemm_q4_k_rs8_avx2, inferno_gemm_q4_k_rs8_scalar, inferno_gemv_q4_k_rs8_avx2, inferno_gemv_q4_k_rs8_scalar};
pub use q8_0::{inferno_gemm_q8_0_rs8_avx2, inferno_gemm_q8_0_rs8_scalar, inferno_gemv_q8_0_rs8_avx2, inferno_gemv_q8_0_rs8_scalar};
```

- [ ] **Step 6: Lint, full test, commit**

Run: `mise run lint && mise run test`
Expected: clean (all GEMM rig tests green; existing GEMV tests unaffected).

```bash
git add -A
git commit -m "feat(kernels): batched gemm_*_rs8 (scalar+avx2, f32/q8_0/q4_k), bit-identical to gemv"
```

---

### Task 6: `KernelSet::gemm` safe wrapper

**Files:**
- Modify: `crates/inferno-kernels/src/registry.rs`

**Interfaces:**
- Consumes: the six GEMM symbols (Task 5).
- Produces: `KernelSet::gemm(&self, y: &mut [f32], xq: &[u8], w: &AlignedBuf, m: usize, rows: usize, k: usize, row_start: usize, row_end: usize) -> Result<()>` — validates lengths/ranges (as `gemv`) plus `m >= 1`, `y.len() == m*rows`, `xq.len() == m * act_len(k)`, then dispatches. Also `KernelSet` gains a private `gemm: GemmFn` field selected in `set()`.

- [ ] **Step 1: Write the failing test**

In `registry.rs` tests, add:

```rust
    #[test]
    fn gemm_wrapper_matches_gemv_and_validates() {
        let (rows, k, m) = (10usize, 64usize, 3usize);
        let s = reference_kernels(&DType::Q8_0).unwrap();
        let vals = pseudo(1, rows * k);
        let file = quant::pack(&DType::Q8_0, &vals).unwrap();
        let w = s.pack(&file, rows, k).unwrap();
        // Panel of m quantized rows.
        let mut panel = Vec::new();
        let mut per_token = Vec::new();
        for t in 0..m {
            let x = pseudo(2 + t as u64, k);
            let xq = s.quantize_row(&x).unwrap();
            panel.extend_from_slice(&xq);
            let mut yv = vec![f32::NAN; rows];
            s.gemv(&mut yv, &xq, &w, rows, k, 0, rows).unwrap();
            per_token.push(yv);
        }
        let mut yg = vec![f32::NAN; m * rows];
        s.gemm(&mut yg, &panel, &w, m, rows, k, 0, rows).unwrap();
        for t in 0..m {
            for r in 0..rows {
                assert_eq!(yg[t * rows + r].to_bits(), per_token[t][r].to_bits(), "t{t} r{r}");
            }
        }
        // Validation: wrong panel length / y length / range.
        assert!(s.gemm(&mut yg, &panel[..panel.len() - 1], &w, m, rows, k, 0, rows).is_err());
        assert!(s.gemm(&mut yg[..m * rows - 1], &panel, &w, m, rows, k, 0, rows).is_err());
        assert!(s.gemm(&mut yg, &panel, &w, m, rows, k, 3, 2).is_err());
        assert!(s.gemm(&mut yg, &panel, &w, 0, rows, k, 0, rows).is_err());
    }
```

Run: `cargo nextest run -p inferno-kernels gemm_wrapper`
Expected: FAIL (`gemm` not found).

- [ ] **Step 2: Add the `GemmFn` type, field, selection, and wrapper**

At the top of `registry.rs`:

```rust
type GemmFn = unsafe extern "C" fn(*mut f32, *const u8, *const u8, usize, usize, usize, usize, usize);
```

Add `gemm: GemmFn,` to `KernelSet`. In `set()`, add to each dtype arm alongside `gemv:`:

```rust
            gemm: match isa {
                KernelIsa::Scalar => f32k::inferno_gemm_f32_rs8_scalar,
                KernelIsa::Avx2 => f32k::inferno_gemm_f32_rs8_avx2,
            },
```

(and the `q8_0::inferno_gemm_q8_0_rs8_*` / `q4_k::inferno_gemm_q4_k_rs8_*` variants in their arms). Add the method:

```rust
    /// Batched GEMV (GEMM): `y[t*rows + r] = W[r]·act_t` for `t in 0..m`,
    /// `r in row_start..row_end`. Validates the same contracts as
    /// [`gemv`](Self::gemv) plus the `m`-row panel/output lengths.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self, y: &mut [f32], xq: &[u8], w: &AlignedBuf,
        m: usize, rows: usize, k: usize, row_start: usize, row_end: usize,
    ) -> Result<()> {
        if m == 0 {
            return Err(KernelError::ZeroRows);
        }
        if k == 0 || !k.is_multiple_of(self.wblock) {
            return Err(KernelError::BadK { k, block: self.wblock });
        }
        if k > crate::MAX_K || rows > crate::MAX_K || m > crate::MAX_K {
            return Err(KernelError::Overflow);
        }
        if y.len() != m * rows {
            return Err(KernelError::SizeMismatch { what: "gemm output (m*rows f32)", got: y.len(), expected: m * rows });
        }
        if row_start > row_end || row_end > rows {
            return Err(KernelError::BadRowRange { row_start, row_end, rows });
        }
        if xq.len() != m * (self.act_len)(k) {
            return Err(KernelError::SizeMismatch { what: "gemm activation panel bytes", got: xq.len(), expected: m * (self.act_len)(k) });
        }
        if w.len() != (self.packed_len)(rows, k) {
            return Err(KernelError::SizeMismatch { what: "packed weight bytes", got: w.len(), expected: (self.packed_len)(rows, k) });
        }
        // SAFETY: every pointer/length/alignment precondition validated above.
        unsafe { (self.gemm)(y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, m, rows, row_start, row_end) };
        Ok(())
    }
```

Run: `cargo nextest run -p inferno-kernels gemm_wrapper`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat(kernels): KernelSet::gemm safe wrapper with panel/range validation"
```

---

## Phase 3 — Prefill tiling

### Task 7: `inferno_par_gemm` pool dispatcher

**Files:**
- Modify: `crates/inferno-pool/src/pool.rs`
- Modify: `crates/inferno-pool/src/lib.rs`
- Modify: `crates/inferno-pool/tests/par_rig.rs`
- Modify: `crates/inferno-core/src/artifact.rs` (retain the new symbol)

**Interfaces:**
- Produces: `inferno_pool::GemmFn = unsafe extern "C" fn(*mut f32, *const u8, *const u8, usize, usize, usize, usize, usize)` (y, xq, w, k, m, rows, row_start, row_end).
- Produces: `Pool::par_gemm(&self, kernel: GemmFn, y, xq, w, k, m, rows)` — splits `0..rows` into the same shard table as `par_gemv`; each shard computes all `m` tokens for its rows. Output bit-identical across thread counts.
- Produces: `#[no_mangle] pub unsafe extern "C" fn inferno_par_gemm(kernel, y, xq, w, k, m, rows)` — the host symbol generated code calls; uncontended-CAS single-dispatcher guard + serial fallback, exactly like `inferno_par_gemv`.

- [ ] **Step 1: Write the failing bit-identity test**

In `crates/inferno-pool/tests/par_rig.rs`, add a `par_gemm` case coercing the real Q8_0 GEMM symbol and asserting the pool result equals a single serial call for several thread counts. (Follow the existing `par_gemv` rig pattern in this file; the panel is `m` quantized rows, `y` is `m*rows`.)

```rust
#[test]
fn par_gemm_bit_identical_across_threads() {
    use inferno_kernels::{act, q8_0, KernelIsa};
    let (rows, k, m) = (129usize, 64usize, 5usize);
    let vals: Vec<f32> = (0..rows * k).map(|i| ((i as f32) * 0.001).sin()).collect();
    let w = q8_0::pack_q8_0_rs8(&inferno_formats::quant::pack(&inferno_formats::DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
    let mut panel = Vec::new();
    for t in 0..m {
        let x: Vec<f32> = (0..k).map(|i| ((i + t) as f32 * 0.01).cos()).collect();
        panel.extend_from_slice(&act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap());
    }
    let kernel: inferno_pool::GemmFn = q8_0::inferno_gemm_q8_0_rs8_avx2;
    // Serial reference (1 lane).
    let mut want = vec![f32::NAN; m * rows];
    // SAFETY: sized buffers; full range.
    unsafe { kernel(want.as_mut_ptr(), panel.as_ptr(), w.as_ptr(), k, m, rows, 0, rows) };
    for threads in [1usize, 2, 3, 8] {
        let pool = inferno_pool::Pool::new(threads);
        let mut got = vec![f32::NAN; m * rows];
        // SAFETY: buffers live for the call; no overlapping dispatch.
        unsafe { pool.par_gemm(kernel, got.as_mut_ptr(), panel.as_ptr(), w.as_ptr(), k, m, rows) };
        for i in 0..m * rows {
            assert_eq!(got[i].to_bits(), want[i].to_bits(), "threads {threads} i {i}");
        }
    }
}
```

Run: `cargo nextest run -p inferno-pool par_gemm`
Expected: FAIL (`par_gemm` / `GemmFn` not found).

- [ ] **Step 2: Add `GemmFn`, `Pool::par_gemm`, and the `Job` gemm variant**

The pool's `Job` currently holds a `GemvFn` and gemv args. The cleanest change that keeps `par_gemv` untouched is to store an enum job payload. Change `Job.kernel` into a small enum:

```rust
enum JobKind {
    Gemv { kernel: GemvFn },
    Gemm { kernel: GemmFn, m: usize, rows: usize },
}
```

Replace `Job.kernel: Option<GemvFn>` with `kind: Option<JobKind>` and keep `y/xq/w/k/shards`. In `Job::empty`, set `kind: None`. Add `pub type GemmFn = ...` next to `GemvFn`.

In the worker execution block (`worker_loop`) and in `par_gemv`'s shard-0 call, dispatch on `JobKind`. Factor the per-shard call into a helper:

```rust
// SAFETY: caller contract covers `[start,end)`; for Gemm the kernel writes
// y[t*rows + r] for t in 0..m — disjoint across shards because shards
// partition the row range and every token uses the same partition.
unsafe fn run_shard(kind: &JobKind, y: *mut f32, xq: *const u8, w: *const u8, k: usize, start: usize, end: usize) {
    match *kind {
        JobKind::Gemv { kernel } => unsafe { kernel(y, xq, w, k, start, end) },
        JobKind::Gemm { kernel, m, rows } => unsafe { kernel(y, xq, w, k, m, rows, start, end) },
    }
}
```

Add `par_gemm` mirroring `par_gemv` (same shard table, epoch/remaining protocol) but publishing a `JobKind::Gemm`:

```rust
    /// Batched GEMM across up to `active_threads()` lanes; splits `0..rows`
    /// into the same shards as `par_gemv`. Each output row (all `m` tokens)
    /// is computed by one lane, so thread count never changes output bits.
    ///
    /// # Safety
    /// As [`par_gemv`], with `y` valid for `m*rows` f32 and `xq` for `m`
    /// activation rows; calls must not overlap.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn par_gemm(&self, kernel: GemmFn, y: *mut f32, xq: *const u8, w: *const u8, k: usize, m: usize, rows: usize) {
        if rows == 0 || m == 0 {
            return;
        }
        let active = self.active_threads();
        let shards = shard_table(rows, active);
        if shards.len() == 1 {
            // SAFETY: caller contract covers the full range for all m tokens.
            unsafe { kernel(y, xq, w, k, m, rows, 0, rows) };
            return;
        }
        // ... identical publish/epoch/unpark/spin as par_gemv, but:
        //   *self.shared.job.get() = Job { kind: Some(JobKind::Gemm { kernel, m, rows }), y, xq, w, k, shards };
        //   run shard 0 via run_shard(&JobKind::Gemm{..}, ...).
    }
```

Refactor `par_gemv` to build `JobKind::Gemv { kernel }` and use `run_shard`. The worker loop's kernel call becomes `run_shard(&job.kind..., ...)`; read `kind` (a `Copy`-able enum of `Copy` fields) out of the job under the same SAFETY protocol.

- [ ] **Step 3: Add the `inferno_par_gemm` host symbol**

In `crates/inferno-pool/src/lib.rs`, add alongside `inferno_par_gemv` a `inferno_par_gemm` with the same `DISPATCH_CLAIMED` CAS guard and serial fallback (share the single guard — GEMV and GEMM never run concurrently within one forward pass):

```rust
/// Host dispatcher for batched prefill GEMM (M4b.2). Same single-dispatcher
/// guard + serial fallback as [`inferno_par_gemv`]; shares `DISPATCH_CLAIMED`
/// (a forward pass issues GEMV and GEMM serially, never overlapping).
///
/// # Safety
/// As [`Pool::par_gemm`]; `kernel` is a valid GEMM fn pointer. Generated code
/// guarantees this by construction.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_par_gemm(
    kernel: GemmFn, y: *mut f32, xq: *const u8, w: *const u8, k: usize, m: usize, rows: usize,
) {
    if rows == 0 || m == 0 {
        return;
    }
    match GLOBAL.get() {
        Some(pool) => {
            if DISPATCH_CLAIMED.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                // SAFETY: forwarding the caller's contract unchanged.
                unsafe { pool.par_gemm(kernel, y, xq, w, k, m, rows) };
                DISPATCH_CLAIMED.store(false, Ordering::Release);
            } else {
                // SAFETY: full range, all m tokens, serial.
                unsafe { kernel(y, xq, w, k, m, rows, 0, rows) };
            }
        }
        // SAFETY: full range, serial.
        None => unsafe { kernel(y, xq, w, k, m, rows, 0, rows) },
    }
}
```

Re-export `GemmFn` and `inferno_par_gemm` from `lib.rs` (`pub use pool::{GemmFn, GemvFn, Pool};`).

- [ ] **Step 4: Retain the symbol in the host binary**

In `crates/inferno-core/src/artifact.rs` `ensure_kernels_linked()`, add the six GEMM symbols and the dispatcher:

```rust
    p(inferno_kernels::inferno_gemm_f32_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemm_f32_rs8_avx2 as *const ());
    p(inferno_kernels::inferno_gemm_q8_0_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemm_q8_0_rs8_avx2 as *const ());
    p(inferno_kernels::inferno_gemm_q4_k_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemm_q4_k_rs8_avx2 as *const ());
    p(inferno_pool::inferno_par_gemm as *const ());
```

Mirror the same additions in `crates/inferno-codegen/tests/differential.rs` `retain_kernel_symbols()`.

- [ ] **Step 5: Run pool + core tests, commit**

Run: `cargo nextest run -p inferno-pool && cargo nextest run -p inferno-core`
Expected: PASS (existing `par_gemv` tests unaffected by the `JobKind` refactor).

```bash
git add -A
git commit -m "feat(pool): inferno_par_gemm dispatcher (row-sharded, bit-identical across threads)"
```

---

### Task 8: Size the activation scratch ×`prefill_tile`

**Files:**
- Modify: `crates/inferno-plan/src/memory.rs`
- Modify: `crates/inferno-plan/src/snapshots/inferno_plan__plan__tests__plan_dump_gguf.snap`

**Interfaces:**
- Consumes: `prefill_tile` (Task 1's `plan_arena` param).
- Produces: `ArenaLayout.act_scratch_bytes == prefill_tile * max_over_matmuls(packed_act_bytes)` — a T-row quantized-activation panel. `act_scratch_off` and `total_f32` (the f32 arena) are unchanged.

- [ ] **Step 1: Write the failing test**

In `memory.rs` tests, add:

```rust
    #[test]
    fn act_scratch_scales_with_prefill_tile() {
        let (graph, weights) = setup();
        let a1 = plan_arena(&graph, &weights, 128, 1).unwrap();
        let a64 = plan_arena(&graph, &weights, 128, 64).unwrap();
        // f32 arena identical; only the quant panel grows ×T.
        assert_eq!(a1.total_f32, a64.total_f32);
        assert_eq!(a64.act_scratch_bytes, a1.act_scratch_bytes * 64);
        assert_eq!(a64.act_scratch_off, a1.act_scratch_off);
    }
```

Run: `cargo nextest run -p inferno-plan act_scratch_scales`
Expected: FAIL (the `×T` multiply is not applied yet — `plan_arena` currently ignores `prefill_tile`).

- [ ] **Step 2: Apply the multiply**

In `plan_arena`, change the scratch computation:

```rust
    let per_row = weights
        .weights
        .iter()
        .map(|w| packed_act_bytes(&w.dtype, w.k))
        .max()
        .unwrap_or(0);
    // The prefill GEMM activation panel holds `prefill_tile` quantized rows
    // (decode uses row 0 only, which fits within the panel). Sizing the
    // scratch ×T is the only arena change tiling needs — the f32 intermediate
    // arena already reserves `max_seq_len` rows per value.
    let act_scratch_bytes = per_row * prefill_tile.max(1);
    let act_scratch_off = total_f32 * 4;
```

Run: `cargo nextest run -p inferno-plan act_scratch_scales`
Expected: PASS.

- [ ] **Step 3: Regenerate + review the plan snapshot**

The default `plan()` test now uses `prefill_tile = 64` (from Task 1's call-site update), so `act_scratch` in the dump grows 64×. Regenerate and eyeball:

Run: `cargo insta test -p inferno-plan --review`
Verify only the `act_scratch=@…+N` number changed (to 64× the old `292` = `18688`) and `total_f32` is unchanged. Accept.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(plan): size activation scratch to prefill_tile rows (GEMM panel)"
```

---

### Task 9: Tile `lower_prefill` into batched GEMM passes

**Files:**
- Modify: `crates/inferno-codegen/src/loopir.rs` (`gemm_symbol`)
- Modify: `crates/inferno-codegen/src/llvm/mod.rs` (declare `inferno_par_gemm`)
- Modify: `crates/inferno-codegen/src/llvm/ops.rs` (tiling + batched `lower_gemv`)
- Modify: `crates/inferno-codegen/tests/differential.rs` (tile-invariance test)

**Interfaces:**
- Consumes: `KernelSet::gemm` symbols (Task 5), `inferno_par_gemm` (Task 7), `prefill_tile` (via `plan.prefill_tile` — add it to `Plan`, or read from `CompileOptions`; see Step 1).
- Produces: a `prefill` entry point that processes tokens in tiles of `PREFILL_TILE`, calling GEMM once per matmul and looping elementwise/attention ops over the tile's rows. Decode is unchanged. Output bitwise-invariant to `T`.

- [ ] **Step 1: Carry `prefill_tile` into the `Plan`**

Add `pub prefill_tile: usize` to `inferno_plan::Plan` (set it in `plan()` from the new param) so codegen reads `plan.prefill_tile` without a separate options thread. Update `Plan::dump` to print it on the first line: `plan (max_seq_len={} prefill_tile={})`. Regenerate the plan snapshot (`cargo insta test -p inferno-plan --review`; the header line gains ` prefill_tile=64`).

- [ ] **Step 2: `gemm_symbol` in loopir**

In `crates/inferno-codegen/src/loopir.rs`, add next to `gemv_symbol`:

```rust
/// `inferno_gemm_{dtype}_rs8_{isa}`: the batched sibling of `gemv_symbol`,
/// selected identically (widened F16/BF16 → f32 kernel).
pub fn gemm_symbol(dtype: &DType, isa: inferno_kernels::KernelIsa) -> String {
    gemv_symbol(dtype, isa).replace("_gemv_", "_gemm_")
}
```

Make `gemv_symbol` `pub(crate)` if not already, or inline the dtype/isa match. (Keep the `Step::Gemv` carrying its gemv symbol; codegen derives the gemm symbol from it at emit time, so `LoopIr`/its snapshot are unchanged.)

- [ ] **Step 3: Declare `inferno_par_gemm` in the module**

In `crates/inferno-codegen/src/llvm/mod.rs` `declare_kernels`, after the `inferno_par_gemv` decl:

```rust
        // void inferno_par_gemm(ptr kernel, ptr y, ptr xq, ptr w, i64 k, i64 m, i64 rows)
        let par_gemm_ty = void.fn_type(
            &[ptr.into(), ptr.into(), ptr.into(), ptr.into(), i64_t.into(), i64_t.into(), i64_t.into()],
            false,
        );
        self.module.add_function("inferno_par_gemm", par_gemm_ty, Some(Linkage::External));
```

Also declare the six `inferno_gemm_*_rs8_*` extern symbols by extending the existing gemv declaration loop (add a `for kind in ["gemv", "gemm"]` and, for gemm, the 8-arg type). Update the `scaffold_verifies` test to also assert `inferno_par_gemm` and `inferno_gemm_` appear.

- [ ] **Step 4: Restructure `lower_prefill` to tile**

Replace the single `range_loop(n, ...)` in `lower_prefill` with an outer tile loop. Within each tile, iterate the program (islands→steps) once, batching matmuls and looping other ops over the tile rows. Add a `lower_tile` method and a batched `lower_gemm` (renaming today's per-token `lower_gemv` body to serve the m-loop panel path). Key structure:

```rust
    fn lower_prefill(&self, func: FunctionValue<'c>, loopir: &LoopIr) {
        let entry = self.ctx.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);
        let tokens = func.get_nth_param(0).unwrap().into_pointer_value();
        let n = func.get_nth_param(1).unwrap().into_int_value();
        let pos_off = func.get_nth_param(2).unwrap().into_int_value();
        let weights = func.get_nth_param(3).unwrap().into_pointer_value();
        let kv = func.get_nth_param(4).unwrap().into_pointer_value();
        let arena = func.get_nth_param(5).unwrap().into_pointer_value();
        let logits_out = func.get_nth_param(6).unwrap().into_pointer_value();
        let t = self.const_i64(self.plan.prefill_tile as u64);

        // for tile_start in (0..n).step_by(T): m = min(T, n - tile_start)
        let ctx_env = TileEnv { tokens, pos_off, weights, kv, arena };
        self.tile_loop(n, t, |cg, tile_start, m| {
            cg.lower_tile(loopir, &ctx_env, tile_start, m);
        });

        // Last token's logits row (n-1), unchanged.
        let last = self.builder.build_int_sub(n, self.const_i64(1), "last").unwrap();
        self.emit_logits_copy(arena, logits_out, last);
        self.builder.build_return(None).unwrap();
    }
```

Where `TileEnv` bundles the entry-point pointers and `tile_loop(count, step, body)` emits a strided loop computing `m = min(step, count - tile_start)` per iteration (a small variant of `range_loop`; write it in full). `lower_tile` iterates steps: for a `Step::Gemv`, call `lower_gemm(env, step, tile_start, m)` (batched); for every other step, emit `range_loop(m, |cg, ti| { let frame = env.frame(cg, tile_start + ti); cg.profiled(label, |cg| cg.lower_step(&frame, step)) })`. Fold the profiler wrap so prefill profiling still attributes per op-kind (the matmul is timed once per tile; elementwise once per tile across its m-loop).

Attention keeps its existing per-row `lower_attention` — inside the `range_loop(m)` each token appends its KV then reads `[0..=pos]`, bit-identical to today's per-token order (a later token's KV is not yet appended when an earlier token reads, and `visible = pos+1` excludes it regardless).

- [ ] **Step 5: Batched `lower_gemm`**

Rewrite the matmul lowering to fill an `m`-row activation panel then issue one `inferno_par_gemm`. For a quantized weight, loop `ti in 0..m` quantizing source row `(tile_start+ti)` into `act_scratch + ti*act_row_bytes`; for F32, pass the arena source rows directly as the panel (contiguous, stride `k` f32 — no copy). The output base is the arena slot of `out` at row `tile_start` (stride `rows` f32 per token, which the GEMM strides internally):

```rust
    /// Batched matmul for a tile of `m` tokens starting at `tile_start`.
    /// Quantized weights: quantize each token's source row into the T-row
    /// act-scratch panel, then one par_gemm. F32 weights: the source rows are
    /// already a contiguous panel in the arena.
    fn lower_gemm(&self, env: &TileEnv<'c>, step: &Step, tile_start: IntValue<'c>, m: IntValue<'c>) {
        let Step::Gemv { symbol, weight, out, rows, k } = step else { unreachable!() };
        let pw = &self.plan.weights.weights[*weight];
        let src = self.graph.nodes[*out - 1].inputs[0];
        let k_c = self.const_i64(*k as u64);
        let rows_c = self.const_i64(*rows as u64);
        let gemm_sym = crate::loopir::gemm_symbol(&pw.dtype, pw.isa);

        // Panel base + per-token stride (bytes).
        let (panel_ptr, stride_bytes) = if pw.dtype != inferno_formats::DType::F32 {
            // Quantize each token's source row into scratch[ti*act_row].
            let act_row = (self.plan.arena.act_scratch_bytes / self.plan.prefill_tile) as u64;
            let scratch = self.act_scratch_ptr_row0(env.arena); // scratch base
            let qsym = Self::quantize_symbol(&pw.dtype, pw.isa);
            let qfn = self.module.get_function(&qsym).expect("quantize kernel declared");
            self.range_loop(m, |cg, ti| {
                let row = cg.add(tile_start, ti);
                let src_ptr = cg.arena_row_ptr_at(env.arena, src, row);
                let dst = cg.byte_ptr(scratch, cg.builder.build_int_mul(ti, cg.const_i64(act_row), "actoff").unwrap());
                cg.builder.build_call(qfn, &[src_ptr.into(), dst.into(), k_c.into()], "q").unwrap();
            });
            (scratch, act_row)
        } else {
            // F32: panel = source row at tile_start, stride = k f32 (bytes = 4k).
            let base = self.arena_row_ptr_at(env.arena, src, tile_start);
            (base, (*k as u64) * 4)
        };
        let _ = stride_bytes; // stride is implied by (dtype,k); kernel recomputes it

        // Output panel base: value `out` row `tile_start`.
        let y_ptr = self.arena_row_ptr_at(env.arena, *out, tile_start);
        let w_ptr = self.byte_ptr(env.weights, self.const_i64(pw.offset as u64));
        let gfn = self.module.get_function(&gemm_sym).expect("gemm kernel declared");
        let pfn = self.module.get_function("inferno_par_gemm").expect("par gemm declared");
        self.builder.build_call(pfn, &[
            gfn.as_global_value().as_pointer_value().into(),
            y_ptr.into(), panel_ptr.into(), w_ptr.into(),
            k_c.into(), m.into(), rows_c.into(),
        ], "par_gemm").unwrap();
        // Bias (if any) is a separate Step handled in the elementwise m-loop.
    }
```

Add the small helpers `arena_row_ptr_at(arena, v, row)` (like `arena_row_ptr` but for an explicit row value, not `frame.row`) and `act_scratch_ptr_row0(arena)` (the scratch base = existing `act_scratch_ptr` with row 0). Reuse `row_base`-style math with an explicit `row: IntValue`.

**Constraint the plan must enforce:** the GEMM activation stride the kernel computes (`act_len(k)` for quantized, `k*4` for f32) must equal the panel stride emitted here. For quantized weights `act_row == act_len(k)` because `act_scratch_bytes == prefill_tile * max_over_matmuls(act_len)` and this matmul's `act_len(k) ≤ that max`; to guarantee they are equal (not just ≤), quantize into `ti * act_len(k)`, not `ti * max_act_row`. **Fix:** use this matmul's own `act_len(k)` as the panel stride, computed from `pw` at compile time, so the panel is tightly packed for this GEMM. Update the code above to compute `act_row = packed_act_bytes(pw.dtype, k)` (import the plan helper or recompute: `q8a_len`/`q8k_len`). The scratch region is large enough because it is sized to `prefill_tile * max_act_row ≥ prefill_tile * this_act_row`.

- [ ] **Step 6: Tile-invariance under the differential gate**

In `crates/inferno-codegen/tests/differential.rs`, add a test compiling the same fixture at two tile sizes and asserting **bitwise-equal** last-token logits:

```rust
#[test]
fn prefill_tiling_is_bit_invariant_to_tile_size() {
    use inferno_codegen::CompileOptions;
    let fixture = "../inferno-formats/tests/fixtures/tiny.gguf";
    let desc = load_desc(Path::new(fixture)).unwrap();
    let graph = build_graph(&desc).unwrap();
    let target = TargetDesc::detect().unwrap();
    let tokens: Vec<u32> = vec![1, 5, 9, 3, 2, 7, 4, 6, 0, 8]; // spans >1 tile at T=4
    let compile_at = |t: usize, dir: &Path| {
        let a = compile(&desc, &graph, &target, 64, &CompileOptions { profile: false, prefill_tile: t }, dir).unwrap();
        let meta: Meta = serde_json::from_slice(&std::fs::read(a.dir.join("meta.json")).unwrap()).unwrap();
        unsafe { run_compiled(&a.dir, &tokens, &meta) }
    };
    let d1 = tempfile::tempdir().unwrap();
    let d4 = tempfile::tempdir().unwrap();
    let l1 = compile_at(1, d1.path());
    let l4 = compile_at(4, d4.path());
    for (i, (a, b)) in l1.iter().zip(&l4).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "logit {i} differs between T=1 and T=4");
    }
}
```

Also confirm the existing `differential_tiny_*` gates still pass (they now exercise the tiled prefill at the default T=64, but the 4-token prompt fits one tile).

Run: `cargo nextest run -p inferno-codegen`
Expected: PASS. If a differential goes red, the tiling has a bug — fix the codegen, never the tolerance.

- [ ] **Step 7: Lint, full test, commit**

Run: `mise run lint && mise run test`

```bash
git add -A
git commit -m "feat(codegen): tile prefill into batched GEMM passes via inferno_par_gemm"
```

---

## Phase 4 — Prefill data point + kernel benches

### Task 10: GEMM criterion benches + prefill protocol run

**Files:**
- Modify: `crates/inferno-kernels/benches/gemv.rs`
- Modify (record only): `docs/superpowers/specs/2026-07-07-m4b2-per-thread-gap-design.md`

**Interfaces:**
- Produces: criterion GEMM benches (batched, over the same Llama-family shapes with representative `m`) behind `--features ggml-compare`, comparing against ggml's batched path where available.

- [ ] **Step 1: Add GEMM benches**

In `crates/inferno-kernels/benches/gemv.rs`, add a `gemm` benchmark group that packs the same shapes and drives `KernelSet::gemm` with `m ∈ {1, 16, 64}` (Throughput = `m * rows * k` MACs), so the M-loop's weight-reuse win is visible as `m` grows. Keep the existing GEMV group. Follow the file's existing structure (`criterion_group!`).

Run: `cargo bench -p inferno-kernels --no-run`
Expected: compiles.

- [ ] **Step 2: Manual — kernel bench numbers**

Inside the devenv shell on the quiet machine: `mise run bench-kernels`. Record the GEMM GiB/s vs ggml for the hot Q8_0 shapes in the spec's Amendments (same rules as M2). This shows whether the batched kernel reaches the memory-bandwidth ceiling per token.

- [ ] **Step 3: Manual — the prefill data point (the exit criterion)**

Inside the devenv shell, release build, against the pinned Q8_0 model, threads=1 (the per-thread criterion):

```bash
mise run bench -- /home/dev/.cache/inferno-tests/qwen2.5-0.5b-instruct-q8_0.gguf --threads 1 --json
```

Record the full data point (table + json) in the spec's **## Amendments**, computing the prefill ratio `inferno_t1_pp / llama_t1_pp`. **Exit criterion: t=1 pp ≥ 0.7× llama.cpp t=1.**

- If met: note it, and run one `--profile` capture (Task 4's path) at T=64 to record the post-tiling prefill profile alongside.
- If **not** met: per the spec's escalation clause, add a scoped amendment proposing register-blocked GEMM tiles targeted at the shapes the profile blames — do **not** silently start that work; it is its own follow-up task.

- [ ] **Step 4: Commit the benches + recorded amendment**

```bash
git add -A
git commit -m "bench(kernels): batched GEMM criterion group; record M4b.2 prefill data point"
```

---

## Gated follow-ups (not tasks in this plan)

These are deliberately left as spec amendments / separate plans, per the design's pre-registered forks:

- **Decode attribution fork (spec Phase 5).** After Task 4's baseline profile and Task 10's post-tiling profile exist, the spec's `## Decode attribution fork` picks **exactly one** decode lever (F16 KV, targeted fusion, or a batched quantize path) via an explicit amendment that also sets the tg exit target. F16 KV — the only lever that touches `logits_abs_tol` — requires switching the interpreter and compiled paths together and re-deriving the tolerance with recorded data; it is out of scope here.
- **Register-blocked GEMM escalation.** Triggered only if Task 10's prefill ratio misses 0.7×; a scoped amendment targeting the profiled hot shapes.
- **Bare-metal re-measurement (inherited from M4b.1).** The threading/scaling questions stay gated on a re-run on unquota'd hardware; nothing in this plan depends on it (everything here is t=1).
- **AVX-512 / VNNI kernels.** Out of scope; the `Isa::X86_64v4` / `Feature::Vnni` detection is already plumbed for when Q4_K performance or bare metal makes it worthwhile.

---

## Self-Review

**Spec coverage:**
- Profiler (built-in, rdtsc, flag-gated, GB/s column, bit-identical, distinct cache key) → Tasks 1–4. ✓
- Prefill tiles (M-loop GEMM kernels, bit-identity, codegen tiling, par_gemm, act_scratch ×T, escalation clause) → Tasks 5–10. ✓
- Split exit criterion (hard pp ≥ 0.7×; decode target by later amendment) → Task 10 Step 3 + gated follow-ups. ✓
- Levers committed (tiles) vs contingent (one decode lever) → Tasks 5–10 + gated follow-ups. ✓
- Standing invariants (thread/tile/ISA bit-identity; frozen tolerances) → Global Constraints + Tasks 5/7/9 tests. ✓
- Out-of-scope (AVX-512, parallel attention, F16 KV, CI perf gates) → gated follow-ups + Global Constraints. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N"; each task shows the code or the exact mechanical transformation with before/after. Q4_K GEMM (Task 5 Step 4) is specified as the identical block-outer/token-inner transformation with the parity bar rather than repeated verbatim — acceptable because it is not on the measured path and its correctness is fully pinned by the rig tests written in that step.

**Type consistency:** `CompileOptions { profile, prefill_tile }`, `GemmFn` (8-arg), `inferno_gemm_{d}_rs8_{isa}`, `KernelSet::gemm(y, xq, w, m, rows, k, row_start, row_end)`, `inferno_par_gemm(kernel, y, xq, w, k, m, rows)`, `Meta.{prefill_tile, profile_slots}`, `Plan.prefill_tile`, `plan(.., max_seq_len, prefill_tile)`, `cache_key(.., opts)` — used consistently across tasks.

**Ambiguity check:** The one real hazard — the GEMM activation-panel stride matching the kernel's internal `act_len(k)` — is called out explicitly in Task 9 Step 5 with the fix (quantize into `ti * act_len(k)`, not `ti * max_act_row`).
