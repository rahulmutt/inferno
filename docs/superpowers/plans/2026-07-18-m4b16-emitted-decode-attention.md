# M4b.16 — Codegen-Emitted Geometry-Specialized Decode Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a per-model, geometry-specialized decode attention function emitted by codegen (bit-identical to the Rust hspan kernels), behind a cache-keyed flag, gated by the two-box quiet-hw tg ladder — after an inkwell 0.9 / LLVM 22 toolchain upgrade lands as its own PR.

**Architecture:** Codegen emits a private `attn_hspan.emitted` function with the exact 13-arg `AttnFn` C ABI; geometry params are ignored in favor of baked constants; `lower_attention` passes its pointer to `inferno_par_attention_heads` instead of the runtime symbol when `CompileOptions::emitted_attn` is set. The emission lives in a self-contained `AttnEmitCtx` so a probe compiler (`compile_attn_probe`) can build a one-function `.so` for the bit-exactness harness without a model.

**Tech Stack:** Rust, inkwell 0.9 (`llvm22-1`), LLVM 22.1.8 (devenv `llvmPackages_22`), libloading (tests), proptest-style loops (plain `for` over fixed geometry/input sets), mise tasks, PhoenixNAP quiet-hw sessions.

**Spec:** `docs/superpowers/specs/2026-07-18-m4b16-emitted-decode-attention-design.md` — read it before starting. The gate ladder, thresholds, and amendment-recording rules there are binding.

## Global Constraints

- **Bit-neutrality is the whole game:** the emitted function must be *bit-identical* to `inferno_attention_f32_scalar_hspan` / `_avx2_hspan` for every geometry, span, and pos. Arithmetic order is copied from `attn_core_scalar` (`crates/inferno-kernels/src/attention.rs:84`) operation for operation. No fast-math flags anywhere in emitted IR. `llvm.fma` only — never `llvm.fmuladd`.
- **Zero edits to `crates/inferno-graph/src/tolerance.rs`** (closing verdict diff-checks this).
- **Rust kernels are byte-untouched** except making `expf` constants `pub` (Task 4). The rig stays the oracle.
- **`inferno-codegen` stays `unsafe`-free** in `src/`; integration tests under `tests/` scope `#![allow(unsafe_code)]` with the same justification comment as `tests/differential.rs`.
- **No `HOST_ABI_VERSION` bump** — the pool fn-pointer contract is unchanged. `CompileOptions::emitted_attn` is hashed into the artifact cache key instead.
- **Vector loads/stores in emitted IR must set alignment 4** (KV/arena pointers are only f32-aligned; a default-aligned `<8 x float>` load is UB → potential `vmovaps` fault).
- **All-lane exactness:** `mise run test`, `mise run lint` green before every commit claim; `cargo test -p inferno-codegen --test differential` and `cargo test -p inferno-core --test artifact` after any lowering change.
- **`mise run metal` spends real money:** Tasks 9–10 are operator-driven; never CI. After any interrupted session run `mise run metal-gc`.
- **Verdicts are human:** session scripts print tables; gate arithmetic and verdicts are computed and recorded by the operator in the spec §Amendments, never by scripts.

## File Map

| File | Role |
|---|---|
| `Cargo.toml` (workspace) | inkwell `0.6`/`llvm18-1` → `0.9`/`llvm22-1` (Task 1) |
| `devenv.nix` | `llvmPackages_18` → `llvmPackages_22`, `LLVM_SYS_181_PREFIX` → `LLVM_SYS_221_PREFIX` (Task 1) |
| `AGENTS.md` | LLVM coupling note (Task 1); decode-attention paragraph (Task 7) |
| `crates/inferno-codegen/src/emit.rs` | `CompileOptions::emitted_attn` field; `compile_attn_probe` (Tasks 2, 4) |
| `crates/inferno-core/src/cache.rs` | hash the flag into `cache_key` (Task 2) |
| `cli/src/main.rs` | `INFERNO_EMITTED_ATTN` env plumbing (Task 2) |
| `crates/inferno-kernels/src/expf.rs` | constants → `pub` (Task 4) |
| `crates/inferno-codegen/src/llvm/attn_emit.rs` | **new** — `AttnEmitCtx` + `emit_attn_hspan_fn` (Task 4) |
| `crates/inferno-codegen/tests/attn_emit.rs` | **new** — bit-exactness harness (Tasks 4–5) |
| `crates/inferno-codegen/src/llvm/ops.rs` | `lower_attention` flag branch (Task 6) |
| `crates/inferno-core/tests/artifact.rs` | same-logits invariant test (Task 6) |
| `scripts/quiet-hw/gate-emitted-attn.sh` | **new** — session lever-vs-baseline script (Task 8) |
| spec §Amendments | local context point, session records, gate + closing verdicts (Tasks 7, 9–11) |

---

### Task 1: Toolchain prerequisite — inkwell 0.9 / LLVM 22 (own branch, own PR, lands first)

**Files:**
- Modify: `Cargo.toml` (workspace `[workspace.dependencies]` inkwell line)
- Modify: `devenv.nix` (LLVM package + `LLVM_SYS_*` env)
- Modify: `AGENTS.md` (coupling note)

**Interfaces:**
- Produces: a workspace that builds and passes all gates on inkwell 0.9 / LLVM 22.1.8. Every later task assumes this toolchain.

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull && git checkout -b llvm22-upgrade
```

- [ ] **Step 2: Bump the pins**

`Cargo.toml`:

```toml
inkwell = { version = "0.9", features = ["llvm22-1"] }
```

`devenv.nix` — replace both LLVM references:

```nix
    pkgs.llvmPackages_22.llvm.dev
```

```nix
  env.LLVM_SYS_221_PREFIX = "${pkgs.llvmPackages_22.llvm.dev}";
```

(Keep the surrounding comments; update `llvm18-1`/`18` digits inside them to `llvm22-1`/`22`. The locked nixpkgs rev `9e92285f…` already carries `llvmPackages_22` = 22.1.8 — do **not** bump the flake lock.)

- [ ] **Step 3: Update the AGENTS.md coupling note**

In the "Toolchain" bullet, change `llvm18-1` → `llvm22-1`, `LLVM 18.1.8` → `LLVM 22.1.8`, `pkgs.llvmPackages_18` → `pkgs.llvmPackages_22`, `LLVM_SYS_181_PREFIX` → `LLVM_SYS_221_PREFIX`. The rule text (exact major.minor match) stays.

- [ ] **Step 4: Re-enter the shell and verify the pairing**

```bash
devenv shell -- llvm-config --version
```

Expected: `22.1.8`. If `llvmPackages_22` is missing from the locked rev, STOP and report — do not bump the lock unilaterally.

- [ ] **Step 5: Compile-fix loop for the inkwell 0.6 → 0.9 API migration**

```bash
devenv shell -- cargo build -p inferno-codegen 2>&1 | head -50
```

Fix errors mechanically, guided by the inkwell changelog (0.7/0.8/0.9). Rules: keep the existing `.unwrap()` style of `llvm/ops.rs`; no behavioral changes; if a builder method was renamed, rename the call — do not restructure. Repeat until the crate builds clean.

- [ ] **Step 6: Full correctness gates**

```bash
devenv shell -- cargo test -p inferno-codegen --test differential
devenv shell -- cargo test -p inferno-core --test artifact
devenv shell -- mise run test
devenv shell -- mise run lint
git diff --stat crates/inferno-graph/src/tolerance.rs   # must be empty
```

Expected: all green, empty tolerance diff. The differential and artifact gates are the acceptance test for the new LLVM major — a red here is a real finding, not a flake; debug before proceeding.

- [ ] **Step 7: Commit, PR, merge**

```bash
git add Cargo.toml Cargo.lock devenv.nix AGENTS.md
git commit -m "toolchain: inkwell 0.9 / LLVM 22.1.8 (M4b.16 prerequisite)"
git push -u origin llvm22-upgrade
gh pr create --title "Toolchain: inkwell 0.9 / LLVM 22 (M4b.16 prerequisite)" \
  --body "Prerequisite PR per docs/superpowers/specs/2026-07-18-m4b16-emitted-decode-attention-design.md §Prerequisite. Correctness gates only; perf re-baselines in the M4b.16 sessions."
```

Wait for CI + operator merge before starting Task 2. (First CI run rebuilds the nix cache — per memory, a FlakeHub "path is not valid" failure is runner cache corruption; rerun once before debugging.)

- [ ] **Step 8: Start the milestone branch**

```bash
git checkout main && git pull && git checkout -b m4b16-emitted-attn
```

---

### Task 2: `CompileOptions::emitted_attn` — flag, cache key, env plumbing

**Files:**
- Modify: `crates/inferno-codegen/src/emit.rs:25-42` (`CompileOptions`)
- Modify: `crates/inferno-core/src/cache.rs` (`cache_key`, + its test)
- Modify: `cli/src/main.rs` (both `CompileOptions` construction sites, ~lines 193/210)

**Interfaces:**
- Produces: `CompileOptions { profile: bool, prefill_tile: usize, emitted_attn: bool }` (default `false`); `INFERNO_EMITTED_ATTN=1` env opt-in at the CLI; cache keys that differ by flag state. Tasks 4/6/8 rely on the field name `emitted_attn` exactly.

- [ ] **Step 1: Write the failing cache-key test**

In `crates/inferno-core/src/cache.rs` `mod tests`, extend `key_is_stable_and_input_sensitive` (or add alongside, matching its style):

```rust
    #[test]
    fn key_differs_by_emitted_attn() {
        let t = TargetDesc::detect().unwrap();
        let m = Path::new("../inferno-formats/tests/fixtures/tiny.gguf");
        let base = inferno_codegen::CompileOptions::default();
        let lever = inferno_codegen::CompileOptions {
            emitted_attn: true,
            ..Default::default()
        };
        let k1 = cache_key(m, &t, 64, &base).unwrap();
        let k2 = cache_key(m, &t, 64, &lever).unwrap();
        assert_ne!(k1, k2, "emitted_attn must be part of the artifact identity");
    }
```

- [ ] **Step 2: Run it — expect compile failure**

```bash
cargo test -p inferno-core key_differs_by_emitted_attn
```

Expected: FAIL — `struct CompileOptions has no field named emitted_attn`.

- [ ] **Step 3: Add the field**

`crates/inferno-codegen/src/emit.rs`:

```rust
    /// M4b.16: dispatch decode attention through the codegen-emitted,
    /// geometry-specialized function instead of the runtime hspan symbol.
    /// Bit-identical output (the attn_emit harness is the guard); a distinct
    /// artifact (folded into the cache key).
    pub emitted_attn: bool,
```

and in `Default::default()`:

```rust
            emitted_attn: false,
```

- [ ] **Step 4: Hash it in `cache_key`**

`crates/inferno-core/src/cache.rs`, after the `prefill_tile` update line:

```rust
    h.update([opts.emitted_attn as u8]);
```

- [ ] **Step 5: Env plumbing in the CLI**

At both `CompileOptions` construction sites in `cli/src/main.rs` (the `compile`/`run` and `bench` arms — grep `profile,` near lines 193/210), add:

```rust
            emitted_attn: std::env::var("INFERNO_EMITTED_ATTN").is_ok_and(|v| v == "1"),
```

(If the sites build the struct with `..Default::default()`, add the field explicitly anyway — the env read must reach every compile path the bench uses.)

- [ ] **Step 6: Run tests, lint, commit**

```bash
cargo test -p inferno-core key_differs_by_emitted_attn   # PASS
mise run test && mise run lint
git add -A && git commit -m "M4b.16: CompileOptions::emitted_attn flag, cache-keyed, INFERNO_EMITTED_ATTN env opt-in"
```

---

### Task 3: Expose the expf constants from `inferno-kernels`

**Files:**
- Modify: `crates/inferno-kernels/src/expf.rs` (visibility only)
- Modify: `crates/inferno-kernels/src/lib.rs` (module visibility only, if `expf` is private)

**Interfaces:**
- Produces: `inferno_kernels::expf::{LOG2E, LN2_HI, LN2_LO, C}` as `pub` consts. Task 4's emission reads these — the constants exist in exactly one place.

- [ ] **Step 1: Make the constants public**

In `crates/inferno-kernels/src/expf.rs` change `const LOG2E` / `LN2_HI` / `LN2_LO` / `C` to `pub const`, and add above them:

```rust
// pub: inferno-codegen's attn_emit reads these to bake the identical
// polynomial into emitted IR (M4b.16). One source of truth — never copy
// the values into codegen.
```

In `crates/inferno-kernels/src/lib.rs`, change `mod expf;` to `pub mod expf;` (keep `expf_scalar`/`expf_avx2` at their current `pub(crate)` visibility — only the constants go public).

- [ ] **Step 2: Verify kernels are otherwise byte-untouched, commit**

```bash
cargo test -p inferno-kernels
git diff --stat crates/inferno-kernels/   # expf.rs + lib.rs only, visibility lines only
git add -A && git commit -m "M4b.16: pub expf constants (single source for emitted IR)"
```

---

### Task 4: `AttnEmitCtx` + `emit_attn_hspan_fn` + `compile_attn_probe` + first bit-exactness test

This is the core task. The emitted function copies `attn_core_scalar`'s arithmetic operation for operation; read `crates/inferno-kernels/src/attention.rs:84-160` and `expf.rs` side by side with this task before writing IR.

**Files:**
- Create: `crates/inferno-codegen/src/llvm/attn_emit.rs`
- Modify: `crates/inferno-codegen/src/llvm/mod.rs` (add `mod attn_emit;` + re-export)
- Modify: `crates/inferno-codegen/src/emit.rs` (add `compile_attn_probe`)
- Test: `crates/inferno-codegen/tests/attn_emit.rs`

**Interfaces:**
- Consumes: `inferno_kernels::expf::{LOG2E, LN2_HI, LN2_LO, C}` (Task 3); `CompileOptions::emitted_attn` (Task 2, probe ignores it).
- Produces:
  - `pub(crate) fn attn_emit::emit_attn_hspan_fn<'c>(ctx: &'c Context, module: &Module<'c>, head_dim: usize, n_heads: usize, n_kv_heads: usize, name: &str, linkage: Linkage) -> FunctionValue<'c>` — emits the specialized function; caller picks name/linkage (private for models, external for probes).
  - `pub fn inferno_codegen::compile_attn_probe(head_dim: usize, n_heads: usize, n_kv_heads: usize, cpu_features: &str, out_dir: &Path) -> Result<PathBuf>` — one-function `.so` exporting `inferno_attn_probe` with the 13-arg `AttnFn` ABI. Tasks 5 dlopens this; Task 6 calls `emit_attn_hspan_fn` from `lower_attention`.

- [ ] **Step 1: Write the failing test (protocol geometry, scalar oracle)**

`crates/inferno-codegen/tests/attn_emit.rs`:

```rust
//! M4b.16 bit-exactness harness: the codegen-emitted geometry-specialized
//! decode attention function vs the Rust hspan kernels (the rig oracle).
//! Exact bit equality, every geometry/span/pos — this is the third lane of
//! the scalar≡AVX2 discipline.
//!
//! # unsafe
//! Same justification as tests/differential.rs: integration test dlopens a
//! shared object and calls a C-ABI fn pointer; production codegen stays
//! unsafe-free.
#![allow(unsafe_code)]

use std::path::PathBuf;

use inferno_codegen::compile_attn_probe;

/// The 13-arg AttnFn ABI (see inferno-kernels registry::AttnFn).
type AttnFn = unsafe extern "C" fn(
    *mut f32,   // out
    *const f32, // q
    *mut f32,   // kv
    *mut f32,   // scores
    usize,      // kv_base
    usize,      // v_off
    usize,      // pos
    usize,      // kv_dim
    usize,      // n_heads
    usize,      // n_kv_heads
    usize,      // head_dim
    usize,      // h_start
    usize,      // h_end
);

/// Deterministic pseudo-random f32 in [-1, 1) — no rand dep, reproducible.
fn lcg_fill(buf: &mut [f32], seed: &mut u64) {
    for v in buf.iter_mut() {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
    }
}

struct Probe {
    // Field order is drop order: the Library must close before its backing
    // tempdir (holding the .so) is removed.
    _lib: libloading::Library,
    f: AttnFn,
    _dir: tempfile::TempDir,
}

fn build_probe(head_dim: usize, n_heads: usize, n_kv_heads: usize, features: &str) -> Probe {
    let dir = tempfile::tempdir().unwrap();
    let so = compile_attn_probe(head_dim, n_heads, n_kv_heads, features, dir.path()).unwrap();
    let lib = unsafe { libloading::Library::new(&so) }.expect("dlopen probe");
    let f = *unsafe { lib.get::<AttnFn>(b"inferno_attn_probe") }.expect("probe symbol");
    Probe { f, _lib: lib, _dir: dir }
}

/// Run emitted-vs-reference for one geometry/pos/span over random inputs.
fn assert_bit_identical(
    probe: &Probe,
    reference: AttnFn,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    pos: usize,
    h_start: usize,
    h_end: usize,
    seed: u64,
) {
    let kv_dim = n_kv_heads * head_dim;
    let seq_len = pos + 1;
    let v_off = seq_len * kv_dim;
    let kv_base = 0usize;
    let mut seed = seed;
    let mut kv = vec![0f32; 2 * v_off];
    let mut q = vec![0f32; n_heads * head_dim];
    lcg_fill(&mut kv, &mut seed);
    lcg_fill(&mut q, &mut seed);

    let mut out_e = vec![0f32; n_heads * head_dim];
    let mut out_r = vec![0f32; n_heads * head_dim];
    let mut sc_e = vec![0f32; seq_len];
    let mut sc_r = vec![0f32; seq_len];

    unsafe {
        (probe.f)(out_e.as_mut_ptr(), q.as_ptr(), kv.as_mut_ptr(), sc_e.as_mut_ptr(),
            kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim, h_start, h_end);
        reference(out_r.as_mut_ptr(), q.as_ptr(), kv.as_mut_ptr(), sc_r.as_mut_ptr(),
            kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim, h_start, h_end);
    }
    for h in h_start..h_end {
        for d in 0..head_dim {
            let i = h * head_dim + d;
            assert_eq!(
                out_e[i].to_bits(), out_r[i].to_bits(),
                "hd={head_dim} nh={n_heads} nkv={n_kv_heads} pos={pos} span=[{h_start},{h_end}) h={h} d={d}: {} vs {}",
                out_e[i], out_r[i]
            );
        }
    }
}

#[test]
fn emitted_matches_scalar_protocol_geometry() {
    // qwen2.5-0.5b decode geometry: head_dim 64, 14 heads, 2 kv heads.
    let p = build_probe(64, 14, 2, "+avx2,+fma");
    assert_bit_identical(
        &p, inferno_kernels::inferno_attention_f32_scalar_hspan,
        64, 14, 2, /*pos*/ 63, /*span*/ 0, 14, 0x4b16,
    );
}
```

Add to `crates/inferno-codegen/Cargo.toml` `[dev-dependencies]` (if absent): `tempfile = "3"` and `inferno-kernels = { path = "../inferno-kernels" }` (libloading is already there for `differential.rs`).

- [ ] **Step 2: Run it — expect compile failure**

```bash
cargo test -p inferno-codegen --test attn_emit
```

Expected: FAIL — `unresolved import inferno_codegen::compile_attn_probe`.

- [ ] **Step 3: Create `attn_emit.rs` — the emission context and helpers**

`crates/inferno-codegen/src/llvm/attn_emit.rs`. The file is self-contained: its own builder, its own helpers, no dependence on `CodeGen`'s insertion point (it saves/restores nothing — callers position their own builders).

```rust
//! M4b.16: geometry-specialized decode attention, emitted as LLVM IR.
//!
//! `emit_attn_hspan_fn` emits ONE function with the exact 13-arg AttnFn ABI
//! (`inferno_attention_f32_scalar_hspan`'s signature). The geometry
//! parameters (kv_dim, n_heads, n_kv_heads, head_dim) are ACCEPTED AND
//! IGNORED — the baked constants are used instead — so the pool dispatcher
//! calls it exactly like the runtime symbol.
//!
//! Bit-neutrality contract: every float op below copies
//! `attn_core_scalar` (inferno-kernels/src/attention.rs) in order —
//! dot8's 8-lane-partitioned FMA chain + reduce8's fixed tree, the
//! sequential f32::max fold, the block-of-8 expf + reduce8 denominator
//! with scalar tail, and the ascending-t mul_add AV accumulation. expf
//! constants come from `inferno_kernels::expf` (single source). No
//! fast-math flags; `llvm.fma` only (mul_add is a guaranteed fused op —
//! `llvm.fmuladd` may split and is forbidden). All vector memory ops are
//! align-4 (pointers are only f32-aligned).
//!
//! The clamp in expf is emitted as compare+select pairs whose operand
//! order matches `_mm256_max_ps(-88, x)` / `_mm256_min_ps(88, x)` exactly
//! (NaN → x), which also matches scalar `f32::clamp` for every input the
//! kernel contract admits (finite scores).

use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::values::{FloatValue, FunctionValue, IntValue, PointerValue, VectorValue};
use inkwell::{AddressSpace, IntPredicate, FloatPredicate};

use inferno_kernels::expf::{C, LN2_HI, LN2_LO, LOG2E};

pub(crate) struct AttnEmitCtx<'c> {
    ctx: &'c Context,
    b: inkwell::builder::Builder<'c>,
    f32_t: inkwell::types::FloatType<'c>,
    i64_t: inkwell::types::IntType<'c>,
    i32_t: inkwell::types::IntType<'c>,
    ptr_t: inkwell::types::PointerType<'c>,
}

impl<'c> AttnEmitCtx<'c> {
    fn new(ctx: &'c Context) -> Self {
        AttnEmitCtx {
            ctx,
            b: ctx.create_builder(),
            f32_t: ctx.f32_type(),
            i64_t: ctx.i64_type(),
            i32_t: ctx.i32_type(),
            ptr_t: ctx.ptr_type(AddressSpace::default()),
        }
    }
    fn v8_t(&self) -> inkwell::types::VectorType<'c> {
        self.f32_t.vec_type(8)
    }
    fn ci64(&self, v: u64) -> IntValue<'c> {
        self.i64_t.const_int(v, false)
    }
    fn cf32(&self, v: f32) -> FloatValue<'c> {
        self.f32_t.const_float(v as f64) // f32→f64 is exact
    }
    /// f32-element pointer offset via ptrtoint/add/inttoptr (the ops.rs
    /// byte_ptr pattern — codegen avoids GEP).
    fn fptr(&self, base: PointerValue<'c>, elem_off: IntValue<'c>) -> PointerValue<'c> {
        let bytes = self.b.build_int_mul(elem_off, self.ci64(4), "boff").unwrap();
        let bi = self.b.build_ptr_to_int(base, self.i64_t, "p2i").unwrap();
        let sum = self.b.build_int_add(bi, bytes, "paddr").unwrap();
        self.b.build_int_to_ptr(sum, self.ptr_t, "i2p").unwrap()
    }
    fn load_f32(&self, p: PointerValue<'c>) -> FloatValue<'c> {
        self.b.build_load(self.f32_t, p, "ld").unwrap().into_float_value()
    }
    fn store_f32(&self, p: PointerValue<'c>, v: FloatValue<'c>) {
        self.b.build_store(p, v).unwrap();
    }
    /// align-4 <8 x float> load (unaligned-safe: vmovups, never vmovaps).
    fn load_v8(&self, p: PointerValue<'c>) -> VectorValue<'c> {
        let ld = self.b.build_load(self.v8_t(), p, "ldv").unwrap();
        ld.as_instruction_value().unwrap().set_alignment(4).unwrap();
        ld.into_vector_value()
    }
    fn store_v8(&self, p: PointerValue<'c>, v: VectorValue<'c>) {
        let st = self.b.build_store(p, v).unwrap();
        st.set_alignment(4).unwrap();
    }
    fn splat(&self, v: FloatValue<'c>) -> VectorValue<'c> {
        let undef = self.v8_t().get_undef();
        let ins = self
            .b
            .build_insert_element(undef, v, self.i32_t.const_zero(), "ins")
            .unwrap();
        let zeros = inkwell::types::VectorType::const_vector(&[self.i32_t.const_zero(); 8]);
        self.b.build_shuffle_vector(ins, undef, zeros, "splat").unwrap()
    }
    fn splat_c(&self, v: f32) -> VectorValue<'c> {
        inkwell::types::VectorType::const_vector(&[self.cf32(v); 8])
    }
    fn fma_v8(&self, m: &Module<'c>, a: VectorValue<'c>, b: VectorValue<'c>, c: VectorValue<'c>) -> VectorValue<'c> {
        let fma = inkwell::intrinsics::Intrinsic::find("llvm.fma")
            .unwrap()
            .get_declaration(m, &[self.v8_t().into()])
            .unwrap();
        self.b
            .build_call(fma, &[a.into(), b.into(), c.into()], "fma")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_vector_value()
    }
    fn fma_f32(&self, m: &Module<'c>, a: FloatValue<'c>, b: FloatValue<'c>, c: FloatValue<'c>) -> FloatValue<'c> {
        let fma = inkwell::intrinsics::Intrinsic::find("llvm.fma")
            .unwrap()
            .get_declaration(m, &[self.f32_t.into()])
            .unwrap();
        self.b
            .build_call(fma, &[a.into(), b.into(), c.into()], "fma")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value()
    }
    /// reduce8: (0+4)(1+5)(2+6)(3+7) then pairwise — kernels' reduce8 tree.
    fn reduce8(&self, v: VectorValue<'c>) -> FloatValue<'c> {
        let mask = |idx: &[u64]| {
            inkwell::types::VectorType::const_vector(
                &idx.iter().map(|&i| self.i32_t.const_int(i, false)).collect::<Vec<_>>(),
            )
        };
        let undef = self.v8_t().get_undef();
        let lo = self.b.build_shuffle_vector(v, undef, mask(&[0, 1, 2, 3]), "lo").unwrap();
        let hi = self.b.build_shuffle_vector(v, undef, mask(&[4, 5, 6, 7]), "hi").unwrap();
        let a = self.b.build_float_add(lo, hi, "a").unwrap();
        let u4 = self.f32_t.vec_type(4).get_undef();
        let a01 = self.b.build_shuffle_vector(a, u4, mask(&[0, 1]), "a01").unwrap();
        let a23 = self.b.build_shuffle_vector(a, u4, mask(&[2, 3]), "a23").unwrap();
        let bb = self.b.build_float_add(a01, a23, "b").unwrap();
        let b0 = self
            .b
            .build_extract_element(bb, self.i32_t.const_zero(), "b0")
            .unwrap()
            .into_float_value();
        let b1 = self
            .b
            .build_extract_element(bb, self.i32_t.const_int(1, false), "b1")
            .unwrap()
            .into_float_value();
        self.b.build_float_add(b0, b1, "s").unwrap()
    }
    /// A while-style IR loop: body(i) for i in [start, end) step `step`.
    fn loop_range(
        &self,
        f: FunctionValue<'c>,
        start: IntValue<'c>,
        end: IntValue<'c>,
        step: u64,
        name: &str,
        body: impl FnOnce(&Self, IntValue<'c>),
    ) {
        let header = self.ctx.append_basic_block(f, &format!("{name}.h"));
        let bodyb = self.ctx.append_basic_block(f, &format!("{name}.b"));
        let exit = self.ctx.append_basic_block(f, &format!("{name}.x"));
        let iv = self.b.build_alloca(self.i64_t, &format!("{name}.i")).unwrap();
        self.b.build_store(iv, start).unwrap();
        self.b.build_unconditional_branch(header).unwrap();
        self.b.position_at_end(header);
        let i = self.b.build_load(self.i64_t, iv, "i").unwrap().into_int_value();
        let cont = self.b.build_int_compare(IntPredicate::ULT, i, end, "lt").unwrap();
        self.b.build_conditional_branch(cont, bodyb, exit).unwrap();
        self.b.position_at_end(bodyb);
        body(self, i);
        let i2 = self.b.build_load(self.i64_t, iv, "i").unwrap().into_int_value();
        let nx = self.b.build_int_add(i2, self.ci64(step), "nx").unwrap();
        self.b.build_store(iv, nx).unwrap();
        self.b.build_unconditional_branch(header).unwrap();
        self.b.position_at_end(exit);
    }
}
```

- [ ] **Step 4: Add the expf emitters to the same impl block**

```rust
impl<'c> AttnEmitCtx<'c> {
    /// Emit expf on a <8 x float>: constants and FMA order verbatim from
    /// inferno_kernels::expf (expf_avx2, bit-identical to expf_scalar).
    fn expf_v8(&self, m: &Module<'c>, x: VectorValue<'c>) -> VectorValue<'c> {
        // clamp: select order matches _mm256_max_ps(-88, x) / _mm256_min_ps(88, x).
        let lo = self.splat_c(-88.0);
        let gt = self.b.build_float_compare(FloatPredicate::OGT, lo, x, "gt").unwrap();
        let x = self.b.build_select(gt, lo, x, "cl").unwrap().into_vector_value();
        let hi = self.splat_c(88.0);
        let lt = self.b.build_float_compare(FloatPredicate::OLT, hi, x, "lt").unwrap();
        let x = self.b.build_select(lt, hi, x, "ch").unwrap().into_vector_value();
        // n = roundeven(x * LOG2E)
        let xl = self.b.build_float_mul(x, self.splat_c(LOG2E), "xl").unwrap();
        let re = inkwell::intrinsics::Intrinsic::find("llvm.roundeven")
            .unwrap()
            .get_declaration(m, &[self.v8_t().into()])
            .unwrap();
        let n = self
            .b
            .build_call(re, &[xl.into()], "n")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_vector_value();
        // r = fma(n, -LN2_LO, fma(n, -LN2_HI, x))
        let r = self.fma_v8(m, n, self.splat_c(-LN2_HI), x);
        let r = self.fma_v8(m, n, self.splat_c(-LN2_LO), r);
        // Horner C6..C0
        let mut p = self.splat_c(C[6]);
        for k in (0..6).rev() {
            p = self.fma_v8(m, p, r, self.splat_c(C[k]));
        }
        // pow2n = bitcast((fptosi(n) + 127) << 23)
        let i32v8 = self.i32_t.vec_type(8);
        let ni = self.b.build_float_to_signed_int(n, i32v8, "ni").unwrap();
        let c127 = inkwell::types::VectorType::const_vector(&[self.i32_t.const_int(127, false); 8]);
        let c23 = inkwell::types::VectorType::const_vector(&[self.i32_t.const_int(23, false); 8]);
        let add = self.b.build_int_add(ni, c127, "e").unwrap();
        let shl = self.b.build_left_shift(add, c23, "bits").unwrap();
        let pf = self.b.build_bit_cast(shl, self.v8_t(), "p2").unwrap().into_vector_value();
        self.b.build_float_mul(p, pf, "exp").unwrap()
    }

    /// Scalar expf, same constants/order (for the softmax tail).
    fn expf_f32(&self, m: &Module<'c>, x: FloatValue<'c>) -> FloatValue<'c> {
        let lo = self.cf32(-88.0);
        let gt = self.b.build_float_compare(FloatPredicate::OGT, lo, x, "gt").unwrap();
        let x = self.b.build_select(gt, lo, x, "cl").unwrap().into_float_value();
        let hi = self.cf32(88.0);
        let lt = self.b.build_float_compare(FloatPredicate::OLT, hi, x, "lt").unwrap();
        let x = self.b.build_select(lt, hi, x, "ch").unwrap().into_float_value();
        let xl = self.b.build_float_mul(x, self.cf32(LOG2E), "xl").unwrap();
        let re = inkwell::intrinsics::Intrinsic::find("llvm.roundeven")
            .unwrap()
            .get_declaration(m, &[self.f32_t.into()])
            .unwrap();
        let n = self
            .b
            .build_call(re, &[xl.into()], "n")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let r = self.fma_f32(m, n, self.cf32(-LN2_HI), x);
        let r = self.fma_f32(m, n, self.cf32(-LN2_LO), r);
        let mut p = self.cf32(C[6]);
        for k in (0..6).rev() {
            p = self.fma_f32(m, p, r, self.cf32(C[k]));
        }
        let ni = self.b.build_float_to_signed_int(n, self.i32_t, "ni").unwrap();
        let add = self.b.build_int_add(ni, self.i32_t.const_int(127, false), "e").unwrap();
        let shl = self.b.build_left_shift(add, self.i32_t.const_int(23, false), "bits").unwrap();
        let pf = self.b.build_bit_cast(shl, self.f32_t, "p2").unwrap().into_float_value();
        self.b.build_float_mul(p, pf, "exp").unwrap()
    }
}
```

- [ ] **Step 5: Emit the attention function itself**

Still in `attn_emit.rs`:

```rust
/// Emit the geometry-specialized hspan attention function. `head_dim` must
/// be a multiple of 8 (existing kernel contract); `n_heads % n_kv_heads == 0`.
pub(crate) fn emit_attn_hspan_fn<'c>(
    ctx: &'c Context,
    module: &Module<'c>,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    name: &str,
    linkage: Linkage,
) -> FunctionValue<'c> {
    assert!(head_dim % 8 == 0, "head_dim must be a multiple of 8");
    assert!(n_heads % n_kv_heads == 0, "GQA group must divide");
    let e = AttnEmitCtx::new(ctx);
    let group = (n_heads / n_kv_heads) as u64;
    let kv_dim = (n_kv_heads * head_dim) as u64;
    let chunks = head_dim / 8;
    // Baked exactly as the kernel computes it at runtime (deterministic).
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let fn_ty = ctx.void_type().fn_type(
        &[
            e.ptr_t.into(), e.ptr_t.into(), e.ptr_t.into(), e.ptr_t.into(), // out q kv scores
            e.i64_t.into(), e.i64_t.into(), e.i64_t.into(),                 // kv_base v_off pos
            e.i64_t.into(), e.i64_t.into(), e.i64_t.into(), e.i64_t.into(), // (ignored geometry)
            e.i64_t.into(), e.i64_t.into(),                                 // h_start h_end
        ],
        false,
    );
    let f = module.add_function(name, fn_ty, Some(linkage));
    let entry = ctx.append_basic_block(f, "entry");
    e.b.position_at_end(entry);

    let out = f.get_nth_param(0).unwrap().into_pointer_value();
    let q = f.get_nth_param(1).unwrap().into_pointer_value();
    let kv = f.get_nth_param(2).unwrap().into_pointer_value();
    let scores = f.get_nth_param(3).unwrap().into_pointer_value();
    let kv_base = f.get_nth_param(4).unwrap().into_int_value();
    let v_off = f.get_nth_param(5).unwrap().into_int_value();
    let pos = f.get_nth_param(6).unwrap().into_int_value();
    // params 7..=10 (kv_dim, n_heads, n_kv_heads, head_dim): IGNORED — baked.
    let h_start = f.get_nth_param(11).unwrap().into_int_value();
    let h_end = f.get_nth_param(12).unwrap().into_int_value();

    let visible = e.b.build_int_add(pos, e.ci64(1), "visible").unwrap();
    let kreg = kv_base;
    let vreg = e.b.build_int_add(kv_base, v_off, "vreg").unwrap();

    // Entry allocas (hoisted out of loops so mem2reg promotes them).
    let maxa = e.b.build_alloca(e.f32_t, "max").unwrap();
    let dena = e.b.build_alloca(e.f32_t, "denom").unwrap();
    let acc = e.b.build_alloca(e.v8_t().array_type(chunks as u32), "avacc").unwrap();

    e.loop_range(f, h_start, h_end, 1, "head", |e, h| {
        let g = e.b.build_int_unsigned_div(h, e.ci64(group), "g").unwrap();
        let hl = e.b.build_int_sub(h, h_start, "hl").unwrap();
        let hoff = e.b.build_int_mul(hl, e.ci64(head_dim as u64), "hoff").unwrap();
        let qh = e.fptr(q, hoff);
        let ghd = e.b.build_int_mul(g, e.ci64(head_dim as u64), "ghd").unwrap();

        // -- scores[t] = reduce8(Σ_c fma(q8, k8)) * scale, t ascending --
        e.loop_range(f, e.ci64(0), visible, 1, "sc", |e, t| {
            let tkv = e.b.build_int_mul(t, e.ci64(kv_dim), "tkv").unwrap();
            let kb = e.b.build_int_add(
                e.b.build_int_add(kreg, tkv, "kb0").unwrap(), ghd, "kb").unwrap();
            let mut a8 = e.v8_t().const_zero();
            for c in 0..chunks {
                let q8 = e.load_v8(e.fptr(qh, e.ci64((c * 8) as u64)));
                let koff = e.b.build_int_add(kb, e.ci64((c * 8) as u64), "ko").unwrap();
                let k8 = e.load_v8(e.fptr(kv, koff));
                a8 = e.fma_v8(module, q8, k8, a8);
            }
            let dot = e.reduce8(a8);
            let s = e.b.build_float_mul(dot, e.cf32(scale), "s").unwrap();
            e.store_f32(e.fptr(scores, t), s);
        });

        // -- max: sequential f32::max fold from NEG_INFINITY --
        e.store_f32(maxa, e.cf32(f32::NEG_INFINITY));
        e.loop_range(f, e.ci64(0), visible, 1, "mx", |e, t| {
            let m0 = e.load_f32(maxa);
            let s = e.load_f32(e.fptr(scores, t));
            let mn = inkwell::intrinsics::Intrinsic::find("llvm.maxnum")
                .unwrap()
                .get_declaration(module, &[e.f32_t.into()])
                .unwrap();
            let m1 = e.b.build_call(mn, &[m0.into(), s.into()], "m")
                .unwrap().try_as_basic_value().left().unwrap().into_float_value();
            e.store_f32(maxa, m1);
        });
        let maxv = e.load_f32(maxa);
        let max8 = e.splat(maxv);

        // -- exp + denom: blocks of 8 with reduce8, then scalar tail --
        e.store_f32(dena, e.cf32(0.0));
        let blocks = e.b.build_and(visible, e.ci64(!7u64), "blk").unwrap(); // visible & !7
        e.loop_range(f, e.ci64(0), blocks, 8, "ex8", |e, t| {
            let sp = e.fptr(scores, t);
            let v = e.load_v8(sp);
            let xm = e.b.build_float_sub(v, max8, "xm").unwrap();
            let ev = e.expf_v8(module, xm);
            e.store_v8(sp, ev);
            let d0 = e.load_f32(dena);
            let d1 = e.b.build_float_add(d0, e.reduce8(ev), "d").unwrap();
            e.store_f32(dena, d1);
        });
        e.loop_range(f, blocks, visible, 1, "ext", |e, t| {
            let sp = e.fptr(scores, t);
            let xm = e.b.build_float_sub(e.load_f32(sp), maxv, "xm").unwrap();
            let ev = e.expf_f32(module, xm);
            e.store_f32(sp, ev);
            let d0 = e.load_f32(dena);
            let d1 = e.b.build_float_add(d0, ev, "d").unwrap();
            e.store_f32(dena, d1);
        });
        let denom = e.load_f32(dena);

        // -- AV: acc chunks zeroed; t ascending: acc_c = fma(splat(w/denom), v8, acc_c) --
        for c in 0..chunks {
            let slot = e.fptr(acc, e.ci64((c * 8) as u64));
            e.store_v8(slot, e.v8_t().const_zero());
        }
        e.loop_range(f, e.ci64(0), visible, 1, "av", |e, t| {
            let w = e.load_f32(e.fptr(scores, t));
            let wn = e.b.build_float_div(w, denom, "wn").unwrap();
            let w8 = e.splat(wn);
            let tkv = e.b.build_int_mul(t, e.ci64(kv_dim), "tkv").unwrap();
            let vb = e.b.build_int_add(
                e.b.build_int_add(vreg, tkv, "vb0").unwrap(), ghd, "vb").unwrap();
            for c in 0..chunks {
                let slot = e.fptr(acc, e.ci64((c * 8) as u64));
                let v8 = e.load_v8(e.fptr(kv, e.b.build_int_add(vb, e.ci64((c * 8) as u64), "vo").unwrap()));
                let a0 = e.load_v8(slot);
                let a1 = e.fma_v8(module, w8, v8, a0);
                e.store_v8(slot, a1);
            }
        });
        // store the accumulated row to out[hl*head_dim ..]
        let oh = e.fptr(out, hoff);
        for c in 0..chunks {
            let slot = e.fptr(acc, e.ci64((c * 8) as u64));
            e.store_v8(e.fptr(oh, e.ci64((c * 8) as u64)), e.load_v8(slot));
        }
    });
    e.b.build_return(None).unwrap();
    f
}
```

Note on the `acc` alloca: `fptr(acc, c*8)` treats it as f32 elements — 8 f32 per v8 chunk, so chunk `c` starts at element `c*8`. The bit-neutrality argument for register/alloca accumulation vs the kernel's in-memory `oh[d]` read-modify-write: the per-lane FMA sequence is identical (`wn.mul_add(v, acc)` per `d`, `t` ascending); only where the intermediate lives differs, which cannot change bits.

Register `mod attn_emit;` in `crates/inferno-codegen/src/llvm/mod.rs` (alongside the existing `mod ops;`), and `pub(crate) use attn_emit::emit_attn_hspan_fn;` if `ops.rs` imports via the parent.

- [ ] **Step 6: Add `compile_attn_probe` to `emit.rs`**

Mirror `compile()`'s TargetMachine + `cc -shared` steps (reuse its imports; keep the same `OptimizationLevel::Aggressive` / `RelocMode::PIC`):

```rust
/// M4b.16 test-harness entry: compile a one-function `.so` exporting the
/// geometry-specialized attention as `inferno_attn_probe`. `cpu_features`
/// is the LLVM feature string ("+avx2,+fma" or "" for baseline x86-64) —
/// the harness checks the emitted function is bit-identical under BOTH.
pub fn compile_attn_probe(
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    cpu_features: &str,
    out_dir: &Path,
) -> Result<PathBuf> {
    let ctx = Context::create();
    let module = ctx.create_module("attn_probe");
    crate::llvm::emit_attn_hspan_fn(
        &ctx, &module, head_dim, n_heads, n_kv_heads,
        "inferno_attn_probe", inkwell::module::Linkage::External,
    );
    module.verify().map_err(|e| CodegenError::Emit(e.to_string()))?;
    Target::initialize_x86(&InitializationConfig::default());
    let triple = TargetMachine::get_default_triple();
    let tgt = Target::from_triple(&triple).map_err(|e| CodegenError::Emit(e.to_string()))?;
    let tm = tgt
        .create_target_machine(
            &triple, "x86-64", cpu_features,
            OptimizationLevel::Aggressive, RelocMode::PIC, CodeModel::Default,
        )
        .ok_or_else(|| CodegenError::Emit("no target machine".into()))?;
    std::fs::create_dir_all(out_dir)?;
    let obj = out_dir.join("attn_probe.o");
    tm.write_to_file(&module, FileType::Object, &obj)
        .map_err(|e| CodegenError::Emit(e.to_string()))?;
    let so = out_dir.join("attn_probe.so");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(cc)
        .args(["-shared", "-o"]).arg(&so).arg(&obj)
        .status()?;
    if !status.success() {
        return Err(CodegenError::Emit("probe link failed".into()));
    }
    Ok(so)
}
```

(Adapt the `module.verify()` / `write_to_file` calls to whatever the post-Task-1 inkwell 0.9 API looks like in `compile()` — copy its exact idioms, including the `raw_module()` wrapper if `compile()` uses one. Export `compile_attn_probe` from `lib.rs` next to `compile`.)

- [ ] **Step 7: Run the test until bit-identical**

```bash
cargo test -p inferno-codegen --test attn_emit -- --nocapture
```

Expected: PASS. Debug protocol if bits differ: first divergence will usually be (a) missing align-4 (crash, not mismatch), (b) `fmuladd` contraction (check emitted asm has `vfmadd` and IR has `llvm.fma`), (c) denominator order (blocks-of-8 + tail, not sequential), (d) clamp select operand order. Dump IR with `module.print_to_stderr()` behind a temporary env check while debugging; remove before commit.

- [ ] **Step 8: Lint, commit**

```bash
mise run lint && mise run test
git add -A && git commit -m "M4b.16: emit_attn_hspan_fn + compile_attn_probe + first bit-exactness test"
```

---

### Task 5: Bit-exactness harness — geometries × ISAs × spans × pos sweep

**Files:**
- Modify: `crates/inferno-codegen/tests/attn_emit.rs`

**Interfaces:**
- Consumes: `build_probe` / `assert_bit_identical` from Task 4 (same file).
- Produces: the milestone's standing guard suite; Task 11's closing verdict cites it by name.

- [ ] **Step 1: Add the sweep tests**

Append to `tests/attn_emit.rs`:

```rust
/// Geometries: protocol (qwen 0.5b), MHA, phi-ish hd80, hd96, llama-7b-ish.
const GEOMETRIES: &[(usize, usize, usize)] = &[
    (64, 14, 2),  // protocol: qwen2.5-0.5b
    (64, 8, 8),   // MHA (group = 1)
    (80, 10, 2),  // phi-family head_dim
    (96, 12, 4),
    (128, 32, 8), // llama-7b-class
];

/// pos values: tiny, the 8-block boundary, both sides of the M4b.15
/// inversion region (>= 1023), and a mid value.
const POSITIONS: &[usize] = &[0, 1, 7, 8, 9, 300, 1022, 1023, 1024, 1500];

fn sweep(features: &str, reference: AttnFn, tag: &str) {
    for &(hd, nh, nkv) in GEOMETRIES {
        let p = build_probe(hd, nh, nkv, features);
        for (pi, &pos) in POSITIONS.iter().enumerate() {
            // spans: full, first head, last head, uneven split point
            let spans = [(0, nh), (0, 1), (nh - 1, nh), (1, (nh / 2).max(2))];
            for (si, &(h0, h1)) in spans.iter().enumerate() {
                let seed = 0x4b16_0000u64 | ((pi as u64) << 8) | si as u64;
                assert_bit_identical(&p, reference, hd, nh, nkv, pos, h0, h1, seed);
            }
        }
        drop(p); // one probe .so per geometry; tempdir cleaned here
    }
    eprintln!("sweep[{tag}]: {} geometries x {} pos x 4 spans OK", GEOMETRIES.len(), POSITIONS.len());
}

#[test]
fn emitted_avx2_features_matches_scalar_rust() {
    sweep("+avx2,+fma", inferno_kernels::inferno_attention_f32_scalar_hspan, "avx2/scalar-oracle");
}

#[test]
fn emitted_baseline_features_matches_scalar_rust() {
    // "" = x86-64 baseline (SSE2): the <8 x float> IR legalizes to 2x4-wide;
    // per-lane order is unchanged, so bits must still match exactly.
    sweep("", inferno_kernels::inferno_attention_f32_scalar_hspan, "sse2/scalar-oracle");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn emitted_matches_avx2_rust_kernel() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    sweep("+avx2,+fma", inferno_kernels::inferno_attention_f32_avx2_hspan, "avx2/avx2-oracle");
}
```

- [ ] **Step 2: Run the harness**

```bash
cargo test -p inferno-codegen --test attn_emit -- --nocapture
```

Expected: PASS, three sweeps. Runtime note: 5 geometries × 2 feature sets ≈ 10 probe compiles + 5 more for the avx2-oracle sweep, each sub-second — the suite should stay under ~2 min. If pos=1500 dominates, it is the KV fill (`lcg_fill` over 2·seq_len·kv_dim floats), not the compile; keep it — the ≥1023 region is exactly where M4b.15 saw inversions.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "M4b.16: bit-exactness sweep (5 geometries x 2 ISAs x 10 pos x 4 spans, avx2-oracle cross-check)"
```

---

### Task 6: Wire into `lower_attention` behind the flag + same-logits invariant

**Files:**
- Modify: `crates/inferno-codegen/src/llvm/ops.rs` (`lower_attention`, ~line 1492; CodeGen struct + constructor for flag/caching)
- Test: `crates/inferno-core/tests/artifact.rs`

**Interfaces:**
- Consumes: `emit_attn_hspan_fn` (Task 4), `CompileOptions::emitted_attn` (Task 2).
- Produces: compiled artifacts whose decode attention dispatches to `attn_hspan.emitted` when the flag is set. Task 8's script relies on `INFERNO_EMITTED_ATTN=0/1` producing the two artifact variants.

- [ ] **Step 1: Write the failing same-logits test**

In `crates/inferno-core/tests/artifact.rs` (following `compiled_prefill_matches_interpreter`'s helpers — `use_temp_cache`, `model_path`, buffer sizing; use a `max_seq_len` no other test uses, e.g. 96, for cache isolation per the file's convention):

```rust
/// M4b.16: the emitted-attention artifact must be BIT-IDENTICAL to the
/// runtime-symbol artifact — prefill logits and every decode_step's logits.
/// (The attn_emit harness proves the function; this proves the wiring.)
#[test]
fn emitted_attn_artifact_logits_bit_identical() {
    use_temp_cache();
    let model = model_path();
    let target = TargetDesc::detect().unwrap();
    let base_opts = CompileOptions::default();
    let lever_opts = CompileOptions { emitted_attn: true, ..Default::default() };
    let base = Artifact::load_or_compile(&model, &target, 96, &base_opts).unwrap();
    let lever = Artifact::load_or_compile(&model, &target, 96, &lever_opts).unwrap();

    let desc = load_desc(&model).unwrap();
    let vocab = desc.hyperparams.vocab_size as usize;
    let tokens = vec![1u32, 4, 7, 2];

    let run = |art: &Artifact| -> Vec<Vec<u32>> {
        let mut kv = vec![0f32; art.meta().kv_total_bytes / 4];
        let mut arena = vec![0f32; art.meta().arena_f32];
        let mut logits = vec![0f32; vocab];
        let mut all = Vec::new();
        art.prefill(&tokens, 0, &mut kv, &mut arena, &mut logits);
        all.push(logits.iter().map(|v| v.to_bits()).collect());
        for step in 0..8 {
            let tok = (3 + step as u32) % 11; // deterministic token walk
            art.decode_step(tok, tokens.len() + step, &mut kv, &mut arena, &mut logits);
            all.push(logits.iter().map(|v| v.to_bits()).collect());
        }
        all
    };
    assert_eq!(run(&base), run(&lever), "emitted attention changed logits bits");
}
```

- [ ] **Step 2: Run it — expect trivial pass, then make it meaningful**

```bash
cargo test -p inferno-core --test artifact emitted_attn_artifact_logits_bit_identical
```

Expected now: PASS **trivially** (the flag is plumbed but not yet consumed — both artifacts are byte-identical modulo cache dir). This step exists to lock the test in green before the wiring; the meaningful red/green is Step 4.

- [ ] **Step 3: Consume the flag in `lower_attention`**

In `ops.rs`: give `CodeGen` the flag and a lazily-emitted cached function. Add fields (thread them from `opts` wherever the constructor already receives `opts` for `profile` — follow that exact path):

```rust
    /// M4b.16: emit + dispatch the geometry-specialized attention fn.
    emitted_attn: bool,
    /// Lazily emitted `attn_hspan.emitted` (one per module; layers share it).
    emitted_attn_fn: std::cell::RefCell<Option<FunctionValue<'c>>>,
```

In `lower_attention` (~line 1520), replace the `afn` binding:

```rust
        let afn = if self.emitted_attn {
            *self.emitted_attn_fn.borrow_mut().get_or_insert_with(|| {
                super::attn_emit::emit_attn_hspan_fn(
                    self.ctx, &self.module, head_dim, n_heads, n_kv_heads,
                    "attn_hspan.emitted", Linkage::Private,
                )
            })
        } else {
            let sym = crate::loopir::attention_hspan_symbol(isa);
            self.module
                .get_function(&sym)
                .expect("hspan attention kernel declared")
        };
```

(Keep the `isa`/`sym` lines inside the else branch; the runtime-symbol declaration in `llvm/mod.rs` stays unconditional — it is the fallback ABI surface. `emit_attn_hspan_fn` uses its own builder and appends its own blocks, so the caller's insertion point is untouched — same discipline as `par_token_loop` documents.)

One-model-one-geometry note: all layers share `n_heads`/`n_kv_heads`/`head_dim` (they come from `HyperParams`), so a single cached function is correct. If a future architecture ever varies geometry per layer, the cache key must grow — leave a one-line comment saying exactly that at the `RefCell` field.

- [ ] **Step 4: Red/green the wiring**

Temporarily break one baked constant (e.g. emit `scale * 2.0`) and run the artifact test — it must FAIL (proves the lever artifact actually routes through the emitted fn). Revert the sabotage, run again — PASS. Also:

```bash
cargo test -p inferno-codegen --test attn_emit
cargo test -p inferno-codegen --test differential
cargo test -p inferno-core --test artifact
```

Expected: all PASS. (differential/artifact default paths are untouched — flag defaults false.)

- [ ] **Step 5: Lint, full suite, commit**

```bash
mise run test && mise run lint
git add -A && git commit -m "M4b.16: lower_attention dispatches attn_hspan.emitted under emitted_attn; same-logits artifact invariant"
```

---

### Task 7: Local admission + AGENTS.md + local context bench

**Files:**
- Modify: `AGENTS.md` (decode-attention paragraph)
- Modify: spec §Amendments (local context record)

- [ ] **Step 1: Full local admission run**

```bash
mise run test && mise run lint
cargo test -p inferno-codegen --test attn_emit
cargo test -p inferno-codegen --test differential
cargo test -p inferno-core --test artifact
git diff main --stat -- crates/inferno-graph/src/tolerance.rs   # empty
git diff main --stat -- crates/inferno-kernels/                 # expf.rs + lib.rs visibility only
```

All green / as expected before any money is spent.

- [ ] **Step 2: Local dev-box context point (never a gate)**

```bash
cargo build --release -p inferno
INFERNO_EMITTED_ATTN=0 target/release/inferno bench models/qwen2.5-0.5b-instruct-q8_0.gguf --pp 512 --tg 128 --reps 5 --threads 0
INFERNO_EMITTED_ATTN=1 target/release/inferno bench models/qwen2.5-0.5b-instruct-q8_0.gguf --pp 512 --tg 128 --reps 5 --threads 0
```

Record both tg numbers in the spec §Amendments under a dated "local context point (non-quiet dev box)" heading, honestly labeled, with the machine name. Sanity check only: if the lever is *slower* locally by more than noise, stop and investigate the emitted asm before scheduling sessions.

- [ ] **Step 3: AGENTS.md paragraph**

In the decode-threading bullet, after the M4b.11 head-sharding sentences, add:

```markdown
  Since M4b.16 the decode attention kernel dispatched through
  `inferno_par_attention_heads` is, when `CompileOptions::emitted_attn`
  is set (env `INFERNO_EMITTED_ATTN`), a codegen-emitted private
  `attn_hspan.emitted` function with the model's geometry baked in as
  constants — bit-identical to the Rust hspan kernels (the
  `attn_emit` harness in `inferno-codegen` is the guard; the Rust
  kernels stay the rig oracle and the fallback). The flag is part of
  the artifact cache key.
```

(Adjust the final sentence when Task 11 flips the default — the paragraph must state the shipping default.)

- [ ] **Step 4: Commit + spec amendment commit**

```bash
git add AGENTS.md docs/superpowers/specs/2026-07-18-m4b16-emitted-decode-attention-design.md
git commit -m "M4b.16: local admission record, context bench point, AGENTS.md decode-attention paragraph"
```

---

### Task 8: Quiet-hw session script

**Files:**
- Create: `scripts/quiet-hw/gate-emitted-attn.sh` (mode 755)

**Interfaces:**
- Consumes: `INFERNO_EMITTED_ATTN` plumbing (Task 2), lever wiring (Task 6), `scripts/quiet-hw/lib.sh` helpers (`smoke_header`, `machine_block`, `phys_cores`).
- Produces: the lever-vs-baseline table both sessions run. Fresh llama.cpp baselines come from the existing `gate-bench-protocol.sh`, run separately in the session checklist.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# M4b.16 gate — emitted-attention lever vs runtime-symbol baseline, same
# binary, same box. Prints both bench tables; the gate arithmetic
# (tg_lever/tg_base per the M4b.11 thresholds: >=5% both boxes ship,
# <3% both STOP) is HUMAN — record tables and verdict in the M4b.16 spec
# §Amendments. Fresh llama.cpp baselines are gate-bench-protocol.sh's
# job (run it in the same session; toolchain changed, they are mandatory).
# Usage: gate-emitted-attn.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-emitted-attn.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PP=32; TG=8; REPS=1; else PP=512; TG=128; REPS=5; fi

smoke_header "gate-emitted-attn (M4b.16 lever vs baseline)"
machine_block
echo

cargo build --release -q -p inferno

echo "== baseline (INFERNO_EMITTED_ATTN=0) =="
INFERNO_EMITTED_ATTN=0 target/release/inferno bench "$MODEL" \
  --pp "$PP" --tg "$TG" --reps "$REPS" --threads 0 \
  | tee "$OUT/bench-baseline.txt"

echo "== lever (INFERNO_EMITTED_ATTN=1) =="
INFERNO_EMITTED_ATTN=1 target/release/inferno bench "$MODEL" \
  --pp "$PP" --tg "$TG" --reps "$REPS" --threads 0 \
  | tee "$OUT/bench-lever.txt"

echo
echo "artifacts: $OUT (bench-baseline.txt, bench-lever.txt)"
echo "VERDICT IS HUMAN: compute tg_lever/tg_base; thresholds per M4b.16 spec."
```

- [ ] **Step 2: Selftest + smoke locally**

```bash
chmod 755 scripts/quiet-hw/gate-emitted-attn.sh
bash -n scripts/quiet-hw/gate-emitted-attn.sh
QHW_SMOKE=1 scripts/quiet-hw/gate-emitted-attn.sh models/qwen2.5-0.5b-instruct-q8_0.gguf
```

Expected: smoke run prints both tables (tiny pp/tg counts) and the artifact paths.

- [ ] **Step 3: Commit**

```bash
git add scripts/quiet-hw/gate-emitted-attn.sh
git commit -m "M4b.16: gate-emitted-attn session script (lever vs baseline, human verdict)"
```

---

### Task 9: Quiet-hw Session A — 16c `d2.c1.medium` (6336Y)

**Operator-driven. Real money.** Follow `docs/runbooks/metal.md`; no parallel provisions; `mise run metal-gc` after ANY interrupted session.

- [ ] **Step 1: Provision + prep per runbook**, push the `m4b16-emitted-attn` branch, build in devenv shell.
- [ ] **Step 2: Fresh llama.cpp baselines (mandatory — toolchain changed):**

```bash
QHW_OUT=/tmp/m4b16-a scripts/quiet-hw/gate-bench-protocol.sh models/qwen2.5-0.5b-instruct-q8_0.gguf
```

- [ ] **Step 3: Lever vs baseline:**

```bash
QHW_OUT=/tmp/m4b16-a scripts/quiet-hw/gate-emitted-attn.sh models/qwen2.5-0.5b-instruct-q8_0.gguf
```

- [ ] **Step 4: Record** — protocol tables verbatim into the M4a spec §Amendments; lever/baseline tables + machine block into the M4b.16 spec §Amendments under "Session A". No verdict yet (needs both boxes).
- [ ] **Step 5: Deprovision; `mise run metal-gc`; confirm zero servers.** Commit the amendment.

```bash
git add docs/superpowers/specs/ && git commit -m "M4b.16: Session A (16c 6336Y) records"
```

---

### Task 10: Quiet-hw Session B — 8c `s2.c2.medium` (E-2388G)

Identical protocol to Task 9 on the 8c box; record under "Session B".

- [ ] **Step 1–5:** as Task 9 with `QHW_OUT=/tmp/m4b16-b`; commit `"M4b.16: Session B (8c E-2388G) records"`.

---

### Task 11: Gate verdict, default flip, closing verdict, PR

**Files:**
- Modify: spec §Amendments (gate verdict + closing verdict)
- Modify (ship only): `crates/inferno-codegen/src/emit.rs` (default), `cli/src/main.rs` (kill-switch semantics), `AGENTS.md` (default wording)

- [ ] **Step 1: Gate arithmetic in the spec §Amendments** — `r_box = tg_lever / tg_base` per box from the session tables; apply the ladder verbatim (≥5% both → ship; <3% both → STOP; else recorded judgment with the argument written out — the spec's §Risks pre-registers the 16c-ships/8c-3-to-5% split as the case needing explicit judgment).
- [ ] **Step 2 (SHIP outcome only): flip the default.**

`emit.rs` `Default`: `emitted_attn: true`. CLI env reads become a kill-switch:

```rust
            emitted_attn: std::env::var("INFERNO_EMITTED_ATTN").map_or(true, |v| v != "0"),
```

Update the AGENTS.md sentence to "default-on; `INFERNO_EMITTED_ATTN=0` is the kill-switch". Re-run `mise run test` (the same-logits test now exercises default-on against an explicit `false` baseline — update its two `CompileOptions` literals to be explicit rather than default-relative:  `CompileOptions { emitted_attn: false, ..Default::default() }` / `{ emitted_attn: true, .. }` — do this edit NOW in this step so the test is default-independent).

**(STOP outcome only):** leave defaults; record the STOP as a finding (M4b.12 precedent) — the harness, probe, and flag stay in-tree as shipped-but-default-off diagnostic instruments.

- [ ] **Step 3: Closing verdict walk** in the spec §Amendments — the eight exit criteria from §Exit criteria, each answered YES/NO with evidence pointers (test names, session headings, `git diff main -- crates/inferno-graph/src/tolerance.rs` empty, `git diff main -- crates/inferno-kernels/` visibility-only). Include the required residual-decode-wall-shape statement with numbers from the session data.
- [ ] **Step 4: Final full gates + PR**

```bash
mise run test && mise run lint
git push -u origin m4b16-emitted-attn
gh pr create --title "M4b.16: codegen-emitted geometry-specialized decode attention" \
  --body "Spec + gate records: docs/superpowers/specs/2026-07-18-m4b16-emitted-decode-attention-design.md. Lever gated per the two-box ladder; see §Amendments for the verdict."
```

---

## Self-Review Notes

- Spec coverage: Prerequisite → Task 1; flag/cache → Task 2; emission + bit-exactness → Tasks 3–5; wiring + same-logits → Task 6; local admission + AGENTS.md → Task 7; sessions + fresh llama baselines → Tasks 8–10; gate ladder, default flip, closing verdict, STOP-as-finding → Task 11. Zero-tolerance-edit and kernels-byte-untouched checks appear in Tasks 1, 6, 7, 11.
- The inkwell API surface in Tasks 4/6 is written against the 0.9-era API as best known; where `compile()`'s post-Task-1 idioms differ (e.g. `raw_module()`, verify signatures), **copy `compile()`'s working idioms** — that instruction is normative, not a placeholder.
- Type consistency: `emit_attn_hspan_fn(ctx, module, head_dim, n_heads, n_kv_heads, name, linkage)` is used identically in Task 4 Step 6 and Task 6 Step 3; `CompileOptions::emitted_attn` is the field name everywhere; the probe symbol is `inferno_attn_probe` in both the emitter call and the test loader.
