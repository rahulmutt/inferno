# M5 — Apple Silicon NEON Bring-up & Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port inferno's compiled inference path to Apple Silicon (M1, pure LLVM-NEON, AMX unused) and record one honest quiet-hardware baseline vs llama.cpp with the gap attributed — no performance gate.

**Architecture:** Three sequential slices (spec Approach A: *toolchain first, tune second*). Slice 1 brings the whole darwin path up end to end — aarch64 target descriptor, LLVM aarch64 target-init, `.dylib` link, macOS dynamic-symbol resolution, `dlopen` — routing heavy compute to the **portable scalar kernels** (the spec's "naive/portable" path: zero NEON intrinsics, so the toolchain is proven before any kernel work). Slice 2 replaces scalar with real NEON microkernels, each bit-identical to its scalar reference and enforced by the existing kernel rig. Slice 3 stands up a single-Mac quiet-hw runbook and records the baseline. Every change is **additive to x86**: aarch64 code is `#[cfg(...)]`-gated and no x86 path is altered.

**Tech Stack:** Rust 1.97.1 (mise-pinned), inkwell 0.9 / LLVM 22.1.8 (devenv), `libloading` (dlopen), `rustix` (mmap), `proptest` (rig), llama.cpp `llama-bench` (comparator), macOS `sysctl`/`thread_policy`/`powermetrics` (detection + quiet-hw).

## Global Constraints

_Copied verbatim from the spec and AGENTS.md; every task's requirements implicitly include this section._

- **Additive to x86.** NEON sits beside AVX2/AVX-512, selected by the target descriptor. No x86 path is removed or altered in behavior; every existing x86 gate stays green in CI (`mise run test`, `mise run lint`, kernel rig, both differentials, the nightly speedup gate).
- **No perf gate this milestone.** "Correct + measured + attributed" is the exit. No performance assertion, no win claim of any kind.
- **AMX/SME deliberately unused.** Pure LLVM-NEON codegen only. Do **not** emit AMX/SME instructions or link Accelerate or any vendor BLAS.
- **Bit-identity is a hard contract** (`inferno-kernels/src/lib.rs:4-7`): every f32 op happens in the same order with the same fusing in every ISA variant. NEON kernels must use FMA (`vfmaq_f32`) in the same accumulation order as the scalar reference and route partial strips through the shared scalar path, exactly as the AVX2 kernels do. The rig's `*_isa_variants_bitwise_equal` proptests enforce this.
- **Tolerances are evidenced, never fudged** (AGENTS.md): `LOGIT_TIE_EPSILON`, `gemv_rel_tol`, `logits_abs_tol` are re-derived from observed ARM error distributions (the rig's `observed_error_*` sweeps), recorded in the spec's Amendments, and **never** nudged to make a red test green.
- **Recorded data points are append-only.** The baseline lands once in the spec's Amendments and is never edited.
- **Toolchain pins must match** (AGENTS.md:16-18): inkwell feature major (`llvm22-1`) and `LLVM_SYS_221_PREFIX` must both be LLVM 22 on the Mac.
- **Target chip:** Apple M1 (base/Pro/Max — exact tier pinned in Task 1). AMX gen1, **no SME**, 128-bit NEON (4×f32).
- **Metal budget: zero.** No cloud Macs, no `metal/` work.

## Environment note (read first)

Tasks are tagged **[any]** (compiles/tests on Linux CI or the Mac) or **[mac]** (requires the physical M1 — darwin toolchain, on-device gates, baseline). The devcontainer this plan may be authored in is Linux; **every [mac] task executes on the M1**, inside `devenv shell` with LLVM 22.1.8 and `mise install` done. `#[cfg(target_arch = "aarch64")]` / `#[cfg(target_os = "macos")]` code is cfg'd out on Linux, so [any] tasks keep x86 CI green while the aarch64 code only builds/runs on the Mac.

---

## Slice 1 — Toolchain & scalar bring-up on aarch64-apple-darwin

Deliverable of the slice: a correct compiled path on the M1 with the codegen differential, artifact differential, and kernel rig all green on-device, using scalar kernels. x86 CI unchanged.

### Task 1: aarch64 target descriptor + macOS detection

**Files:**
- Modify: `crates/inferno-target/src/desc.rs:11-44` (add `Isa`/`Feature`/topology fields)
- Create: `crates/inferno-target/src/detect_macos.rs` (the sysctl detection path)
- Modify: `crates/inferno-target/src/detect.rs:12-66` (dispatch to the macOS path on aarch64/macos)
- Modify: `crates/inferno-target/src/lib.rs` (declare the new module, `#[cfg(target_os = "macos")]`)
- Test: `crates/inferno-target/src/detect_macos.rs` (inline `#[cfg(test)]`)

**Interfaces:**
- Produces: `Isa::Aarch64Neon` variant; `Feature::{Dotprod, I8mm}`; `CoreTopology { physical_cores, logical_cores, smt, perf_cores: Option<u32>, eff_cores: Option<u32> }`; `TargetDesc::detect()` returns `Ok(TargetDesc)` on Apple Silicon.
- Consumes: nothing from other tasks.

Pin the exact M1 first. On the Mac, record the tier so the spec's Task-1 Amendment can cite it:

- [ ] **Step 1: Record the machine [mac]**

Run: `sysctl -n machdep.cpu.brand_string hw.model hw.physicalcpu hw.logicalcpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.l1dcachesize hw.l2cachesize hw.cachelinesize hw.pagesize hw.memsize; sysctl -a | grep -iE 'hw.optional.(neon|arm)'`
Expected: prints the brand (e.g. `Apple M1 Pro`), P/E core split, cache sizes, page size, and NEON/dotprod/i8mm optional flags. Paste this block into the spec's `## Amendments` under a `### Task 1 — machine pinned` heading (do not commit code yet; this is the recorded machine identity).

- [ ] **Step 2: Extend the descriptor enums (write the change) [any]**

In `crates/inferno-target/src/desc.rs`, add to `Isa` (currently lines 12-19, `X86_64v3`/`X86_64v4`):

```rust
    /// Apple Silicon / ARMv8-A with mandatory NEON (128-bit Advanced SIMD).
    /// AMX/SME are deliberately not modeled — inferno emits pure NEON (M5).
    Aarch64Neon,
```

Add to `Feature` (currently lines 22-27, `Vnni`/`Bf16`):

```rust
    /// ARM dotprod (SDOT/UDOT) — present on Apple M1+.
    Dotprod,
    /// ARM i8mm (SMMLA) — present on Apple M1+.
    I8mm,
```

Extend `CoreTopology` (currently lines 39-44) with P/E fields, defaulted so x86 detection is unaffected:

```rust
pub struct CoreTopology {
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub smt: bool,
    /// Performance-core count on heterogeneous chips (Apple P/E). `None` on flat SMP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perf_cores: Option<u32>,
    /// Efficiency-core count on heterogeneous chips. `None` on flat SMP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eff_cores: Option<u32>,
}
```

Update the x86 `parse_topology` (`detect.rs`) construction site to set `perf_cores: None, eff_cores: None` so it still compiles.

- [ ] **Step 3: Write the failing detection test [mac]**

Create `crates/inferno-target/src/detect_macos.rs` with the test first:

```rust
//! macOS/aarch64 hardware detection via sysctl. Populates the same
//! `TargetDesc` the x86 sysfs path produces (crate-doc equivalence contract).
#![cfg(target_os = "macos")]

// ... detect_macos() defined in Step 5 ...

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::{Feature, Isa};

    #[test]
    fn detect_populates_apple_silicon_descriptor() {
        let d = detect_macos().expect("detection must succeed on Apple Silicon");
        assert_eq!(d.isa, Isa::Aarch64Neon);
        assert!(d.features.contains(&Feature::Dotprod));
        assert!(d.topology.physical_cores >= 4);
        assert!(d.topology.perf_cores.unwrap_or(0) >= 1, "P-core count must be detected");
        assert!(d.page_size >= 4096);
        assert!(!d.caches.is_empty(), "at least L1d + L2 must be detected");
    }
}
```

- [ ] **Step 4: Run it to verify it fails [mac]**

Run: `cargo test -p inferno-target --lib detect_populates_apple_silicon_descriptor`
Expected: FAIL — `detect_macos` not defined.

- [ ] **Step 5: Implement `detect_macos` [mac]**

In `crates/inferno-target/src/detect_macos.rs`, above the test module. Read scalars via `sysctlbyname` (use the `libc` crate — already transitively available via `rustix`, or add `libc` to `Cargo.toml` `[target.'cfg(target_os = "macos")'.dependencies]`):

```rust
use crate::desc::{BwClass, CacheLevel, CoreTopology, Feature, Isa, TargetDesc};
use crate::error::{TargetError, TargetResult};
use std::collections::BTreeSet;
use std::ffi::CString;

fn sysctl_u64(name: &str) -> Option<u64> {
    let cname = CString::new(name).ok()?;
    let mut val: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: writing a u64 out-param sized exactly; name is NUL-terminated.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            (&mut val as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && len == std::mem::size_of::<u64>() { Some(val) } else { None }
}

fn sysctl_flag(name: &str) -> bool {
    sysctl_u64(name).unwrap_or(0) != 0
}

pub fn detect_macos() -> TargetResult<TargetDesc> {
    // NEON is mandatory on ARMv8; treat its absence as a non-Apple-Silicon host.
    if !sysctl_flag("hw.optional.neon") && !sysctl_flag("hw.optional.AdvSIMD") {
        return Err(TargetError::UnsupportedPlatform {
            detail: "NEON not reported by sysctl; not an Apple Silicon host".into(),
        });
    }
    let mut features = BTreeSet::new();
    if sysctl_flag("hw.optional.arm.FEAT_DotProd") { features.insert(Feature::Dotprod); }
    if sysctl_flag("hw.optional.arm.FEAT_I8MM") { features.insert(Feature::I8mm); }

    let physical = sysctl_u64("hw.physicalcpu").unwrap_or(0) as u32;
    let logical = sysctl_u64("hw.logicalcpu").unwrap_or(physical as u64) as u32;
    let perf = sysctl_u64("hw.perflevel0.physicalcpu").map(|v| v as u32);
    let eff = sysctl_u64("hw.perflevel1.physicalcpu").map(|v| v as u32);
    if physical == 0 {
        return Err(TargetError::UnsupportedPlatform {
            detail: "hw.physicalcpu unavailable".into(),
        });
    }
    let topology = CoreTopology {
        physical_cores: physical,
        logical_cores: logical,
        smt: logical > physical, // Apple has no SMT; this stays false
        perf_cores: perf,
        eff_cores: eff,
    };

    let line = sysctl_u64("hw.cachelinesize").unwrap_or(128) as u32;
    let mut caches = Vec::new();
    if let Some(l1) = sysctl_u64("hw.perflevel0.l1dcachesize").or_else(|| sysctl_u64("hw.l1dcachesize")) {
        caches.push(CacheLevel { level: 1, size_bytes: l1, line_bytes: line, shared_by: 1 });
    }
    if let Some(l2) = sysctl_u64("hw.perflevel0.l2cachesize").or_else(|| sysctl_u64("hw.l2cachesize")) {
        // Apple L2 is shared per performance cluster.
        caches.push(CacheLevel { level: 2, size_bytes: l2, line_bytes: line,
            shared_by: perf.unwrap_or(1) });
    }

    Ok(TargetDesc {
        isa: Isa::Aarch64Neon,
        features,
        page_size: sysctl_u64("hw.pagesize").unwrap_or(16384),
        memory_bw_class: None, // profile-only, never detected (matches x86 path)
        topology,
        caches,
    })
}
```

Add to `crates/inferno-target/src/lib.rs`: `#[cfg(target_os = "macos")] mod detect_macos;`. Add `libc = "0.2"` under `[target.'cfg(target_os = "macos")'.dependencies]` in `crates/inferno-target/Cargo.toml`.

- [ ] **Step 6: Dispatch `detect()` to the macOS path [any]**

In `crates/inferno-target/src/detect.rs`, replace the non-x86 error arm (lines 61-66) so that on macOS it calls `detect_macos`, and only genuinely-unsupported hosts error:

```rust
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub fn detect() -> TargetResult<TargetDesc> {
    crate::detect_macos::detect_macos()
}

#[cfg(not(any(target_arch = "x86_64", all(target_arch = "aarch64", target_os = "macos"))))]
pub fn detect() -> TargetResult<TargetDesc> {
    Err(TargetError::UnsupportedPlatform {
        detail: "only x86-64 (M2) and aarch64-apple-darwin (M5) detection implemented".into(),
    })
}
```

Keep the existing `#[cfg(target_arch = "x86_64")]` `detect()` unchanged. Ensure `TargetDesc::detect()` (the public wrapper at `detect.rs:12-24`) forwards to the arch-selected free function; if it currently inlines the x86 body, extract the x86 body into `#[cfg(target_arch = "x86_64")] fn detect()` and have the wrapper call the cfg-selected one.

- [ ] **Step 7: Run the detection test [mac]**

Run: `cargo test -p inferno-target --lib detect_populates_apple_silicon_descriptor`
Expected: PASS.

- [ ] **Step 8: Verify x86 unaffected [any]**

Run: `cargo test -p inferno-target` (on Linux CI or an x86 box)
Expected: PASS — existing `detect_matches_expected_profile` (`detect.rs:283`) and all target tests still green; the new `Option` topology fields serialize away when `None`.

- [ ] **Step 9: Commit**

```bash
git add crates/inferno-target/
git commit -m "target: aarch64 Isa + macOS sysctl detection (M5 Task 1)"
```

### Task 2: make inferno-kernels compile on aarch64 (scalar path)

**Files:**
- Modify: `crates/inferno-kernels/src/lib.rs:16-46` (relax the arch `compile_error!`; cfg the AVX2 re-exports)
- Modify: `crates/inferno-kernels/src/registry.rs:282-330` (`kernels_for` / `attention_kernel`: aarch64 → scalar)
- Modify: `crates/inferno-plan/src/weights.rs:52` (planner Isa→kernel mapping accepts aarch64)

**Interfaces:**
- Consumes: `Isa::Aarch64Neon` (Task 1).
- Produces: `inferno-kernels` builds on aarch64; `kernels_for(dtype, Isa::Aarch64Neon)` returns the **scalar** `KernelSet`; `attention_kernel(Isa::Aarch64Neon)` returns the scalar attention fn. (Slice 2 flips these to NEON.)

- [ ] **Step 1: Relax the whole-crate arch gate [any]**

In `crates/inferno-kernels/src/lib.rs`, replace the hard block at lines 16-17:

```rust
#[cfg(not(target_arch = "x86_64"))]
compile_error!("inferno-kernels is x86-64-only until the v2 NEON milestone");
```

with an allow-list that admits aarch64:

```rust
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("inferno-kernels supports x86-64 (M2) and aarch64 (M5) only");
```

The AVX2 symbol re-exports and `KernelIsa::Avx2` are already `#[cfg(target_arch = "x86_64")]`-gated (`lib.rs:38-43`), so they vanish on aarch64; the scalar symbols are exported unconditionally (`lib.rs:44-46`) and are portable Rust. No scalar kernel body changes.

- [ ] **Step 2: Route aarch64 → scalar in the registry [any]**

In `crates/inferno-kernels/src/registry.rs`, `kernels_for` (lines 282-291) currently matches only x86 `Isa` arms. Add the aarch64 arm mapping to `KernelIsa::Scalar` for now:

```rust
pub fn kernels_for(dtype: &DType, isa: Isa) -> Option<KernelSet> {
    let kisa = match isa {
        Isa::X86_64v3 | Isa::X86_64v4 => KernelIsa::Avx2,
        Isa::Aarch64Neon => KernelIsa::Scalar, // Slice 1: scalar bring-up; Slice 2 → Neon
    };
    if !kisa.available() { return None; }
    set(dtype, kisa)
}
```

Same for `attention_kernel` (lines 320-330): add `Isa::Aarch64Neon => KernelIsa::Scalar`, and ensure the `#[cfg(target_arch = "x86_64")]` fast-path block is skipped on aarch64 so it falls through to `crate::inferno_attention_f32_scalar` (already the non-x86 default).

- [ ] **Step 3: Fix the planner's Isa match [any]**

In `crates/inferno-plan/src/weights.rs` (around line 52, `kernels_for(&dtype, target.isa)`), the call is generic over `Isa`; confirm any *other* exhaustive `match target.isa` in the planner gains an `Isa::Aarch64Neon` arm. Grep: `grep -rn "match .*isa" crates/inferno-plan/src`. Add arms mirroring the x86 behavior (same packing; rs8 is ISA-independent).

- [ ] **Step 4: Build on aarch64 [mac]**

Run: `cargo build -p inferno-kernels -p inferno-plan`
Expected: compiles clean on the M1.

- [ ] **Step 5: Run the kernel rig with scalar only [mac]**

Run: `cargo test -p inferno-kernels --test rig`
Expected: PASS. `KernelIsa::all_available()` returns `[Scalar]` on aarch64 (no `Avx2` arm compiled), so every `*_isa_variants_bitwise_equal` proptest hits its `if !KernelIsa::Avx2.available() { return Ok(()) }` early-return and the oracle tests run scalar-vs-oracle. No NEON arm yet.

- [ ] **Step 6: Verify x86 rig unaffected [any]**

Run: `cargo test -p inferno-kernels --test rig` (x86)
Expected: PASS — scalar-vs-AVX2 bitwise tests still active and green.

- [ ] **Step 7: Commit**

```bash
git add crates/inferno-kernels/ crates/inferno-plan/
git commit -m "kernels: compile + scalar dispatch on aarch64 (M5 Task 2)"
```

### Task 3: artifact filename helper (`.so` vs `.dylib`)

**Files:**
- Modify: `crates/inferno-codegen/src/lib.rs` (add `ARTIFACT_LIB_FILENAME`)
- Modify: `crates/inferno-codegen/src/emit.rs:127` (use the const for the model lib)
- Modify: `crates/inferno-core/src/artifact.rs:297,490` (use the const)
- Modify: `crates/inferno-codegen/tests/differential.rs:96` (use the const)

**Interfaces:**
- Produces: `pub const inferno_codegen::ARTIFACT_LIB_FILENAME: &str` — `"model.dylib"` on macOS, `"model.so"` elsewhere.
- Consumes: nothing.

- [ ] **Step 1: Define the const [any]**

In `crates/inferno-codegen/src/lib.rs` (near `HOST_ABI_VERSION`):

```rust
/// Filename of the compiled model shared object inside an artifact dir.
/// macOS links a Mach-O dylib; ELF platforms a `.so`. The extension lives
/// only here plus the emitter's link command — cache_key never stores it.
pub const ARTIFACT_LIB_FILENAME: &str =
    if cfg!(target_os = "macos") { "model.dylib" } else { "model.so" };
```

- [ ] **Step 2: Replace the hardcoded names [any]**

- `crates/inferno-codegen/src/emit.rs:127`: `let so = out_dir.join(crate::ARTIFACT_LIB_FILENAME);`
- `crates/inferno-core/src/artifact.rs:297`: `libloading::Library::new(dir.join(inferno_codegen::ARTIFACT_LIB_FILENAME))`
- `crates/inferno-core/src/artifact.rs:490`: `dir.join(inferno_codegen::ARTIFACT_LIB_FILENAME).exists()`
- `crates/inferno-codegen/tests/differential.rs:96`: `art_dir.join(inferno_codegen::ARTIFACT_LIB_FILENAME)`

Leave the attn-probe (`emit.rs:220`) as `attn_probe.so` for now — it is only reached when `emitted_attn` is on, which stays off on aarch64 (Task 12 note); still, for macOS correctness of that path, apply the same treatment there in Step 3.

- [ ] **Step 3: Probe filename too [any]**

`crates/inferno-codegen/src/emit.rs:220`: replace `"attn_probe.so"` with a `cfg!`-selected `"attn_probe.dylib"`/`"attn_probe.so"` (inline `if cfg!(target_os = "macos")`), matching the link command changed in Task 4.

- [ ] **Step 4: Build [any]**

Run: `cargo build -p inferno-codegen -p inferno-core`
Expected: compiles on both x86 and (later) aarch64. On x86 the const is `"model.so"` — behavior identical to before.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-codegen/ crates/inferno-core/
git commit -m "codegen: single-source the artifact lib filename (.so/.dylib) (M5 Task 3)"
```

### Task 4: aarch64 LLVM target-init + dylib link

**Files:**
- Modify: `crates/inferno-codegen/src/emit.rs:105` (arch-conditional target init, `compile`)
- Modify: `crates/inferno-codegen/src/emit.rs:203-215` (probe target init + host CPU literal)
- Modify: `crates/inferno-codegen/src/emit.rs:128-134` (link command: `-shared`→`-dynamiclib`, add dynamic_lookup)
- Modify: `crates/inferno-codegen/src/emit.rs:220-230` (probe link command)

**Interfaces:**
- Consumes: `ARTIFACT_LIB_FILENAME` (Task 3).
- Produces: `compile(...)` emits an aarch64 object and links `model.dylib` on macOS. No signature change.

- [ ] **Step 1: Add a native-target-init helper [any]**

In `crates/inferno-codegen/src/emit.rs`, add:

```rust
fn initialize_native_target() {
    #[cfg(target_arch = "x86_64")]
    inkwell::targets::Target::initialize_x86(&inkwell::targets::InitializationConfig::default());
    #[cfg(target_arch = "aarch64")]
    inkwell::targets::Target::initialize_aarch64(&inkwell::targets::InitializationConfig::default());
}
```

Replace `Target::initialize_x86(...)` at `emit.rs:105` and `emit.rs:203` with `initialize_native_target();`. At `emit.rs:209`, replace the hardcoded CPU literal `"x86-64"` with `&TargetMachine::get_host_cpu_name().to_string()` so the probe also builds on aarch64. (`get_default_triple` + `get_host_cpu_name`/`get_host_cpu_features` are already host-relative and need no change — they auto-produce `apple-m1` + NEON features.)

- [ ] **Step 2: Make the link command arch-aware [any]**

Replace the link block at `emit.rs:128-134`:

```rust
let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
let mut cmd = Command::new(cc);
#[cfg(target_os = "macos")]
{
    // Mach-O dylib whose undefined kernel symbols resolve against the host
    // process at dlopen time (the macOS analog of ELF -rdynamic).
    cmd.arg("-dynamiclib")
        .arg("-Wl,-undefined,dynamic_lookup");
}
#[cfg(not(target_os = "macos"))]
{
    cmd.arg("-shared");
}
let status = cmd.arg("-o").arg(&so).arg(&obj).status()?;
if !status.success() {
    return Err(CodegenError::Link(format!("linker exited {status}")));
}
```

Apply the identical `#[cfg]` treatment to the probe link at `emit.rs:220-230`.

- [ ] **Step 3: Build codegen on aarch64 [mac]**

Run: `cargo build -p inferno-codegen`
Expected: compiles on the M1.

- [ ] **Step 4: Emit+link a dylib directly [mac]**

Run: `cargo test -p inferno-codegen --test differential differential_tiny_gguf -- --nocapture`
Expected: this exercises `compile(...)` → `.o` → `cc -dynamiclib` → `model.dylib`. It may still FAIL at the dlopen/symbol-resolution step (fixed in Task 5) — but it must **get past the link step** (no `CodegenError::Link`). If it fails, capture the linker stderr; a `-Wl,-undefined,dynamic_lookup` rejection here means the installed `ld` is too old — note it as the Task-4 risk.

- [ ] **Step 5: Verify x86 link unchanged [any]**

Run: `cargo test -p inferno-codegen --test differential` (x86)
Expected: PASS — `-shared`/`model.so` path is byte-for-byte the prior behavior.

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-codegen/
git commit -m "codegen: aarch64 target-init + macOS dylib link (M5 Task 4)"
```

### Task 5: macOS host dynamic-symbol export (dlopen resolution)

**Files:**
- Modify: `crates/inferno-core/build.rs:21` (macOS export-dynamic for tests)
- Modify: `crates/inferno-codegen/build.rs:16` (macOS export-dynamic for tests)
- Modify: `cli/build.rs:15` (macOS export-dynamic for the real binary)

**Interfaces:**
- Consumes: the dylib from Task 4.
- Produces: host binaries export their kernel symbols so `dlopen`ed `model.dylib` resolves `inferno_*` / `inferno_par_*` at load. This is what makes Task 6's gates pass.

- [ ] **Step 1: Replace `-rdynamic` per-OS in all three build scripts [any]**

Each build script currently emits an ELF `-rdynamic` link arg. Make it arch/OS-aware. For `crates/inferno-core/build.rs:21` (`cargo:rustc-link-arg-tests=-rdynamic`):

```rust
if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
    // Mach-O: export the host's symbols so a dlopen'd model.dylib linked with
    // -undefined dynamic_lookup can bind inferno_* kernels at load time.
    println!("cargo:rustc-link-arg-tests=-Wl,-export_dynamic");
} else {
    println!("cargo:rustc-link-arg-tests=-rdynamic");
}
```

Apply the same shape to `crates/inferno-codegen/build.rs:16` (also `-link-arg-tests`) and `cli/build.rs:15` (which uses `-link-arg-bins` — keep `bins`, swap `-rdynamic`→`-Wl,-export_dynamic` on macOS).

- [ ] **Step 2: Confirm kernel-symbol retention still holds [mac]**

The differential test already `black_box`-references every kernel via `retain_kernel_symbols` (`differential.rs:58-83`) so the linker keeps them. No change needed, but verify on aarch64 that the retained symbols are the scalar ones (AVX2 symbols don't exist on aarch64 — the `#[cfg]`-gated references in that function must compile; if `retain_kernel_symbols` names AVX2 symbols unconditionally, add `#[cfg(target_arch = "x86_64")]` around those lines).

- [ ] **Step 3: Build the CLI on aarch64 [mac]**

Run: `cargo build -p inferno --release`
Expected: links clean with `-Wl,-export_dynamic`.

- [ ] **Step 4: Verify x86 unchanged [any]**

Run: `cargo build -p inferno && cargo test -p inferno-core --test artifact` (x86)
Expected: PASS — `-rdynamic` path untouched on Linux.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-core/build.rs crates/inferno-codegen/build.rs cli/build.rs
git commit -m "build: macOS export-dynamic for dlopen symbol resolution (M5 Task 5)"
```

### Task 6: Slice-1 gate — correct scalar path green on-device

**Files:**
- Test only (no source changes expected; if a gate fails, the fix belongs in Tasks 1-5 and this task re-runs).
- Modify: `crates/inferno-pool/examples/gemv_stream.rs:306` (guard `Advice::LinuxHugepage` behind `cfg(target_os = "linux")` if it blocks `mise run test`)

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: recorded on-device green run — the Slice-1 milestone.

- [ ] **Step 1: Guard the Linux-only madvise example [any]**

`crates/inferno-pool/examples/gemv_stream.rs:306` calls `rustix::mm::madvise(..., Advice::LinuxHugepage)` — Linux-only. Wrap that call in `#[cfg(target_os = "linux")]` (with a no-op `#[cfg(not(target_os = "linux"))]` branch) so the workspace builds on macOS. It is an experiment, not the compiled path.

- [ ] **Step 2: The three correctness gates on-device [mac]**

Run each inside `devenv shell` on the M1:

```bash
cargo test -p inferno-codegen --test differential
cargo test -p inferno-core --test artifact
cargo test -p inferno-kernels --test rig
```

Expected: all PASS. The differential is the primary gate — compiled (scalar-kernel) path vs the scalar interpreter, last-token logits within `logits_abs_tol`. If the differential exceeds tolerance, do **not** widen the tolerance; it means the LLVM-autovectorized elementwise ops (rmsnorm/softmax/rope, `ops.rs`) reordered a reduction on NEON. Capture the observed max abs error and defer the evidenced tolerance decision to Task 14 — but first try `-C target-feature` / confirm no fast-math crept in (`emit.rs:108-110` deliberately sets none).

- [ ] **Step 3: End-to-end smoke run [mac]**

Run: `cargo run -p inferno --release -- run crates/inferno-formats/tests/fixtures/tiny.gguf --prompt "the" --max-tokens 4`
Expected: compiles the fixture to `model.dylib`, `dlopen`s it, generates 4 tokens without error. Then re-run to confirm the artifact cache hit (no recompile).

- [ ] **Step 4: Full blocking-tier suite on-device [mac]**

Run: `mise run test` then `mise run lint`
Expected: PASS on the M1. (`mise run test` runs `cargo nextest run --workspace`; the aarch64 code paths are now exercised.)

- [ ] **Step 5: Record + commit [mac]**

Append to the spec's `## Amendments` a `### Task 6 — Slice 1 on-device (scalar)` note: the three gate results, the smoke run, and the machine. Then:

```bash
git add crates/inferno-pool/examples/gemv_stream.rs docs/superpowers/specs/2026-07-19-m5-apple-silicon-neon-baseline-design.md
git commit -m "pool: guard Linux-only madvise; record Slice-1 on-device green (M5 Task 6)"
```

---

## Slice 2 — Real NEON microkernels

Deliverable: pure-NEON compiled path, every kernel bit-identical to its scalar reference (rig-enforced), all gates green on-device, tolerances re-derived with ARM evidence. Each kernel task follows the same TDD shape: add the NEON symbol + the rig's NEON bitwise arm (failing), implement, go green. NEON kernels **route partial strips through the shared scalar path** exactly as the AVX2 kernels do (`f32k.rs:91-132` is the reference structure).

### Task 7: `KernelIsa::Neon` variant + rig scaffolding

**Files:**
- Modify: `crates/inferno-kernels/src/lib.rs:83-107` (`KernelIsa` enum + `available()`/`all_available()`)
- Modify: `crates/inferno-kernels/tests/rig.rs` (add `Neon` arms to the ~12 `match isa` drivers; generalize the `Avx2`-only early-returns)

**Interfaces:**
- Produces: `KernelIsa::Neon`; `KernelIsa::Neon.available()` = `std::arch::is_aarch64_feature_detected!("neon")`; rig drivers dispatch a `Neon` arm.
- Consumes: nothing new.

- [ ] **Step 1: Add the enum variant [any]**

In `crates/inferno-kernels/src/lib.rs`, extend `KernelIsa` (lines 83-107):

```rust
pub enum KernelIsa { Scalar, Avx2, Neon }

impl KernelIsa {
    pub fn available(self) -> bool {
        match self {
            KernelIsa::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            KernelIsa::Avx2 => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(not(target_arch = "x86_64"))]
            KernelIsa::Avx2 => false,
            #[cfg(target_arch = "aarch64")]
            KernelIsa::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(not(target_arch = "aarch64"))]
            KernelIsa::Neon => false,
        }
    }
}
```

`all_available()` filters by `available()`, so on aarch64 it returns `[Scalar, Neon]` and on x86 `[Scalar, Avx2]`.

- [ ] **Step 2: Add `Neon` arms to the rig drivers [any]**

In `crates/inferno-kernels/tests/rig.rs`, each `match isa` driver (`gemv_f32` :70, `gemv_q8_0` :324, `gemv_q4_k` :709, and the `gemm_*` peers) is exhaustive over `{Scalar, Avx2}`. Add a `KernelIsa::Neon => <neon symbol>` arm to each — the symbols are created in Tasks 8-11; until then, point the arm at the scalar symbol so the crate compiles (the arm becomes real per kernel task). Generalize the bitwise-equality early-returns: replace `if !KernelIsa::Avx2.available() { return Ok(()); }` with a helper that picks the available SIMD ISA:

```rust
fn simd_isa() -> Option<KernelIsa> {
    KernelIsa::all_available().into_iter().find(|i| *i != KernelIsa::Scalar)
}
```

and rewrite each `*_isa_variants_bitwise_equal` to `let Some(simd) = simd_isa() else { return Ok(()) };` then compare `Scalar` vs `simd`. This makes the same proptest assert scalar-vs-NEON on the Mac and scalar-vs-AVX2 on x86, with no duplication.

- [ ] **Step 3: Build + run rig (still scalar-effective) [mac]**

Run: `cargo test -p inferno-kernels --test rig`
Expected: PASS — the `Neon` arms currently alias scalar, so bitwise equality is trivially true; nothing regresses. This is the scaffold.

- [ ] **Step 4: x86 rig unaffected [any]**

Run: `cargo test -p inferno-kernels --test rig` (x86)
Expected: PASS — `simd_isa()` returns `Avx2`, identical coverage to before.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/
git commit -m "kernels: KernelIsa::Neon variant + rig scalar-vs-simd generalization (M5 Task 7)"
```

### Task 8: NEON f32 GEMV + GEMM (worked reference)

**Files:**
- Modify: `crates/inferno-kernels/src/f32k.rs` (add `_neon` symbols beside `_scalar`/`_avx2`)
- Modify: `crates/inferno-kernels/tests/rig.rs` (point the f32 `Neon` driver arms at the new symbols)

**Interfaces:**
- Consumes: `GemvFn`/`GemmFn` ABI (`registry.rs:20-22`), `STRIP=8` (`lib.rs:66`), the scalar `gemv_rows`/`gemm_scalar` fallbacks (`f32k.rs:54-66`).
- Produces: `inferno_gemv_f32_rs8_neon`, `inferno_gemm_f32_rs8_neon` — bit-identical to `_scalar`.

- [ ] **Step 1: Wire the rig arm to the (not-yet-existing) symbol [any]**

In `rig.rs`, the f32 `Neon` arm from Task 7 → `f32k::inferno_gemv_f32_rs8_neon` (and gemm peer). This makes the rig reference an undefined symbol → the failing test.

- [ ] **Step 2: Run to verify it fails [mac]**

Run: `cargo test -p inferno-kernels --test rig f32`
Expected: FAIL — `inferno_gemv_f32_rs8_neon` not found.

- [ ] **Step 3: Implement the NEON f32 GEMV [mac]**

In `crates/inferno-kernels/src/f32k.rs`, translating the AVX2 structure (`f32k.rs:91-132`) to NEON. `STRIP=8` rows = two 128-bit vectors (4+4); each output row is a single-lane FMA chain across `k`, identical order to scalar, so bit-identical:

```rust
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_f32_rs8_neon(
    y: *mut f32, x: *const u8, w: *const u8, k: usize, row_start: usize, row_end: usize,
) {
    use std::arch::aarch64::*;
    let xf = x.cast::<f32>();
    let wf = w.cast::<f32>();
    let mut r = row_start;
    let head = row_start.next_multiple_of(STRIP).min(row_end);
    if head > r { unsafe { gemv_rows(y, xf, wf, k, r, head) }; r = head; } // scalar head
    while r + STRIP <= row_end {
        let base = unsafe { wf.add((r / STRIP) * k * STRIP) };
        let mut acc0 = unsafe { vdupq_n_f32(0.0) }; // rows r..r+4
        let mut acc1 = unsafe { vdupq_n_f32(0.0) }; // rows r+4..r+8
        for c in 0..k {
            let wv0 = unsafe { vld1q_f32(base.add(c * STRIP)) };
            let wv1 = unsafe { vld1q_f32(base.add(c * STRIP + 4)) };
            let xv = unsafe { vdupq_n_f32(xf.add(c).read_unaligned()) };
            acc0 = unsafe { vfmaq_f32(acc0, wv0, xv) };
            acc1 = unsafe { vfmaq_f32(acc1, wv1, xv) };
        }
        unsafe { vst1q_f32(y.add(r), acc0) };
        unsafe { vst1q_f32(y.add(r + 4), acc1) };
        r += STRIP;
    }
    if r < row_end { unsafe { gemv_rows(y, xf, wf, k, r, row_end) }; } // scalar tail
}
```

Implement `inferno_gemm_f32_rs8_neon` analogously from `inferno_gemm_f32_rs8_avx2` (`f32k.rs:176`), keeping the same per-(row,col-of-x) accumulation nesting. Do **not** use a reduction across lanes — each lane is an independent output row, preserving scalar order.

- [ ] **Step 4: Run the rig to verify bit-identity [mac]**

Run: `cargo test -p inferno-kernels --test rig f32`
Expected: PASS — `f32_isa_variants_bitwise_equal` now compares scalar vs NEON and asserts `to_bits()` equality per row; `f32_gemv_matches_oracle`, the range-partition, and `gemm(m=1)==gemv` tests pass for the `Neon` ISA. If a bit differs, the accumulation order diverges from scalar — align it; never relax `gemv_rel_tol`.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/src/f32k.rs crates/inferno-kernels/tests/rig.rs
git commit -m "kernels: NEON f32 gemv+gemm, bit-identical to scalar (M5 Task 8)"
```

### Task 9: NEON Q8_0 GEMV + GEMM

**Files:**
- Modify: `crates/inferno-kernels/src/q8_0.rs` (add `_neon` symbols)
- Modify: `crates/inferno-kernels/tests/rig.rs` (q8_0 `Neon` arms)

**Interfaces:**
- Consumes: rs8-packed Q8_0 layout (`pack_q8_0_rs8`, `q8_0.rs:44`); scalar `_scalar` reference (`q8_0.rs:89`); the AVX2 reference (`q8_0.rs:150`, incl. `hsum8_i32` at :128).
- Produces: `inferno_gemv_q8_0_rs8_neon`, `inferno_gemm_q8_0_rs8_neon`.

- [ ] **Step 1: Point the rig q8_0 `Neon` arms at the new symbols; run to fail [mac]**

Run: `cargo test -p inferno-kernels --test rig q8_0`
Expected: FAIL — symbol not found.

- [ ] **Step 2: Implement, translating the AVX2 kernel [mac]**

Q8_0 dot products accumulate in **i32** (integer add is associative → tree vs sequential is bitwise-identical, which is why Q8_0 tolerates a vector reduction where f32 does not). Use `vdotq_s32` (dotprod is present on M1 — `Feature::Dotprod` from Task 1; assert it) or `vmull`/`vpadalq` fallback, then apply the per-block f32 scale exactly as `_scalar` does. Mirror the AVX2 body at `q8_0.rs:150-247`: same block iteration, same scale-application order, `hsum` via `vaddvq_s32` (full reduction of the i32 lanes — associative, safe). The final f32 accumulation of per-block `sum * scale` must match the scalar order.

Concrete inner shape (per row-strip, per 32-elem block):

```rust
// SAFETY blocks omitted for brevity; wrap each intrinsic as elsewhere in the file.
use std::arch::aarch64::*;
let mut iacc = vdupq_n_s32(0);
// load 16 i8 weights + 16 i8 activations per half-block, dotprod-accumulate
iacc = vdotq_s32(iacc, w_i8, x_i8);
let block_dot = vaddvq_s32(iacc);              // i32, order-independent
facc += (block_dot as f32) * w_scale * x_scale; // same order as _scalar
```

Implement gemm from `inferno_gemm_q8_0_rs8_avx2` (`q8_0.rs:299`) with the same tiling.

- [ ] **Step 3: Verify bit-identity [mac]**

Run: `cargo test -p inferno-kernels --test rig q8_0`
Expected: PASS — `q8_0_isa_variants_bitwise_equal` (:382), `q8_0_gemm_prefill_tile_matches_per_token_gemv` (:534), pack-inverse, and oracle tests green for `Neon`.

- [ ] **Step 4: Commit**

```bash
git add crates/inferno-kernels/src/q8_0.rs crates/inferno-kernels/tests/rig.rs
git commit -m "kernels: NEON Q8_0 gemv+gemm via dotprod, bit-identical (M5 Task 9)"
```

### Task 10: NEON Q4_K GEMV + GEMM

**Files:**
- Modify: `crates/inferno-kernels/src/q4_k.rs` (add `_neon` symbols)
- Modify: `crates/inferno-kernels/tests/rig.rs` (q4_k `Neon` arms)

**Interfaces:**
- Consumes: `pack_q4_k_rs8` (`q4_k.rs:37`); `_scalar` (`q4_k.rs:89`) and `_avx2` (`q4_k.rs:144`) references.
- Produces: `inferno_gemv_q4_k_rs8_neon`, `inferno_gemm_q4_k_rs8_neon`.

- [ ] **Step 1: Wire rig arms, run to fail [mac]**

Run: `cargo test -p inferno-kernels --test rig q4_k`
Expected: FAIL — symbol not found.

- [ ] **Step 2: Implement from the AVX2 reference [mac]**

Q4_K has the super-block structure (256-elem, 6-bit scales/mins). Translate `inferno_gemv_q4_k_rs8_avx2` (`q4_k.rs:144-300`) op-for-op to NEON: nibble-unpack with `vandq_u8`/`vshrq_n_u8`, widen, i32 dotprod-accumulate per sub-block (associative), then apply the sub-block scale/min in the **same f32 order** as `_scalar`. This is the most intricate translation; the rig's `observed_error_q4_k` sweep (`rig.rs:981`) and the bitwise test are the spec.

- [ ] **Step 3: Verify bit-identity [mac]**

Run: `cargo test -p inferno-kernels --test rig q4_k`
Expected: PASS — `q4_k_isa_variants_bitwise_equal` (:767), pack-inverse, oracle green for `Neon`.

- [ ] **Step 4: Commit**

```bash
git add crates/inferno-kernels/src/q4_k.rs crates/inferno-kernels/tests/rig.rs
git commit -m "kernels: NEON Q4_K gemv+gemm, bit-identical (M5 Task 10)"
```

### Task 11: NEON activation quantize (q8a / q8k)

**Files:**
- Modify: `crates/inferno-kernels/src/act.rs` (add `_neon` symbols)
- Modify: `crates/inferno-kernels/tests/rig.rs` and/or `crates/inferno-kernels/src/act.rs` tests (quantize `Neon` arm)

**Interfaces:**
- Consumes: `QuantFn` ABI (`registry.rs:19`); `_scalar` refs (`act.rs:43,63`); block consts `Q8A_BLOCK=32`, `Q8K_BLOCK=256` (`act.rs:14-17`).
- Produces: `inferno_quantize_row_q8a_neon`, `inferno_quantize_row_q8k_neon`; extend `set()` (Task 13) to select them.

- [ ] **Step 1: Wire the quantize `Neon` arm in the rig; run to fail [mac]**

Run: `cargo test -p inferno-kernels --test rig quantize` (or the act tests)
Expected: FAIL — symbol not found.

- [ ] **Step 2: Implement from the AVX2 quantizers [mac]**

Translate `inferno_quantize_row_q8a_avx2` (`act.rs:155`) and `_q8k_avx2` (`act.rs:172`): per block, find max-abs (`vmaxvq_f32` over `vabsq_f32`), compute the scale exactly as scalar, multiply-round-clamp to i8 (`vcvtnq_s32_f32` for round-to-nearest-even matching scalar `.round()`; verify the rounding mode matches `_scalar`'s). The rounding-mode match is the bitwise-identity crux here.

- [ ] **Step 3: Verify [mac]**

Run: `cargo test -p inferno-kernels --test rig` (quantize + downstream q8_0/q4_k oracle tests that consume quantized activations)
Expected: PASS — quantize `Neon` bit-identical to scalar; no downstream drift.

- [ ] **Step 4: Commit**

```bash
git add crates/inferno-kernels/src/act.rs crates/inferno-kernels/tests/rig.rs
git commit -m "kernels: NEON activation quantize q8a/q8k, bit-identical (M5 Task 11)"
```

### Task 12: expf_neon + NEON attention (single / qblock / hspan)

**Files:**
- Modify: `crates/inferno-kernels/src/expf.rs` (add `expf_neon`)
- Modify: `crates/inferno-kernels/src/attention.rs` (add `_neon` variants of the three kernels)
- Modify: `crates/inferno-kernels/tests/rig.rs` (attention `Neon` drivers + guards)

**Interfaces:**
- Consumes: `expf_scalar` polynomial (`expf.rs:37`) — the exact poly softmax bit-parity depends on; `AttnFn` ABI (`registry.rs:299`); the three scalar refs (`attention.rs:56,167,309`).
- Produces: `expf_neon(float32x4_t) -> float32x4_t`; `inferno_attention_f32_neon`, `_neon_qblock`, `_neon_hspan`.

- [ ] **Step 1: Write `expf_neon` matching the scalar poly [mac]**

In `crates/inferno-kernels/src/expf.rs`, replicate `expf_scalar`'s polynomial (`expf.rs:37`) over `float32x4_t` — same constants, same Horner order, same range reduction. Add a rig/unit test asserting `expf_neon` lanes bit-match `expf_scalar` per element over a sweep (mirror how `expf_avx2` is validated).

- [ ] **Step 2: Wire attention `Neon` rig drivers; run to fail [mac]**

The attention rig section (`rig.rs:1012+`) hard-wires `attn_kernel_scalar`/`_avx2` drivers rather than `match isa`. Add `attn_kernel_neon` / `_neon_qblock` / `_neon_hspan` drivers referencing the new symbols, plus the availability guard mirrored from `attn_kernel_avx2`.
Run: `cargo test -p inferno-kernels --test rig attn`
Expected: FAIL — symbols not found.

- [ ] **Step 3: Implement the three NEON attention kernels [mac]**

Translate `inferno_attention_f32_scalar` (:56), `_scalar_qblock` (:167), `_scalar_hspan` (:309) — **not** the AVX2 ones — using `expf_neon`, keeping the scalar kernel's 4-lane-friendly partition and **fixed reduction tree** so softmax is bit-identical. (The AVX2 attention uses an 8-lane partition; NEON must use its own 4-lane partition that still reduces in an order bitwise-equal to scalar — the scalar reference is the oracle, not AVX2.) Match head/kv layout params exactly (`kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim`, plus qblock's `pos0, m_block, q_stride, out_stride` and hspan's head-range args).

- [ ] **Step 4: Verify attention bit-identity [mac]**

Run: `cargo test -p inferno-kernels --test rig attn`
Expected: PASS — scalar-vs-NEON bitwise for single/qblock/hspan; `attn_rel_tol` oracle match; hspan tiling equals whole-call. Note: keep codegen `emitted_attn` **off** on aarch64 (the `<8 x float>` emitter in `attn_emit.rs:51` is AVX2-shaped and out of scope for M5); the microkernel path above is the decode/prefill attention on ARM.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/src/expf.rs crates/inferno-kernels/src/attention.rs crates/inferno-kernels/tests/rig.rs
git commit -m "kernels: expf_neon + NEON attention (single/qblock/hspan), bit-identical (M5 Task 12)"
```

### Task 13: flip aarch64 dispatch to NEON; re-run all gates on-device

**Files:**
- Modify: `crates/inferno-kernels/src/registry.rs:212-330` (`set()` NEON arms; `kernels_for`/`attention_kernel` aarch64 → `KernelIsa::Neon`)

**Interfaces:**
- Consumes: all NEON symbols (Tasks 8-12).
- Produces: the compiled path on aarch64 now dispatches NEON end to end.

- [ ] **Step 1: Add `Neon` arms to `set()` [any]**

In `crates/inferno-kernels/src/registry.rs`, the three `match isa` blocks in `set()` (F32 :223-230, Q8_0 :240-251, Q4_K :261-272) are exhaustive over `{Scalar, Avx2}`. Add `KernelIsa::Neon => <_neon symbol>` for `gemv`, `gemm`, and (Q8_0/Q4_K) `quantize`, and for the F32/activation quantize peers. These `_neon` symbols are `#[cfg(target_arch = "aarch64")]`, so wrap the `Neon` arms in the same cfg (or provide an unreachable scalar alias on x86, since `Neon.available()` is false there and the arm is never taken).

- [ ] **Step 2: Flip the ISA mapping [any]**

Change the Slice-1 placeholder in `kernels_for` (Task 2): `Isa::Aarch64Neon => KernelIsa::Neon`. Same in `attention_kernel`, returning the NEON attention fn when `KernelIsa::Neon.available()`.

- [ ] **Step 3: The three gates, now NEON, on-device [mac]**

Run inside `devenv shell` on the M1:

```bash
cargo test -p inferno-kernels --test rig
cargo test -p inferno-codegen --test differential
cargo test -p inferno-core --test artifact
```

Expected: all PASS with NEON dispatched. The differential now exercises the NEON kernels vs the scalar interpreter — the real correctness proof of the port.

- [ ] **Step 4: x86 unaffected [any]**

Run: `cargo test -p inferno-kernels --test rig && cargo test -p inferno-codegen --test differential` (x86)
Expected: PASS — the `Neon` arms are never selected on x86 (`available()` false).

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/src/registry.rs
git commit -m "kernels: dispatch NEON on aarch64 end-to-end (M5 Task 13)"
```

### Task 14: ARM tolerance re-derivation (evidenced)

**Files:**
- Modify (only if evidence requires): `crates/inferno-graph/src/tolerance.rs`
- Modify: `docs/superpowers/specs/2026-07-19-m5-apple-silicon-neon-baseline-design.md` (Amendments: the ARM error data)

**Interfaces:**
- Consumes: the on-device gates (Task 13) and the rig's `observed_error_*` sweeps.
- Produces: recorded ARM error distributions; a tolerance change **only** if the data demands it, with the derivation recorded.

- [ ] **Step 1: Run the observed-error sweeps on the M1 [mac]**

```bash
cargo test -p inferno-kernels --test rig observed_error_q8_0 -- --ignored --nocapture
cargo test -p inferno-kernels --test rig observed_error_q4_k -- --ignored --nocapture
cargo test -p inferno-kernels --test rig observed_error_attention -- --ignored --nocapture
```

Expected: prints `observed_error_*: max rel <e>` for each. Because NEON kernels are bitwise-identical to scalar (Tasks 8-12), the *kernel* rel-errors should match the x86-recorded values in `tolerance.rs` (Q8_0 ~2.4e-6, Q4_K ~9.2e-6, attn ~2.4e-7). Record the ARM numbers regardless.

- [ ] **Step 2: Assess the differential margin [mac]**

The remaining ARM-specific risk is LLVM autovectorization of the **elementwise** ops (`ops.rs`, mechanism 2) reordering reductions vs the interpreter. From Task 13's differential, capture the last-token `max_abs` and compare to `logits_abs_tol` (Q8_0/Q4_K band ~1e-2, f32 ~1e-4). If it passes with margin, record "x86 tolerances held on ARM with margin X" — **no change**. If it fails, the fix order is: (a) confirm no fast-math on the NEON build; (b) if a specific elementwise op autovectorized to a reordered reduction, pin that loop's reduction order in `ops.rs` (arch-neutral) rather than widening tolerance; (c) only if the divergence is a genuine, understood ARM FP property, re-derive the constant from the observed distribution and record the derivation.

- [ ] **Step 3: Record + commit [mac]**

Append the ARM error data and the tolerance verdict (changed or held) to the spec's `## Amendments` under `### Task 14 — ARM tolerance re-derivation`.

```bash
git add crates/inferno-graph/src/tolerance.rs docs/superpowers/specs/2026-07-19-m5-apple-silicon-neon-baseline-design.md
git commit -m "graph: record ARM error distributions; tolerance verdict (M5 Task 14)"
```

---

## Slice 3 — Measurement rig & baseline

Deliverable: a single-Mac quiet-hw runbook and one recorded baseline vs llama.cpp (best-of and NEON-only), gap attributed — the milestone's citable output and exit.

### Task 15: macOS quiet-hardware runbook

**Files:**
- Create: `docs/runbooks/quiet-hw-macos.md`
- Create: `scripts/quiet-hw/macos-preflight.sh` (a soft advisory check, not a hard gate)

**Interfaces:**
- Produces: the runbook the baseline (Task 17) follows; a preflight that reports throttle/quiescence.
- Consumes: nothing.

- [ ] **Step 1: Write the runbook [any]**

Create `docs/runbooks/quiet-hw-macos.md` as the darwin analog of `docs/runbooks/quiet-hw-verification.md`. Content (the Linux mechanisms it replaces are in parentheses):

- **Machine bar:** a Mac the author owns, on AC power, Low Power Mode **off** (`System Settings > Battery`), a cooled chassis preferred (Mac mini / 14" Pro over a fanless Air — the Air throttles under sustained pp512/tg128 and its reps must be discarded).
- **P-core steering** (replaces numactl socket-pinning): run the benchmark at `QOS_CLASS_USER_INTERACTIVE` so the scheduler places threads on the performance cluster; keep `--threads` ≤ detected `perf_cores` (Task 1) so E-cores stay out of the measured path. Record which cluster ran via `powermetrics`.
- **Thermal honesty** (replaces cgroup `nr_throttled`): before and during the run, sample `sudo powermetrics --samplers smc,cpu_power -i 1000 -n <secs>`; if CPU frequency drops or thermal pressure rises mid-run, **discard the affected reps** rather than averaging them in.
- **Quiescing checklist** (replaces PSI): Spotlight indexing paused (`sudo mdutil -a -i off` for the session, restore after), background apps quit, Wi-Fi/network idle, no Time Machine, Activity Monitor showing near-idle before start.
- **Recording:** every baseline number cites the exact M1 tier from Task 1 and states chassis + thread count. Data points are append-only in the spec's Amendments.

- [ ] **Step 2: Write the advisory preflight [any]**

Create `scripts/quiet-hw/macos-preflight.sh`: checks Low Power Mode off (`pmset -g | grep lowpowermode`), AC power (`pmset -g batt`), core count ≥ a floor, and prints a one-line `MACOS-PREFLIGHT: OK|WARN` with reasons. Advisory only (unlike the Linux hard gate) — a single owned Mac can't meet the multi-core dedicated-server bar and isn't meant to.

- [ ] **Step 3: Verify it runs [mac]**

Run: `bash scripts/quiet-hw/macos-preflight.sh`
Expected: prints `MACOS-PREFLIGHT: OK` (or `WARN` with the reason) on the M1.

- [ ] **Step 4: Commit**

```bash
git add docs/runbooks/quiet-hw-macos.md scripts/quiet-hw/macos-preflight.sh
git commit -m "runbook: single-Mac quiet-hw protocol (M5 Task 15)"
```

### Task 16: llama.cpp comparator builds + bench wiring on macOS

**Files:**
- Verify/modify: `cli/src/llama_bench.rs` (confirm `llama-bench -o json` parsing works with a macOS build; no change expected)
- Create: `scripts/build-llama-macos.sh` (builds both comparator variants)

**Interfaces:**
- Consumes: the existing `inferno bench` (`cli/src/bench.rs:368`) + `run_llama_bench` (`llama_bench.rs:42`).
- Produces: two `llama-bench` binaries — Accelerate/AMX and NEON-only — for the best-of + codegen-vs-codegen ratios.

- [ ] **Step 1: Build both llama.cpp variants [mac]**

Create `scripts/build-llama-macos.sh` that clones/pins the devenv-matched llama.cpp commit and builds twice:

```bash
# Accelerate/AMX build (llama at its genuine best on Apple Silicon)
cmake -B build-accel -DGGML_METAL=OFF -DGGML_ACCELERATE=ON  && cmake --build build-accel -j --target llama-bench
# NEON-only build (isolates codegen-vs-codegen; no AMX)
cmake -B build-neon  -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF && cmake --build build-neon -j --target llama-bench
```

Both `GGML_METAL=OFF` (CPU only; GPU is out of scope). Emit the two `llama-bench` paths.

- [ ] **Step 2: Smoke both against the model [mac]**

Fetch the model (`bash scripts/fetch-qwen-gguf.sh` — referenced in `docs/runbooks/metal.md`) and run each `llama-bench` once to confirm JSON output parses. Confirm inferno's `run_llama_bench` (`llama_bench.rs:55-60`) accepts the `-o json` rows from a macOS build (fields: `build_commit`, `cpu_info`, `avg_ts`, `stddev_ts`). Fix any parse mismatch in `llama_bench.rs` if the macOS `cpu_info` string shape breaks `find_row`.

- [ ] **Step 3: One end-to-end bench dry run [mac]**

Run: `mise run bench -- <qwen model> --llama-bench <build-accel path> --reps 1`
Expected: prints the `BenchReport` table ending in `ratio (inferno/llama.cpp): pp Nx | tg Nx`. This is a dry run (reps=1) to prove the pipeline, not the recorded baseline.

- [ ] **Step 4: Commit**

```bash
git add scripts/build-llama-macos.sh cli/src/llama_bench.rs
git commit -m "bench: macOS llama.cpp comparator builds (Accelerate + NEON-only) (M5 Task 16)"
```

### Task 17: run the baseline; record + attribute

**Files:**
- Modify: `docs/superpowers/specs/2026-07-19-m5-apple-silicon-neon-baseline-design.md` (Amendments: the recorded baseline)

**Interfaces:**
- Consumes: Tasks 13-16.
- Produces: the milestone's citable baseline and verdict.

- [ ] **Step 1: Quiesce per the runbook [mac]**

Follow `docs/runbooks/quiet-hw-macos.md`: Low Power off, AC, Spotlight paused, apps quit, `bash scripts/quiet-hw/macos-preflight.sh` → OK, `powermetrics` sampling in a side terminal.

- [ ] **Step 2: Run the protocol vs BOTH llama builds [mac]**

Model Qwen2.5-0.5B-Instruct Q8_0, pp512/tg128, full-thread (threads = detected `perf_cores`), release build, inside `devenv shell`:

```bash
mise run bench -- <qwen-q8_0.gguf> --llama-bench <build-accel path> --json   # best-of (Accelerate/AMX)
mise run bench -- <qwen-q8_0.gguf> --llama-bench <build-neon  path> --json   # codegen-vs-codegen (NEON-only)
```

Discard any rep flagged by `powermetrics` as throttled and re-run. Capture inferno pp/tg (mean ± stddev) once and both llama pp/tg.

- [ ] **Step 3: Attribute the gap [mac]**

From the two ratios, decompose per the spec's Exit Criterion 4:
- **Codegen quality** = inferno vs the **NEON-only** llama ratio (apples-to-apples, no AMX).
- **Hardware asymmetry** = the additional gap the **Accelerate/AMX** build opens over the NEON-only build (the AMX access inferno lacks).
- **NEON-kernel headroom** = any residual you can name (e.g. µbench roofline vs achieved), noted as the input to the next milestone.

- [ ] **Step 4: Record — append-only [mac]**

Append to the spec's `## Amendments` a `### 2026-07-19 — M5 baseline` section: the exact M1 tier, chassis, thread count; inferno pp512/tg128 ± stddev; both llama builds' pp/tg ± stddev and build commits; the two ratios; the three-way attribution; and Exit Criterion 5's verdict naming what the next milestone attacks (the "should inferno emit AMX/SME" input). No figure in this section is ever edited later.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-19-m5-apple-silicon-neon-baseline-design.md
git commit -m "M5 baseline: inferno vs llama.cpp on M1, gap attributed (M5 Task 17)"
```

### Task 18: closing verification + non-regression + PR

**Files:**
- Modify: `README.md` (status line: note Apple Silicon support + the M5 baseline, mirroring the v1-close honesty)
- Modify: `docs/superpowers/specs/2026-07-19-m5-apple-silicon-neon-baseline-design.md` (Amendments: closing verification walk)

**Interfaces:**
- Consumes: the whole milestone.
- Produces: the verification record and the PR.

- [ ] **Step 1: x86 non-regression in CI [any]**

Push the branch; confirm `ci.yml` (`mise run lint` + `mise run test` + `audit`, in `devenv shell`) is green on x86. This is Exit Criterion 2 — the port added paths, removed none.

- [ ] **Step 2: On-device gate recap [mac]**

Re-run the three gates + `mise run test` + `mise run lint` on the M1 one final time; confirm all green. Record the results.

- [ ] **Step 3: README refresh [any]**

Update `README.md`'s status/quickstart to state Apple Silicon (M1) support and cite the M5 baseline with the same "not a win, gap attributed" honesty the v1 close used — do **not** claim a win; M5 has no win gate. Cite the spec.

- [ ] **Step 4: Write the closing verification walk [any]**

Append to the spec's `## Amendments` a `### 2026-07-19 — M5 closing verification` section walking §Verification: on-device green (item 1), x86 CI green (item 2), tolerance data recorded (item 3, Task 14), baseline protocol-faithful and attributed (item 4, Task 17), no scope creep — no AMX/SME emitted, no Accelerate linked, no cloud Mac, no perf gate, metal budget zero (item 5).

- [ ] **Step 5: Commit + open PR**

```bash
git add README.md docs/superpowers/specs/2026-07-19-m5-apple-silicon-neon-baseline-design.md
git commit -m "M5 close: Apple Silicon NEON path + baseline recorded; verification walk (M5 Task 18)"
git push -u origin m5-apple-silicon-neon-baseline
gh pr create --fill --title "M5: Apple Silicon NEON bring-up & baseline (v2 milestone 1)"
```

---

## Self-Review

**Spec coverage** (each spec section → task):
- *Correct compiled path on M1* → Tasks 1-6 (scalar bring-up) + Tasks 7-13 (NEON) → gates green on-device (Tasks 6, 13).
- *Pure NEON, AMX/SME unused* → Tasks 8-13 emit only NEON; Task 12 Step 4 keeps `emitted_attn` off; Global Constraints forbid AMX/Accelerate.
- *Additive to x86* → every task has an x86 non-regression step (1.8, 2.6, 3.4, 4.5, 5.4, 7.4, 8/9/10/11/12 rig on x86, 13.4) + Task 18.1.
- *Codegen differential as primary gate, on-device* → Tasks 6.2, 13.3.
- *Scalar-vs-NEON bit-identity* → Tasks 7-12 (rig arms), enforced per kernel.
- *Tolerances re-derived, evidenced, never fudged* → Task 14.
- *Single-Mac quiet-hw runbook (P-core steering, thermal discards, no numactl)* → Task 15.
- *Baseline vs both llama builds, gap attributed* → Tasks 16-17.
- *Exit criteria 1-5* → 1: Tasks 6/13; 2: Task 18.1; 3: Task 14; 4: Task 17; 5: Task 17.4.
- *`.so`→`.dylib`, target-init, dynamic-lookup, sysctl detection* → Tasks 3, 4, 5, 1.
- No gap found.

**Placeholder scan:** kernel Tasks 9-12 give the AVX2/scalar reference to translate + the exact rig test as the spec rather than a full intrinsic listing — this is TDD (concrete failing test + worked f32 example in Task 8 + per-kernel recipe), not a "TODO." Task 8 is fully worked. No "TBD"/"implement later"/"add error handling" strings.

**Type consistency:** `Isa::Aarch64Neon`, `KernelIsa::Neon`, `ARTIFACT_LIB_FILENAME`, `initialize_native_target`, `detect_macos`, `simd_isa()`, `expf_neon`, and the `inferno_*_neon` symbol names are used identically across the tasks that define and consume them. `CoreTopology.perf_cores`/`eff_cores` are defined in Task 1 and consumed in Tasks 15/17. Consistent.
