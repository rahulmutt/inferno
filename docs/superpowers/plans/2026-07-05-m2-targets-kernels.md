# M2 — Targets + Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `inferno-target` (Linux hardware detection + named TOML profiles) and `inferno-kernels` (AVX2 GEMV microkernels for Q4_K/Q8_0/F32 behind a fixed C ABI), with a kernel-vs-scalar-oracle rig and criterion benches compared against the devenv-pinned llama.cpp.

**Architecture:** Two standalone crates, no runtime/CLI integration (spec: `docs/superpowers/specs/2026-07-05-m2-targets-kernels-design.md`). Kernels are GEMV-first over repacked strip-of-8 weight layouts; activations quantize on the fly (Q4_K×Q8_K, Q8_0×Q8_0 pairings, integer SIMD dots). Scalar and AVX2 variants share defined accumulation semantics and are **bit-identical** — integer dots are exact and every f32 op happens in the same order with the same fusing.

**Tech Stack:** Rust 1.96 (edition 2024), `core::arch::x86_64` intrinsics, serde+toml, rustix (page size), proptest, criterion, libloading (bench-only ggml FFI).

## Deviations from the spec, resolved during planning

Verified against the actual pinned llama.cpp and CI reality; Task 9 amends the spec to match:

1. **ggml comparison uses `dlopen`, not link-time FFI.** The nix `llama-cpp` package exports `ggml_vec_dot_q4_K_q8_K` / `ggml_vec_dot_q8_0_q8_0` / `quantize_row_q8_K` / `quantize_row_q8_0` / `ggml_vec_dot_f32` from its per-ISA CPU backends (`bin/libggml-cpu-<arch>.so`), not from `lib/libggml*.so`. The bench `libloading`-loads `libggml-cpu-haswell.so` (AVX2+FMA — same ISA class as our kernels) at a path from `$INFERNO_GGML_CPU_LIB`. No build script, and the spec's `ggml_mul_mat` fallback is unnecessary.
2. **detect==profile cannot run in nightly CI** — GitHub runners are not the dev Ryzen 3900. The test is env-gated (`INFERNO_EXPECT_PROFILE=ryzen-3900`) and passes vacuously when unset; nightly CI instead gains a `test-full` job with enlarged `PROPTEST_CASES`.
3. **`pack_*` is a plain safe Rust fn, not an `extern "C"` symbol.** Its only caller (M3 planner) is Rust, and a Vec-returning C ABI is a fiction. `quantize_row_*` and `gemv_*` stay `#[unsafe(no_mangle)] extern "C"`.
4. **Scalar-vs-AVX2 is bitwise-equal, not ~1e-6.** Both variants compute exact integer block dots and identical f32 combine sequences, so the rig asserts exact equality — stronger and simpler than a tolerance.
5. **`pack_q8_0_rs8` clamps weight bytes `-128 → -127`.** The AVX2 sign-trick (`_mm256_sign_epi8`) mis-multiplies `(-128)×(-128)`; ggml's own quantizer never emits −128 but a hostile GGUF can. Clamping once at pack time (max error: 1 part in 128 on one already-extreme value) keeps scalar/AVX2 bit-identical on untrusted input.

## Global Constraints

- **Edition 2024:** exported symbols need `#[unsafe(no_mangle)]` (not bare `#[no_mangle]`).
- **Workspace lints deny `unsafe_code`.** Only `inferno-kernels` overrides this with its own `[lints]` table; `inferno-target` and everything else inherit the workspace deny. `inferno-formats` stays `#![forbid(unsafe_code)]`.
- **After touching any `inferno-formats` code** (Task 6 makes one function `pub`): run `mise run fuzz -- gguf_parse` and `mise run fuzz -- safetensors_parse`.
- **Tolerances live only in `crates/inferno-graph/src/tolerance.rs`** and are tuned from observed data, never to make a test pass without understanding.
- **Workflows are mise tasks:** validate with `mise run test` and `mise run lint` before every commit; blocking tier stays ≤ 5 min wall-clock.
- **`ModelDesc`/`DType` stay format-agnostic:** activation formats (Q8_K etc.) live in `inferno-kernels`, never in `inferno-formats::DType`.
- **Commit after every task** (lefthook runs gitleaks pre-commit).
- **x86-64 Linux only.** Dev box and CI are both x86-64 Linux; the kernels
  crate and rig reference AVX2 symbols unconditionally and detection is
  Linux-only. Non-x86 builds are out of scope until the v2 NEON milestone.
- Dev machine ground truth (captured 2026-07-05, AMD Ryzen 9 3900, Zen 2): 12 physical / 24 logical cores, SMT, L1d 32 KiB (line 64, shared by 2), L2 512 KiB (shared by 2), L3 16 MiB per CCX slice (shared by 6), page size 4096, AVX2+FMA, **no AVX-512**.

## File Structure

```
Cargo.toml                                  # + members, workspace deps (toml, rustix, criterion, libloading)
crates/inferno-target/
├── Cargo.toml
├── profiles/ryzen-3900.toml                # captured named profile
├── examples/print_target.rs                # detect() → TOML on stdout
├── src/lib.rs                              # crate docs + re-exports
├── src/desc.rs                             # TargetDesc, Isa, Feature, CacheLevel, CoreTopology, BwClass
├── src/error.rs                            # TargetError
├── src/profile.rs                          # embedded profiles, from_profile()
├── src/detect.rs                           # detect(); pure sysfs parsers + thin live layer
└── tests/fixtures/sys-cpu-ryzen-3900/      # captured /sys/devices/system/cpu subset
crates/inferno-kernels/
├── Cargo.toml                              # own [lints]; feature ggml-compare; [[bench]]
├── src/lib.rs                              # crate docs, STRIP, KernelIsa
├── src/error.rs                            # KernelError
├── src/buf.rs                              # AlignedBuf (32-byte aligned)
├── src/act.rs                              # q8a/q8k activation formats + quantize_row (scalar/AVX2)
├── src/f32k.rs                             # F32 rs8 pack + gemv
├── src/q8_0.rs                             # Q8_0 rs8 pack + gemv
├── src/q4_k.rs                             # Q4_K rs8 pack + gemv
├── src/registry.rs                         # KernelSet safe wrappers, kernels_for()
├── benches/gemv.rs                         # criterion + ggml-compare FFI
└── tests/rig.rs                            # kernel-vs-oracle properties
crates/inferno-graph/src/tolerance.rs       # + gemv_rel_tol()
crates/inferno-formats/src/quant.rs         # get_scale_min_k4 → pub (Task 6)
devenv.nix                                  # + INFERNO_GGML_CPU_LIB
mise.toml                                   # + bench-kernels task
.github/workflows/nightly.yml               # + test-full job (PROPTEST_CASES=1024)
ARCHITECTURE.md, AGENTS.md, README.md       # M2 docs (Task 9)
docs/superpowers/specs/2026-07-05-m2-targets-kernels-design.md  # + amendments (Task 9)
```

---

### Task 1: `inferno-target` — TargetDesc types + named TOML profiles

**Files:**
- Modify: `Cargo.toml` (workspace members + deps)
- Create: `crates/inferno-target/Cargo.toml`
- Create: `crates/inferno-target/src/lib.rs`, `src/desc.rs`, `src/error.rs`, `src/profile.rs`
- Create: `crates/inferno-target/profiles/ryzen-3900.toml`

**Interfaces:**
- Consumes: nothing (leaf crate).
- Produces (used by Tasks 2, 7, 8):
  - `pub struct TargetDesc { pub isa: Isa, pub features: BTreeSet<Feature>, pub page_size: u64, pub memory_bw_class: Option<BwClass>, pub topology: CoreTopology, pub caches: Vec<CacheLevel> }`
  - `pub enum Isa { X86_64v3, X86_64v4 }` (serde names `"x86-64-v3"` / `"x86-64-v4"`)
  - `pub enum Feature { Vnni, Bf16 }`, `pub struct CacheLevel { pub level: u8, pub size_bytes: u64, pub line_bytes: u32, pub shared_by: u32 }`, `pub struct CoreTopology { pub physical_cores: u32, pub logical_cores: u32, pub smt: bool }`, `pub enum BwClass { Consumer, Workstation, Server }`
  - `TargetDesc::from_profile(name: &str) -> Result<TargetDesc>`, `pub fn available_profiles() -> Vec<&'static str>`
  - `pub enum TargetError` + `pub type Result<T>`

- [ ] **Step 1: Workspace wiring**

In root `Cargo.toml`, extend members and workspace deps (kernels entries land now so later tasks don't touch the root again):

```toml
members = [
    "crates/inferno-formats",
    "crates/inferno-graph",
    "crates/inferno-kernels",
    "crates/inferno-runtime",
    "crates/inferno-target",
    "cli",
]
```

and add to `[workspace.dependencies]`:

```toml
inferno-kernels = { path = "crates/inferno-kernels" }
inferno-target = { path = "crates/inferno-target" }
toml = "0.9"
rustix = { version = "1", default-features = false, features = ["param"] }
criterion = "0.7"
libloading = "0.9"
```

**Timing:** `crates/inferno-kernels` doesn't exist until Task 3, and a `members` entry for a missing crate breaks the workspace. In this task add only `inferno-target` to `members` and only `inferno-target`'s `[workspace.dependencies]` entry (plus `toml` and `rustix`); Task 3 adds the `inferno-kernels` member + dep entry along with `criterion` and `libloading`.

Create `crates/inferno-target/Cargo.toml`:

```toml
[package]
name = "inferno-target"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
serde.workspace = true
thiserror.workspace = true
toml.workspace = true
rustix.workspace = true

[lints]
workspace = true
```

- [ ] **Step 2: Write failing tests** (`#[cfg(test)]` at the bottom of `src/profile.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BwClass, CacheLevel, CoreTopology, Isa, TargetError};

    #[test]
    fn ryzen_profile_loads() {
        let t = TargetDesc::from_profile("ryzen-3900").unwrap();
        assert_eq!(t.isa, Isa::X86_64v3);
        assert!(t.features.is_empty());
        assert_eq!(t.page_size, 4096);
        assert_eq!(t.memory_bw_class, Some(BwClass::Consumer));
        assert_eq!(
            t.topology,
            CoreTopology { physical_cores: 12, logical_cores: 24, smt: true }
        );
        assert_eq!(
            t.caches,
            vec![
                CacheLevel { level: 1, size_bytes: 32 * 1024, line_bytes: 64, shared_by: 2 },
                CacheLevel { level: 2, size_bytes: 512 * 1024, line_bytes: 64, shared_by: 2 },
                CacheLevel { level: 3, size_bytes: 16 * 1024 * 1024, line_bytes: 64, shared_by: 6 },
            ]
        );
    }

    #[test]
    fn unknown_profile_lists_available() {
        let err = TargetDesc::from_profile("m3").unwrap_err();
        let TargetError::UnknownProfile { name, available } = err else {
            panic!("wrong variant: {err}");
        };
        assert_eq!(name, "m3");
        assert!(available.contains("ryzen-3900"));
    }

    #[test]
    fn toml_roundtrip_is_identity() {
        let t = TargetDesc::from_profile("ryzen-3900").unwrap();
        let text = toml::to_string_pretty(&t).unwrap();
        let back: TargetDesc = toml::from_str(&text).unwrap();
        assert_eq!(t, back);
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo nextest run -p inferno-target`
Expected: compile error (types missing).

- [ ] **Step 4: Implement**

`src/desc.rs`:

```rust
//! `TargetDesc`: plain serde-able data describing a machine. The same struct
//! whether probed live or loaded from a named profile — that equivalence is
//! the future cross-compile interface (spec §inferno-target). Always an
//! explicit input to planning/codegen; nothing downstream re-probes.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// ISA at kernel-dispatch granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Isa {
    /// AVX2 + FMA (+BMI1/2, F16C, LZCNT, MOVBE). All M2 kernels target this.
    #[serde(rename = "x86-64-v3")]
    X86_64v3,
    /// v3 + AVX-512 F/BW/CD/DQ/VL. Defined for dispatch; no M2 kernels.
    #[serde(rename = "x86-64-v4")]
    X86_64v4,
}

/// Features outside the ISA level that future kernels may dispatch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Feature {
    Vnni,
    Bf16,
}

/// One data/unified cache level as seen by a single core.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheLevel {
    pub level: u8,
    pub size_bytes: u64,
    pub line_bytes: u32,
    /// Logical CPUs sharing this cache instance.
    pub shared_by: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreTopology {
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub smt: bool,
}

/// Coarse memory-bandwidth class. Profile-only: nothing detects it; the M3
/// planner may consume it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BwClass {
    Consumer,
    Workstation,
    Server,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetDesc {
    pub isa: Isa,
    pub features: BTreeSet<Feature>,
    pub page_size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_bw_class: Option<BwClass>,
    pub topology: CoreTopology,
    pub caches: Vec<CacheLevel>,
}
```

`src/error.rs`:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TargetError {
    #[error("unknown target profile `{name}` (available: {available})")]
    UnknownProfile { name: String, available: String },
    #[error("malformed target profile `{name}`: {detail}")]
    MalformedProfile { name: String, detail: String },
    #[error("cannot detect hardware on this platform ({detail}); pass a named profile instead")]
    UnsupportedPlatform { detail: String },
    #[error("malformed sysfs data at {path}: {detail}")]
    MalformedSysfs { path: String, detail: String },
}

pub type Result<T> = std::result::Result<T, TargetError>;
```

`src/profile.rs`:

```rust
//! Named target profiles: TOML files embedded at build time. A profile and a
//! live detection produce the same `TargetDesc`.

use crate::{Result, TargetDesc, TargetError};

const PROFILES: &[(&str, &str)] =
    &[("ryzen-3900", include_str!("../profiles/ryzen-3900.toml"))];

pub fn available_profiles() -> Vec<&'static str> {
    PROFILES.iter().map(|(n, _)| *n).collect()
}

impl TargetDesc {
    pub fn from_profile(name: &str) -> Result<TargetDesc> {
        let Some((_, text)) = PROFILES.iter().find(|(n, _)| *n == name) else {
            return Err(TargetError::UnknownProfile {
                name: name.to_string(),
                available: available_profiles().join(", "),
            });
        };
        toml::from_str(text).map_err(|e| TargetError::MalformedProfile {
            name: name.to_string(),
            detail: e.to_string(),
        })
    }
}
```

`profiles/ryzen-3900.toml` (values captured from the dev machine — see Global Constraints):

```toml
# AMD Ryzen 9 3900 (Zen 2), captured 2026-07-05 from live detection.
# L3 is the per-CCX slice a single core sees, not the package total.
isa = "x86-64-v3"
features = []
page_size = 4096
memory_bw_class = "consumer"

[topology]
physical_cores = 12
logical_cores = 24
smt = true

[[caches]]
level = 1
size_bytes = 32768
line_bytes = 64
shared_by = 2

[[caches]]
level = 2
size_bytes = 524288
line_bytes = 64
shared_by = 2

[[caches]]
level = 3
size_bytes = 16777216
line_bytes = 64
shared_by = 6
```

`src/lib.rs`:

```rust
//! `TargetDesc` + hardware detection + named target profiles. Pure data and
//! probing: always an explicit input to planning and codegen (spec
//! §inferno-target). A detected target and a profile-loaded target are the
//! same struct — that equivalence is the cross-compilation interface.

mod desc;
mod error;
mod profile;

pub use desc::{BwClass, CacheLevel, CoreTopology, Feature, Isa, TargetDesc};
pub use error::{Result, TargetError};
pub use profile::available_profiles;
```

- [ ] **Step 5: Run tests**

Run: `cargo nextest run -p inferno-target`
Expected: 3 tests PASS. (Note: `memory_bw_class = Some(Consumer)` in the profile but detection later yields `None` — the detect==profile test in Task 2 compares with the profile's field cleared; see Task 2.)

- [ ] **Step 6: Lint and commit**

```bash
mise run lint && mise run test
git add -A && git commit -m "feat(target): TargetDesc types and named TOML profiles"
```

---

### Task 2: `inferno-target` — layered Linux detection + sysfs fixtures

**Files:**
- Create: `crates/inferno-target/src/detect.rs`
- Create: `crates/inferno-target/examples/print_target.rs`
- Create: `crates/inferno-target/tests/fixtures/sys-cpu-ryzen-3900/` (captured tree)
- Modify: `crates/inferno-target/src/lib.rs` (add `mod detect;`)

**Interfaces:**
- Consumes: Task 1 types.
- Produces (used by Task 7's feature checks and the M3 CLI):
  - `TargetDesc::detect() -> Result<TargetDesc>`
  - internal pure parsers `parse_topology(root: &Path) -> Result<CoreTopology>`, `parse_caches(root: &Path) -> Result<Vec<CacheLevel>>` (unit-tested against the fixture tree)

- [ ] **Step 1: Capture the sysfs fixture tree** (run on the dev machine; files are tiny text)

```bash
FIX=crates/inferno-target/tests/fixtures/sys-cpu-ryzen-3900
mkdir -p "$FIX"
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  n=$(basename "$c")
  mkdir -p "$FIX/$n/topology"
  cp "$c/topology/core_id" "$c/topology/physical_package_id" "$FIX/$n/topology/"
done
for d in /sys/devices/system/cpu/cpu0/cache/index[0-9]*; do
  i=$(basename "$d")
  mkdir -p "$FIX/cpu0/cache/$i"
  cp "$d/level" "$d/type" "$d/size" "$d/coherency_line_size" "$d/shared_cpu_list" "$FIX/cpu0/cache/$i/"
done
```

Sanity-check: `cat $FIX/cpu0/cache/index3/size` → `16384K`; `ls $FIX | wc -l` → 24.

- [ ] **Step 2: Write failing tests** (`#[cfg(test)]` at the bottom of the new `src/detect.rs`; create the file with just the test module first)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheLevel, CoreTopology, TargetDesc};
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sys-cpu-ryzen-3900")
    }

    #[test]
    fn fixture_topology() {
        assert_eq!(
            parse_topology(&fixture()).unwrap(),
            CoreTopology { physical_cores: 12, logical_cores: 24, smt: true }
        );
    }

    #[test]
    fn fixture_caches() {
        // index1 (32K Instruction) must be skipped; Data/Unified kept, sorted by level.
        assert_eq!(
            parse_caches(&fixture()).unwrap(),
            vec![
                CacheLevel { level: 1, size_bytes: 32768, line_bytes: 64, shared_by: 2 },
                CacheLevel { level: 2, size_bytes: 524288, line_bytes: 64, shared_by: 2 },
                CacheLevel { level: 3, size_bytes: 16777216, line_bytes: 64, shared_by: 6 },
            ]
        );
    }

    #[test]
    fn missing_root_is_typed_error() {
        let err = parse_topology(std::path::Path::new("/nonexistent-sys")).unwrap_err();
        assert!(matches!(err, crate::TargetError::MalformedSysfs { .. }), "{err}");
    }

    #[test]
    fn cpu_list_and_size_parsers() {
        assert_eq!(parse_cpu_list("0,12", "t").unwrap(), 2);
        assert_eq!(parse_cpu_list("0-2,12-14", "t").unwrap(), 6);
        assert_eq!(parse_cpu_list("7", "t").unwrap(), 1);
        assert_eq!(parse_size("32K", "t").unwrap(), 32768);
        assert_eq!(parse_size("16384K", "t").unwrap(), 16777216);
        assert_eq!(parse_size("4M", "t").unwrap(), 4 * 1024 * 1024);
        assert_eq!(parse_size("512", "t").unwrap(), 512);
        assert!(parse_size("32Q", "t").is_err());
    }

    /// Live detection must succeed and be internally coherent wherever the
    /// suite runs (CI runners and the dev box are both x86-64-v3+ Linux).
    #[test]
    fn live_detect_is_coherent() {
        let t = TargetDesc::detect().unwrap();
        assert!(t.topology.logical_cores >= t.topology.physical_cores);
        assert!(t.topology.physical_cores >= 1);
        assert!(!t.caches.is_empty());
        assert!(t.page_size >= 4096);
        assert!(t.memory_bw_class.is_none());
    }

    /// detect == profile equivalence (spec §inferno-target). Machine-specific:
    /// gated on INFERNO_EXPECT_PROFILE, set on the dev box; vacuous elsewhere.
    /// memory_bw_class is profile-only, so it is cleared before comparing.
    #[test]
    fn detect_matches_expected_profile() {
        let Ok(name) = std::env::var("INFERNO_EXPECT_PROFILE") else { return };
        let mut profile = TargetDesc::from_profile(&name).unwrap();
        profile.memory_bw_class = None;
        assert_eq!(TargetDesc::detect().unwrap(), profile);
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo nextest run -p inferno-target detect`
Expected: compile error (`parse_topology` etc. missing).

- [ ] **Step 4: Implement** (top of `src/detect.rs`, above the tests)

```rust
//! Live hardware detection, layered for testability (spec §inferno-target):
//! pure functions parse a captured `/sys/devices/system/cpu` tree passed as a
//! root path; a thin live layer supplies the real root, `is_x86_feature_detected!`
//! for the ISA, and the page size. No downstream crate re-probes hardware.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use crate::{CacheLevel, CoreTopology, Feature, Isa, Result, TargetDesc, TargetError};

impl TargetDesc {
    pub fn detect() -> Result<TargetDesc> {
        let (isa, features) = detect_isa()?;
        let root = Path::new("/sys/devices/system/cpu");
        Ok(TargetDesc {
            isa,
            features,
            page_size: rustix::param::page_size() as u64,
            memory_bw_class: None,
            topology: parse_topology(root)?,
            caches: parse_caches(root)?,
        })
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_isa() -> Result<(Isa, BTreeSet<Feature>)> {
    macro_rules! det {
        ($f:literal) => {
            std::arch::is_x86_feature_detected!($f)
        };
    }
    let v3 = det!("avx2")
        && det!("fma")
        && det!("bmi1")
        && det!("bmi2")
        && det!("f16c")
        && det!("lzcnt")
        && det!("movbe");
    if !v3 {
        return Err(TargetError::UnsupportedPlatform {
            detail: "CPU below x86-64-v3 (needs AVX2+FMA)".to_string(),
        });
    }
    let v4 = det!("avx512f")
        && det!("avx512bw")
        && det!("avx512cd")
        && det!("avx512dq")
        && det!("avx512vl");
    let mut features = BTreeSet::new();
    if det!("avx512vnni") {
        features.insert(Feature::Vnni);
    }
    if det!("avx512bf16") {
        features.insert(Feature::Bf16);
    }
    Ok((if v4 { Isa::X86_64v4 } else { Isa::X86_64v3 }, features))
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_isa() -> Result<(Isa, BTreeSet<Feature>)> {
    Err(TargetError::UnsupportedPlatform {
        detail: "only x86-64 detection is implemented (M2)".to_string(),
    })
}

fn bad(path: &Path, detail: impl Into<String>) -> TargetError {
    TargetError::MalformedSysfs { path: path.display().to_string(), detail: detail.into() }
}

fn read_trim(path: &Path) -> Result<String> {
    fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .map_err(|e| bad(path, e.to_string()))
}

fn parse_topology(root: &Path) -> Result<CoreTopology> {
    let entries = fs::read_dir(root).map_err(|e| bad(root, e.to_string()))?;
    let mut logical = 0u32;
    let mut cores = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|e| bad(root, e.to_string()))?;
        let name = entry.file_name();
        let Some(idx) = name.to_string_lossy().strip_prefix("cpu").map(str::to_string) else {
            continue;
        };
        if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let topo = entry.path().join("topology");
        let pkg = read_trim(&topo.join("physical_package_id"))?;
        let core = read_trim(&topo.join("core_id"))?;
        cores.insert((pkg, core));
        logical += 1;
    }
    if logical == 0 {
        return Err(bad(root, "no cpuN directories"));
    }
    let physical = cores.len() as u32;
    Ok(CoreTopology { physical_cores: physical, logical_cores: logical, smt: logical > physical })
}

fn parse_caches(root: &Path) -> Result<Vec<CacheLevel>> {
    let dir = root.join("cpu0/cache");
    let entries = fs::read_dir(&dir).map_err(|e| bad(&dir, e.to_string()))?;
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| bad(&dir, e.to_string()))?;
        if !entry.file_name().to_string_lossy().starts_with("index") {
            continue;
        }
        let p = entry.path();
        let ty = read_trim(&p.join("type"))?;
        if ty != "Data" && ty != "Unified" {
            continue;
        }
        let level = read_trim(&p.join("level"))?
            .parse::<u8>()
            .map_err(|e| bad(&p, format!("level: {e}")))?;
        let size_bytes = parse_size(&read_trim(&p.join("size"))?, &p.display().to_string())?;
        let line_bytes = read_trim(&p.join("coherency_line_size"))?
            .parse::<u32>()
            .map_err(|e| bad(&p, format!("coherency_line_size: {e}")))?;
        let shared_by =
            parse_cpu_list(&read_trim(&p.join("shared_cpu_list"))?, &p.display().to_string())?;
        out.push(CacheLevel { level, size_bytes, line_bytes, shared_by });
    }
    if out.is_empty() {
        return Err(bad(&dir, "no Data/Unified cache index directories"));
    }
    out.sort_by_key(|c| c.level);
    Ok(out)
}

/// "32K" | "4M" | "512" → bytes.
fn parse_size(s: &str, ctx: &str) -> Result<u64> {
    let err = || TargetError::MalformedSysfs {
        path: ctx.to_string(),
        detail: format!("unparseable cache size `{s}`"),
    };
    if let Some(n) = s.strip_suffix('K') {
        return n.parse::<u64>().map(|n| n * 1024).map_err(|_| err());
    }
    if let Some(n) = s.strip_suffix('M') {
        return n.parse::<u64>().map(|n| n * 1024 * 1024).map_err(|_| err());
    }
    s.parse::<u64>().map_err(|_| err())
}

/// "0-2,12-14" → count of listed CPUs.
fn parse_cpu_list(s: &str, ctx: &str) -> Result<u32> {
    let err = |d: String| TargetError::MalformedSysfs { path: ctx.to_string(), detail: d };
    let mut count = 0u32;
    for part in s.split(',') {
        match part.split_once('-') {
            Some((a, b)) => {
                let a: u32 = a.parse().map_err(|_| err(format!("bad range `{part}`")))?;
                let b: u32 = b.parse().map_err(|_| err(format!("bad range `{part}`")))?;
                if b < a {
                    return Err(err(format!("inverted range `{part}`")));
                }
                count += b - a + 1;
            }
            None => {
                part.parse::<u32>().map_err(|_| err(format!("bad cpu id `{part}`")))?;
                count += 1;
            }
        }
    }
    Ok(count)
}
```

Add `mod detect;` to `src/lib.rs` (no new re-exports — `detect()` is an inherent method).

Create `examples/print_target.rs` (handy for capturing future profiles):

```rust
//! Print the detected TargetDesc as profile-ready TOML.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    print!("{}", toml::to_string_pretty(&inferno_target::TargetDesc::detect()?)?);
    Ok(())
}
```

- [ ] **Step 5: Run tests, including the machine-gated one**

```bash
cargo nextest run -p inferno-target
INFERNO_EXPECT_PROFILE=ryzen-3900 cargo nextest run -p inferno-target detect_matches
cargo run -p inferno-target --example print_target
```

Expected: all PASS; the example prints TOML matching `profiles/ryzen-3900.toml` minus `memory_bw_class`. If `detect_matches` fails, the *profile* is wrong — fix the TOML from the example output, never fudge the parsers.

- [ ] **Step 6: Lint and commit**

```bash
mise run lint && mise run test
git add -A && git commit -m "feat(target): layered Linux hardware detection with sysfs fixtures"
```

---

### Task 3: `inferno-kernels` skeleton — aligned buffers + activation quantization (q8a/q8k)

**Files:**
- Modify: `Cargo.toml` (add `crates/inferno-kernels` to members; add `inferno-kernels` to `[workspace.dependencies]` if not done in Task 1)
- Create: `crates/inferno-kernels/Cargo.toml`
- Create: `crates/inferno-kernels/src/lib.rs`, `src/error.rs`, `src/buf.rs`, `src/act.rs`

**Interfaces:**
- Consumes: `inferno_graph::tolerance::roundtrip_rel_tol` (dev-dep, for the round-trip bound).
- Produces (used by Tasks 4–8):
  - `pub const STRIP: usize = 8`
  - `pub enum KernelIsa { Scalar, Avx2 }` with `KernelIsa::available(self) -> bool` and `KernelIsa::all_available() -> Vec<KernelIsa>`
  - `pub struct AlignedBuf` — `zeroed(len)`, `len()`, `is_empty()`, `as_slice()`, `as_mut_slice()`, `as_ptr()`; 32-byte aligned by construction
  - `pub enum KernelError` + `pub type Result<T>`
  - `act`: `Q8A_BLOCK=32`, `Q8A_BLOCK_BYTES=36`, `Q8K_BLOCK=256`, `Q8K_BLOCK_BYTES=292`, `q8a_len(k)`, `q8k_len(k)`, safe `quantize_row_q8a(isa, x: &[f32]) -> Result<Vec<u8>>`, `quantize_row_q8k(isa, x: &[f32]) -> Result<Vec<u8>>`, and the four C symbols `inferno_quantize_row_{q8a,q8k}_{scalar,avx2}(x: *const f32, y: *mut u8, k: usize)`

Layouts (little-endian throughout):
- **q8a** (pairs with Q8_0 weights), per 32 elements: `[d: f32][qs: 32 × i8]` = 36 bytes.
- **q8k** (pairs with Q4_K weights), per 256 elements: `[d: f32][qs: 256 × i8][bsums: 8 × i32]` = 292 bytes; `bsums[j] = Σ qs[j*32..(j+1)*32]` feeds the Q4_K dmin correction.

- [ ] **Step 1: Crate scaffolding**

`crates/inferno-kernels/Cargo.toml`:

```toml
[package]
name = "inferno-kernels"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
inferno-formats.workspace = true
inferno-target.workspace = true
thiserror.workspace = true
libloading = { workspace = true, optional = true }

[dev-dependencies]
inferno-graph.workspace = true
proptest.workspace = true
criterion.workspace = true

[features]
# Bench-only FFI against the devenv-pinned ggml CPU backend. Never default:
# shipping builds contain zero FFI (spec §Benchmarks).
ggml-compare = ["dep:libloading"]

# The workspace lints deny `unsafe_code`; this crate is the sanctioned
# exception (intrinsics + the C ABI M3 codegen calls by symbol). Every unsafe
# fn documents its contract; unsafe ops inside unsafe fns still need blocks.
[lints.rust]
unsafe_op_in_unsafe_fn = "deny"

[[bench]]
name = "gemv"
harness = false
```

Also create a placeholder `benches/gemv.rs` so `--all-targets` compiles (real benches land in Task 8):

```rust
//! Kernel µbenches land in Task 8 (spec §Benchmarks).
fn main() {}
```

`src/lib.rs`:

```rust
//! Hand-tuned CPU microkernels behind a fixed C ABI (spec §inferno-kernels).
//! One `#[unsafe(no_mangle)] extern "C"` symbol per (op × dtype × ISA);
//! M3-generated code calls these by name. Weight packing is safe Rust.
//!
//! Numeric contract: integer block dots are exact and every f32 operation
//! happens in the same order with the same fusing in every ISA variant, so
//! variants are **bit-identical** — the rig asserts exact equality.
//!
//! Activation-side quantization formats (q8a/q8k) live here and never in
//! `inferno_formats::DType`: they are a kernel implementation detail, not a
//! weight-file dtype (spec boundary rule).

pub mod act;
mod buf;
mod error;

pub use buf::AlignedBuf;
pub use error::{KernelError, Result};

/// Rows per packed strip: every rs8 layout interleaves 8 rows.
pub const STRIP: usize = 8;

/// Which implementation of a kernel to run. Scalar is always available; SIMD
/// variants only where the CPU supports them (the registry enforces this —
/// the single place runtime feature detection happens).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelIsa {
    Scalar,
    Avx2,
}

impl KernelIsa {
    pub fn available(self) -> bool {
        match self {
            KernelIsa::Scalar => true,
            KernelIsa::Avx2 => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
        }
    }

    /// All variants runnable on this CPU (rig helpers iterate this).
    pub fn all_available() -> Vec<KernelIsa> {
        [KernelIsa::Scalar, KernelIsa::Avx2].into_iter().filter(|i| i.available()).collect()
    }
}
```

`src/error.rs`:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("{what}: got {got} bytes, expected {expected}")]
    SizeMismatch { what: &'static str, got: usize, expected: usize },
    #[error("k={k} is not a positive multiple of the {block}-element block")]
    BadK { k: usize, block: usize },
    #[error("rows must be non-zero")]
    ZeroRows,
    #[error("row range {row_start}..{row_end} invalid for {rows} rows")]
    BadRowRange { row_start: usize, row_end: usize, rows: usize },
    #[error("size overflow computing a buffer length")]
    Overflow,
}

pub type Result<T> = std::result::Result<T, KernelError>;
```

`src/buf.rs`:

```rust
//! 32-byte-aligned byte buffers: packed weight images satisfy the kernels'
//! aligned-load contract by construction, so callers can't get it wrong.

/// One aligned lane; `Vec<Lane>` keeps the whole allocation 32-byte aligned.
#[derive(Clone)]
#[repr(C, align(32))]
struct Lane([u8; 32]);

#[derive(Clone)]
pub struct AlignedBuf {
    lanes: Vec<Lane>,
    len: usize,
}

impl AlignedBuf {
    pub fn zeroed(len: usize) -> Self {
        AlignedBuf { lanes: vec![Lane([0u8; 32]); len.div_ceil(32)], len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.lanes.as_ptr().cast()
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `lanes` is one contiguous allocation of lanes.len()*32 >= len
        // initialized bytes, and the cast pointer lives as long as &self.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as as_slice, and &mut self guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.lanes.as_mut_ptr().cast(), self.len) }
    }
}

impl std::fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuf").field("len", &self.len).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_and_sized() {
        for len in [0usize, 1, 31, 32, 33, 4096] {
            let mut b = AlignedBuf::zeroed(len);
            assert_eq!(b.as_ptr() as usize % 32, 0);
            assert_eq!(b.as_slice().len(), len);
            assert_eq!(b.as_mut_slice().len(), len);
            assert!(b.as_slice().iter().all(|&x| x == 0));
        }
    }
}
```

- [ ] **Step 2: Write failing tests for `act`** (`#[cfg(test)]` at the bottom of `src/act.rs`; create with the module doc + constants + test module first)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelIsa;
    use inferno_formats::DType;
    use inferno_graph::tolerance::roundtrip_rel_tol;
    use proptest::prelude::*;

    /// Deterministic pseudo-random f32s in [-1, 1) — cheaper than proptest
    /// vec strategies for large inputs, still seed-driven.
    pub(crate) fn pseudo(mut seed: u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn decode_q8a(buf: &[u8], k: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(k);
        for b in buf.chunks_exact(Q8A_BLOCK_BYTES) {
            let d = f32::from_le_bytes(b[..4].try_into().unwrap());
            out.extend(b[4..].iter().map(|&q| d * f32::from(q as i8)));
        }
        out
    }

    fn decode_q8k(buf: &[u8], k: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(k);
        for b in buf.chunks_exact(Q8K_BLOCK_BYTES) {
            let d = f32::from_le_bytes(b[..4].try_into().unwrap());
            out.extend(b[4..260].iter().map(|&q| d * f32::from(q as i8)));
        }
        out
    }

    fn check_roundtrip(vals: &[f32], decoded: &[f32]) {
        // 8-bit block quant: same error class as Q8_0 weights.
        let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
        let tol = roundtrip_rel_tol(&DType::Q8_0) * amax;
        for (i, (a, b)) in vals.iter().zip(decoded).enumerate() {
            assert!((a - b).abs() <= tol, "[{i}]: {a} vs {b} (tol {tol})");
        }
    }

    proptest! {
        #[test]
        fn q8a_roundtrip_all_isas(seed in any::<u64>(), blocks in 1usize..5) {
            let k = blocks * Q8A_BLOCK;
            let x = pseudo(seed, k);
            for isa in KernelIsa::all_available() {
                let buf = quantize_row_q8a(isa, &x).unwrap();
                prop_assert_eq!(buf.len(), q8a_len(k));
                check_roundtrip(&x, &decode_q8a(&buf, k));
            }
        }

        #[test]
        fn q8k_roundtrip_and_exact_bsums(seed in any::<u64>(), blocks in 1usize..3) {
            let k = blocks * Q8K_BLOCK;
            let x = pseudo(seed, k);
            for isa in KernelIsa::all_available() {
                let buf = quantize_row_q8k(isa, &x).unwrap();
                prop_assert_eq!(buf.len(), q8k_len(k));
                check_roundtrip(&x, &decode_q8k(&buf, k));
                for b in buf.chunks_exact(Q8K_BLOCK_BYTES) {
                    for j in 0..8 {
                        let want: i32 =
                            b[4 + j * 32..4 + (j + 1) * 32].iter().map(|&q| i32::from(q as i8)).sum();
                        let got =
                            i32::from_le_bytes(b[260 + j * 4..264 + j * 4].try_into().unwrap());
                        prop_assert_eq!(got, want, "bsum {}", j);
                    }
                }
            }
        }

        /// ISA variants must produce byte-identical activation buffers.
        #[test]
        fn quantize_isa_variants_bitwise_equal(seed in any::<u64>()) {
            if !KernelIsa::Avx2.available() { return Ok(()); }
            let x = pseudo(seed, 2 * Q8K_BLOCK);
            prop_assert_eq!(
                quantize_row_q8a(KernelIsa::Scalar, &x).unwrap(),
                quantize_row_q8a(KernelIsa::Avx2, &x).unwrap()
            );
            prop_assert_eq!(
                quantize_row_q8k(KernelIsa::Scalar, &x).unwrap(),
                quantize_row_q8k(KernelIsa::Avx2, &x).unwrap()
            );
        }
    }

    #[test]
    fn zero_block_has_zero_scale() {
        let buf = quantize_row_q8a(KernelIsa::Scalar, &[0f32; 32]).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn bad_k_rejected() {
        assert!(quantize_row_q8a(KernelIsa::Scalar, &[0f32; 31]).is_err());
        assert!(quantize_row_q8k(KernelIsa::Scalar, &[0f32; 255]).is_err());
        assert!(quantize_row_q8a(KernelIsa::Scalar, &[]).is_err());
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo nextest run -p inferno-kernels`
Expected: compile error (functions missing).

- [ ] **Step 4: Implement `src/act.rs`**

```rust
//! Activation-side quantization: f32 rows → integer blocks at the kernel
//! boundary, mirroring ggml's pairings (Q4_K×Q8_K, Q8_0×Q8_0). Kernel-internal
//! formats — never `inferno_formats::DType` (spec boundary rule).
//!
//! q8a, per 32 elems:  [d: f32 le][qs: 32 × i8]                      = 36 B
//! q8k, per 256 elems: [d: f32 le][qs: 256 × i8][bsums: 8 × i32 le]  = 292 B
//!
//! Rounding is ties-to-even in every variant (`round_ties_even` scalar,
//! `cvtps` under the default MXCSR mode in AVX2) so variants stay bitwise
//! equal; quantized values are clamped to [-127, 127].

use crate::{KernelError, KernelIsa, Result};

pub const Q8A_BLOCK: usize = 32;
pub const Q8A_BLOCK_BYTES: usize = 36;
pub const Q8K_BLOCK: usize = 256;
pub const Q8K_BLOCK_BYTES: usize = 292;

pub fn q8a_len(k: usize) -> usize {
    k / Q8A_BLOCK * Q8A_BLOCK_BYTES
}

pub fn q8k_len(k: usize) -> usize {
    k / Q8K_BLOCK * Q8K_BLOCK_BYTES
}

/// Scalar semantic core: quantize one block, returning its scale.
fn quantize_block(x: &[f32], qs: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
    for (q, v) in qs.iter_mut().zip(x) {
        *q = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
    }
    d
}

/// # Safety
/// - `x` valid for `k` f32 reads; `y` valid for `q8a_len(k)` byte writes.
/// - `k` is a multiple of 32. All inputs finite (NaN/Inf are precondition
///   violations — kernels do not check; the hot path stays branch-free).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_quantize_row_q8a_scalar(x: *const f32, y: *mut u8, k: usize) {
    let x = unsafe { std::slice::from_raw_parts(x, k) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, q8a_len(k)) };
    for (xb, yb) in x.chunks_exact(Q8A_BLOCK).zip(y.chunks_exact_mut(Q8A_BLOCK_BYTES)) {
        let mut qs = [0i8; Q8A_BLOCK];
        let d = quantize_block(xb, &mut qs);
        yb[..4].copy_from_slice(&d.to_le_bytes());
        for (dst, q) in yb[4..].iter_mut().zip(qs) {
            *dst = q as u8;
        }
    }
}

/// # Safety
/// As [`inferno_quantize_row_q8a_scalar`], with `k` a multiple of 256 and `y`
/// valid for `q8k_len(k)` byte writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_quantize_row_q8k_scalar(x: *const f32, y: *mut u8, k: usize) {
    let x = unsafe { std::slice::from_raw_parts(x, k) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, q8k_len(k)) };
    for (xb, yb) in x.chunks_exact(Q8K_BLOCK).zip(y.chunks_exact_mut(Q8K_BLOCK_BYTES)) {
        let mut qs = [0i8; Q8K_BLOCK];
        let d = quantize_block(xb, &mut qs);
        yb[..4].copy_from_slice(&d.to_le_bytes());
        for (dst, q) in yb[4..260].iter_mut().zip(qs) {
            *dst = q as u8;
        }
        for j in 0..8 {
            let s: i32 = qs[j * 32..(j + 1) * 32].iter().map(|&q| i32::from(q)).sum();
            yb[260 + j * 4..264 + j * 4].copy_from_slice(&s.to_le_bytes());
        }
    }
}

/// AVX2 core: quantize one 32-f32 chunk against a precomputed `inv`, writing
/// 32 i8 to `dst` and returning the four pre-narrowing i32 vectors' sum (the
/// caller uses it for q8k bsums). Must match `quantize_block` bitwise:
/// same mul, ties-to-even rounding, clamp to [-127, 127].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quant32_avx2(x: *const f32, inv: f32, dst: *mut u8) -> i32 {
    use std::arch::x86_64::*;
    let vinv = _mm256_set1_ps(inv);
    let lo127 = _mm256_set1_epi32(-127);
    let hi127 = _mm256_set1_epi32(127);
    let mut q = [_mm256_setzero_si256(); 4];
    let mut bsum = _mm256_setzero_si256();
    for (i, qi) in q.iter_mut().enumerate() {
        let v = unsafe { _mm256_loadu_ps(x.add(i * 8)) };
        let r = _mm256_cvtps_epi32(_mm256_mul_ps(v, vinv)); // ties-to-even
        let c = _mm256_max_epi32(lo127, _mm256_min_epi32(hi127, r));
        bsum = _mm256_add_epi32(bsum, c);
        *qi = c;
    }
    let p0 = _mm256_packs_epi32(q[0], q[1]);
    let p1 = _mm256_packs_epi32(q[2], q[3]);
    let packed = _mm256_packs_epi16(p0, p1);
    // packs interleaves 128-bit lanes; restore element order.
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let packed = _mm256_permutevar8x32_epi32(packed, order);
    unsafe { _mm256_storeu_si256(dst.cast(), packed) };
    hsum_i32(bsum)
}

/// Horizontal sum of 8 × i32. Exact (integer), so reduction order is free.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn hsum_i32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_hadd_epi32(s, s);
    let s = _mm_hadd_epi32(s, s);
    _mm_cvtsi128_si32(s)
}

/// amax of one 32-f32 chunk. max is exact and order-free on finite input.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn amax32_avx2(x: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let sign = _mm256_set1_ps(-0.0);
    let mut m = _mm256_setzero_ps();
    for i in 0..4 {
        let v = unsafe { _mm256_loadu_ps(x.add(i * 8)) };
        m = _mm256_max_ps(m, _mm256_andnot_ps(sign, v));
    }
    let s = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps::<1>(m));
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_max_ss(s, _mm_shuffle_ps::<1>(s, s));
    _mm_cvtss_f32(s)
}

/// # Safety
/// As [`inferno_quantize_row_q8a_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_quantize_row_q8a_avx2(x: *const f32, y: *mut u8, k: usize) {
    for b in 0..k / Q8A_BLOCK {
        let xb = unsafe { x.add(b * Q8A_BLOCK) };
        let yb = unsafe { y.add(b * Q8A_BLOCK_BYTES) };
        let amax = unsafe { amax32_avx2(xb) };
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        unsafe { yb.cast::<[u8; 4]>().write_unaligned(d.to_le_bytes()) };
        unsafe { quant32_avx2(xb, inv, yb.add(4)) };
    }
}

/// # Safety
/// As [`inferno_quantize_row_q8k_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_quantize_row_q8k_avx2(x: *const f32, y: *mut u8, k: usize) {
    for b in 0..k / Q8K_BLOCK {
        let xb = unsafe { x.add(b * Q8K_BLOCK) };
        let yb = unsafe { y.add(b * Q8K_BLOCK_BYTES) };
        let mut amax = 0f32;
        for c in 0..8 {
            amax = amax.max(unsafe { amax32_avx2(xb.add(c * 32)) });
        }
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        unsafe { yb.cast::<[u8; 4]>().write_unaligned(d.to_le_bytes()) };
        for j in 0..8 {
            let bsum = unsafe { quant32_avx2(xb.add(j * 32), inv, yb.add(4 + j * 32)) };
            unsafe {
                yb.add(260 + j * 4).cast::<[u8; 4]>().write_unaligned(bsum.to_le_bytes())
            };
        }
    }
}

fn validate(k: usize, block: usize) -> Result<()> {
    if k == 0 || k % block != 0 {
        return Err(KernelError::BadK { k, block });
    }
    Ok(())
}

/// Safe wrapper (tests, benches, M3 planner). The raw symbols stay unchecked.
pub fn quantize_row_q8a(isa: KernelIsa, x: &[f32]) -> Result<Vec<u8>> {
    validate(x.len(), Q8A_BLOCK)?;
    let mut out = vec![0u8; q8a_len(x.len())];
    match isa {
        // SAFETY: x/out lengths validated against the symbol's contract.
        KernelIsa::Scalar => unsafe {
            inferno_quantize_row_q8a_scalar(x.as_ptr(), out.as_mut_ptr(), x.len())
        },
        // SAFETY: as above; KernelIsa::Avx2 callers hold the feature invariant
        // (the registry refuses to hand out AVX2 kernels without CPU support).
        KernelIsa::Avx2 => unsafe {
            inferno_quantize_row_q8a_avx2(x.as_ptr(), out.as_mut_ptr(), x.len())
        },
    }
    Ok(out)
}

pub fn quantize_row_q8k(isa: KernelIsa, x: &[f32]) -> Result<Vec<u8>> {
    validate(x.len(), Q8K_BLOCK)?;
    let mut out = vec![0u8; q8k_len(x.len())];
    match isa {
        // SAFETY: as quantize_row_q8a.
        KernelIsa::Scalar => unsafe {
            inferno_quantize_row_q8k_scalar(x.as_ptr(), out.as_mut_ptr(), x.len())
        },
        // SAFETY: as quantize_row_q8a.
        KernelIsa::Avx2 => unsafe {
            inferno_quantize_row_q8k_avx2(x.as_ptr(), out.as_mut_ptr(), x.len())
        },
    }
    Ok(out)
}
```

- [ ] **Step 5: Run tests**

Run: `cargo nextest run -p inferno-kernels`
Expected: all PASS, including bitwise ISA equality (this machine has AVX2).

- [ ] **Step 6: Lint and commit**

```bash
mise run lint && mise run test
git add -A && git commit -m "feat(kernels): crate skeleton, aligned buffers, q8a/q8k activation quantization"
```

---

### Task 4: F32 rs8 kernels + the oracle rig baseline

**Files:**
- Create: `crates/inferno-kernels/src/f32k.rs` (add `pub mod f32k;` to lib.rs)
- Create: `crates/inferno-kernels/tests/rig.rs`
- Modify: `crates/inferno-graph/src/tolerance.rs` (add `gemv_rel_tol`)

**Interfaces:**
- Consumes: `AlignedBuf`, `STRIP`, `KernelIsa` (Task 3); `inferno_formats::quant::{pack, dequant}`; `inferno_graph::ops::matmul`, `inferno_graph::Tensor`.
- Produces (used by Tasks 7, 8):
  - `f32k::packed_len_f32_rs8(rows, k) -> usize`
  - `f32k::pack_f32_rs8(bytes: &[u8], rows: usize, k: usize) -> Result<AlignedBuf>`
  - C symbols `inferno_gemv_f32_rs8_{scalar,avx2}(y: *mut f32, x: *const u8, w: *const u8, k: usize, row_start: usize, row_end: usize)` — **unified ABI**: `x` is always `*const u8` (for F32 it points at raw f32 LE bytes; for quant kernels at a q8a/q8k buffer)
  - `inferno_graph::tolerance::gemv_rel_tol(dtype: &DType) -> f32`
  - rig helpers in `tests/rig.rs`: `oracle()`, `assert_close()`, `pseudo()`

**rs8/f32 layout:** rows padded to multiples of 8 with zeros; buffer is, for each strip, K columns of 8 consecutive f32 — each column one aligned 32-byte vector. `y[r]` for `r in row_start..row_end` only; padding rows are never written.

- [ ] **Step 1: Add the tolerance home entry** (in `crates/inferno-graph/src/tolerance.rs`, after `logits_abs_tol`)

```rust
/// Kernel-GEMV vs dequant+reference-matmul, relative to max(1, max|y_ref|).
/// Quant paths are dominated by on-the-fly activation quantization (8-bit
/// blocks, ~0.4% per element); weight quantization error cancels because
/// both sides consume identical quantized weights. Initial values; tuned
/// against the observed error distributions printed by the rig's ignored
/// `observed_error_*` diagnostics (see AGENTS.md tolerance rule).
pub fn gemv_rel_tol(dtype: &DType) -> f32 {
    match dtype {
        DType::F32 => 1e-6, // fma-vs-mul+add rounding only
        DType::Q8_0 | DType::Q4_K => 2e-2,
        // No M2 kernels exist for these; the rig never asks.
        DType::F16 | DType::BF16 | DType::Unsupported(_) => 0.0,
    }
}
```

- [ ] **Step 2: Write failing tests** (`crates/inferno-kernels/tests/rig.rs`)

```rust
//! Kernel-vs-oracle rig (spec §Testing): every kernel is compared against
//! `inferno_formats::quant::dequant` + the scalar reference matmul, ISA
//! variants are compared bitwise, and row-range partitioning is bit-stable.

use inferno_formats::{DType, quant};
use inferno_graph::Tensor;
use inferno_graph::tolerance::gemv_rel_tol;
use inferno_kernels::{KernelIsa, f32k};
use proptest::prelude::*;

/// Deterministic pseudo-random f32s in [-1, 1).
fn pseudo(mut seed: u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Trusted reference: dequantize the same file-order weight bytes the kernel
/// packed, then the obviously-correct scalar matmul.
fn oracle(dtype: &DType, wbytes: &[u8], rows: usize, k: usize, x: &[f32]) -> Vec<f32> {
    let wf = quant::dequant(dtype, wbytes, rows * k).unwrap();
    let xt = Tensor { shape: vec![1, k], data: x.to_vec() };
    inferno_graph::ops::matmul(&xt, &wf, rows, k, None).data
}

fn assert_close(dtype: &DType, got: &[f32], want: &[f32]) {
    let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));
    let tol = gemv_rel_tol(dtype) * scale;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol, "row {i}: got {g}, want {w} (tol {tol})");
    }
}

// ---------- F32 ----------

fn gemv_f32(isa: KernelIsa, w: &inferno_kernels::AlignedBuf, x: &[f32], rows: usize, k: usize)
-> Vec<f32> {
    let mut y = vec![f32::NAN; rows];
    let xb: &[u8] = bytemuck_free_cast(x); // see helper below
    // SAFETY: w is a pack_f32_rs8 image for (rows, k); x has k f32; y has rows.
    unsafe {
        match isa {
            KernelIsa::Scalar => inferno_kernels::inferno_gemv_f32_rs8_scalar(
                y.as_mut_ptr(), xb.as_ptr(), w.as_ptr(), k, 0, rows,
            ),
            KernelIsa::Avx2 => inferno_kernels::inferno_gemv_f32_rs8_avx2(
                y.as_mut_ptr(), xb.as_ptr(), w.as_ptr(), k, 0, rows,
            ),
        }
    }
    y
}

/// f32 slice → its little-endian bytes (test-only; no bytemuck dep needed).
fn bytemuck_free_cast(x: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; alignment shrinks; lifetime tied to input.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len() * 4) }
}

proptest! {
    #[test]
    fn f32_gemv_matches_oracle(seed in any::<u64>(), rows in 1usize..20, k in 1usize..48) {
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 0x9e3779b97f4a7c15, k);
        let wbytes = quant::pack(&DType::F32, &vals).unwrap();
        let w = f32k::pack_f32_rs8(&wbytes, rows, k).unwrap();
        let want = oracle(&DType::F32, &wbytes, rows, k, &x);
        for isa in KernelIsa::all_available() {
            assert_close(&DType::F32, &gemv_f32(isa, &w, &x, rows, k), &want);
        }
    }

    #[test]
    fn f32_isa_variants_bitwise_equal(seed in any::<u64>(), rows in 1usize..20, k in 1usize..48) {
        if !KernelIsa::Avx2.available() { return Ok(()); }
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 1, k);
        let w = f32k::pack_f32_rs8(&quant::pack(&DType::F32, &vals).unwrap(), rows, k).unwrap();
        let a = gemv_f32(KernelIsa::Scalar, &w, &x, rows, k);
        let b = gemv_f32(KernelIsa::Avx2, &w, &x, rows, k);
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
        }
    }

    /// GEMV over [0, rows) must equal any two-part split, bitwise — the
    /// property M3's thread partitioning relies on.
    #[test]
    fn f32_range_partition_bitwise(seed in any::<u64>(), rows in 2usize..24, k in 1usize..32) {
        let split = (seed % rows as u64) as usize;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 2, k);
        let w = f32k::pack_f32_rs8(&quant::pack(&DType::F32, &vals).unwrap(), rows, k).unwrap();
        let full = gemv_f32(KernelIsa::Scalar, &w, &x, rows, k);
        for isa in KernelIsa::all_available() {
            let mut y = vec![f32::NAN; rows];
            let xb = bytemuck_free_cast(&x);
            // SAFETY: as gemv_f32, split ranges stay within rows.
            unsafe {
                let f = match isa {
                    KernelIsa::Scalar => inferno_kernels::inferno_gemv_f32_rs8_scalar
                        as unsafe extern "C" fn(*mut f32, *const u8, *const u8, usize, usize, usize),
                    KernelIsa::Avx2 => inferno_kernels::inferno_gemv_f32_rs8_avx2,
                };
                f(y.as_mut_ptr(), xb.as_ptr(), w.as_ptr(), k, 0, split);
                f(y.as_mut_ptr(), xb.as_ptr(), w.as_ptr(), k, split, rows);
            }
            for (i, (a, b)) in full.iter().zip(&y).enumerate() {
                prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
            }
        }
    }
}

#[test]
fn f32_pack_inverse() {
    let rows = 11; // partial strip
    let k = 7;
    let vals = pseudo(3, rows * k);
    let bytes = quant::pack(&DType::F32, &vals).unwrap();
    let w = f32k::pack_f32_rs8(&bytes, rows, k).unwrap();
    // Unpack: read each (row, col) back out of the strip layout.
    let p = w.as_slice();
    for r in 0..rows {
        for c in 0..k {
            let off = (((r / 8) * k + c) * 8 + r % 8) * 4;
            let got = f32::from_le_bytes(p[off..off + 4].try_into().unwrap());
            assert_eq!(got.to_bits(), vals[r * k + c].to_bits(), "({r},{c})");
        }
    }
    // Padding rows are zero.
    assert_eq!(w.len(), f32k::packed_len_f32_rs8(rows, k));
    for c in 0..k {
        for lane in 3..8 {
            // rows 11..16
            let off = ((8usize / 8 * k + c) * 8 + lane) * 4;
            assert_eq!(&p[off..off + 4], &[0u8; 4]);
        }
    }
}

#[test]
fn f32_empty_range_is_noop_and_pack_validates() {
    let vals = pseudo(4, 8 * 4);
    let w = f32k::pack_f32_rs8(&quant::pack(&DType::F32, &vals).unwrap(), 8, 4).unwrap();
    let x = pseudo(5, 4);
    let mut y = vec![42f32; 8];
    // SAFETY: valid image; empty range must not touch y.
    unsafe {
        inferno_kernels::inferno_gemv_f32_rs8_scalar(
            y.as_mut_ptr(),
            bytemuck_free_cast(&x).as_ptr(),
            w.as_ptr(),
            4,
            5,
            5,
        );
    }
    assert!(y.iter().all(|&v| v == 42.0));
    assert!(f32k::pack_f32_rs8(&[0u8; 12], 2, 2).is_err()); // 12 != 16
    assert!(f32k::pack_f32_rs8(&[], 0, 4).is_err());
    assert!(f32k::pack_f32_rs8(&[], 4, 0).is_err());
}
```

Note: `tests/rig.rs` is an *integration* test — it calls the raw `pub unsafe extern "C"` symbols directly (in-crate `unsafe` is fine in a test that states the contract). The lint override lives in the kernels crate, so this file may use `unsafe`.

- [ ] **Step 3: Run to verify failure**

Run: `cargo nextest run -p inferno-kernels rig`
Expected: compile error (`f32k` missing).

- [ ] **Step 4: Implement `src/f32k.rs`** (and add `pub mod f32k;` plus re-export nothing — callers use the module path; but re-export the symbols at crate root for the rig: add `pub use f32k::{inferno_gemv_f32_rs8_avx2, inferno_gemv_f32_rs8_scalar};` to lib.rs)

```rust
//! F32 GEMV in the rs8 layout — the trivial baseline that validates the rig
//! itself (spec §Scope). Layout: rows padded to strips of 8; per strip, K
//! columns of 8 consecutive f32 — one aligned 32-byte vector per column.

use crate::{AlignedBuf, KernelError, Result, STRIP};

pub fn packed_len_f32_rs8(rows: usize, k: usize) -> usize {
    rows.next_multiple_of(STRIP) * k * 4
}

pub fn pack_f32_rs8(bytes: &[u8], rows: usize, k: usize) -> Result<AlignedBuf> {
    if rows == 0 {
        return Err(KernelError::ZeroRows);
    }
    if k == 0 {
        return Err(KernelError::BadK { k, block: 1 });
    }
    let expected =
        rows.checked_mul(k).and_then(|n| n.checked_mul(4)).ok_or(KernelError::Overflow)?;
    if bytes.len() != expected {
        return Err(KernelError::SizeMismatch { what: "f32 weight bytes", got: bytes.len(), expected });
    }
    let mut out = AlignedBuf::zeroed(packed_len_f32_rs8(rows, k));
    let dst = out.as_mut_slice();
    for r in 0..rows {
        let (strip, lane) = (r / STRIP, r % STRIP);
        for c in 0..k {
            let s = (r * k + c) * 4;
            let d = (((strip * k) + c) * STRIP + lane) * 4;
            dst[d..d + 4].copy_from_slice(&bytes[s..s + 4]);
        }
    }
    Ok(out)
}

/// Scalar row loop. Both ISA symbols route partial strips here so every
/// variant computes the identical fma sequence per row (bitwise contract).
///
/// # Safety
/// Contract of [`inferno_gemv_f32_rs8_scalar`].
unsafe fn gemv_rows(y: *mut f32, x: *const f32, w: *const f32, k: usize, r0: usize, r1: usize) {
    for r in r0..r1 {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let base = unsafe { w.add(strip * k * STRIP + lane) };
        let mut acc = 0f32;
        for c in 0..k {
            let wv = unsafe { base.add(c * STRIP).read() };
            acc = wv.mul_add(unsafe { x.add(c).read() }, acc);
        }
        unsafe { y.add(r).write(acc) };
    }
}

/// Unified GEMV ABI (all dtypes): `y[row_start..row_end] = W · x`.
///
/// # Safety
/// - `y` valid for f32 writes at indices `row_start..row_end`.
/// - `x` points at the activation buffer — for F32, `k` raw little-endian
///   f32 values (4-byte aligned).
/// - `w` is a `pack_f32_rs8` image built with this exact `k` and at least
///   `row_end` rows, 32-byte aligned (guaranteed by `AlignedBuf`).
/// - `row_start <= row_end`; all values finite.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_f32_rs8_scalar(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    unsafe { gemv_rows(y, x.cast(), w.cast(), k, row_start, row_end) }
}

/// # Safety
/// As [`inferno_gemv_f32_rs8_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_f32_rs8_avx2(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let xf = x.cast::<f32>();
    let wf = w.cast::<f32>();
    let mut r = row_start;
    let head = row_start.next_multiple_of(STRIP).min(row_end);
    if head > r {
        unsafe { gemv_rows(y, xf, wf, k, r, head) };
        r = head;
    }
    while r + STRIP <= row_end {
        let base = unsafe { wf.add((r / STRIP) * k * STRIP) };
        let mut acc = _mm256_setzero_ps();
        for c in 0..k {
            let wv = unsafe { _mm256_load_ps(base.add(c * STRIP)) };
            let xv = _mm256_set1_ps(unsafe { xf.add(c).read() });
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        unsafe { _mm256_storeu_ps(y.add(r), acc) };
        r += STRIP;
    }
    if r < row_end {
        unsafe { gemv_rows(y, xf, wf, k, r, row_end) };
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo nextest run -p inferno-kernels`
Expected: all PASS (workspace still green: `mise run test`).

- [ ] **Step 6: Lint and commit**

```bash
mise run lint && mise run test
git add -A && git commit -m "feat(kernels): f32 rs8 GEMV baseline and the kernel-vs-oracle rig"
```

---

### Task 5: Q8_0 rs8 kernels

**Files:**
- Create: `crates/inferno-kernels/src/q8_0.rs` (add `pub mod q8_0;` and symbol re-exports to lib.rs)
- Modify: `crates/inferno-kernels/tests/rig.rs` (Q8_0 section)

**Interfaces:**
- Consumes: `act::{quantize_row_q8a, q8a_len, Q8A_BLOCK, Q8A_BLOCK_BYTES}`, `act::hsum_i32`, `AlignedBuf`, `STRIP`, `inferno_formats::quant::f16_to_f32`.
- Produces (used by Tasks 7, 8):
  - `q8_0::packed_len_q8_0_rs8(rows, k) -> usize`
  - `q8_0::pack_q8_0_rs8(bytes: &[u8], rows: usize, k: usize) -> Result<AlignedBuf>`
  - C symbols `inferno_gemv_q8_0_rs8_{scalar,avx2}` (unified GEMV ABI; `x` is a q8a buffer)

**rs8/q8_0 layout.** `k` must be a positive multiple of 32; `nb = k/32`. Per (strip, block) group, 288 bytes: `[d: 8 × f32]` (32 B, one per lane, converted from the file's f16 at pack time) then `[qs: 8 × 32 i8]` (row-major by lane). Group offset `(strip*nb + b) * 288`; everything stays 32-byte aligned (288 = 9×32). **Pack clamps qs byte `-128 → -127`** (see Deviations §5).

**Math (identical in both ISAs, bitwise):** per row, `acc = Σ_b (d_w[b] * d_x[b]).mul_add(isum_b as f32, acc)` in block order, where `isum_b = Σ_{i<32} qw_i * qx_i` exactly in i32.

- [ ] **Step 1: Write failing tests** (append to `tests/rig.rs`)

```rust
// ---------- Q8_0 ----------

use inferno_kernels::{act, q8_0};

fn gemv_q8_0(
    isa: KernelIsa,
    w: &inferno_kernels::AlignedBuf,
    xq: &[u8],
    rows: usize,
    k: usize,
    range: (usize, usize),
    y: &mut [f32],
) {
    // SAFETY: w is a pack_q8_0_rs8 image for (rows, k); xq is a q8a buffer
    // for k; y has rows elements; range within rows.
    unsafe {
        match isa {
            KernelIsa::Scalar => inferno_kernels::inferno_gemv_q8_0_rs8_scalar(
                y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, range.0, range.1,
            ),
            KernelIsa::Avx2 => inferno_kernels::inferno_gemv_q8_0_rs8_avx2(
                y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, range.0, range.1,
            ),
        }
    }
}

proptest! {
    #[test]
    fn q8_0_gemv_matches_oracle(seed in any::<u64>(), rows in 1usize..20, nb in 1usize..5) {
        let k = nb * 32;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 0xabcdef, k);
        let wbytes = quant::pack(&DType::Q8_0, &vals).unwrap();
        let w = q8_0::pack_q8_0_rs8(&wbytes, rows, k).unwrap();
        let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
        // Oracle consumes the same quantized *weights*; activation quant is
        // the kernel's own error and must fit gemv_rel_tol.
        let want = oracle(&DType::Q8_0, &wbytes, rows, k, &x);
        for isa in KernelIsa::all_available() {
            let mut y = vec![f32::NAN; rows];
            gemv_q8_0(isa, &w, &xq, rows, k, (0, rows), &mut y);
            assert_close(&DType::Q8_0, &y, &want);
        }
    }

    #[test]
    fn q8_0_isa_variants_bitwise_equal(seed in any::<u64>(), rows in 1usize..20, nb in 1usize..5) {
        if !KernelIsa::Avx2.available() { return Ok(()); }
        let k = nb * 32;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 3, k);
        let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
        let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
        let (mut a, mut b) = (vec![f32::NAN; rows], vec![f32::NAN; rows]);
        gemv_q8_0(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut a);
        gemv_q8_0(KernelIsa::Avx2, &w, &xq, rows, k, (0, rows), &mut b);
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
        }
    }

    #[test]
    fn q8_0_range_partition_bitwise(seed in any::<u64>(), rows in 2usize..24, nb in 1usize..4) {
        let k = nb * 32;
        let split = (seed % rows as u64) as usize;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 4, k);
        let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
        let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
        let mut full = vec![f32::NAN; rows];
        gemv_q8_0(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut full);
        for isa in KernelIsa::all_available() {
            let mut y = vec![f32::NAN; rows];
            gemv_q8_0(isa, &w, &xq, rows, k, (0, split), &mut y);
            gemv_q8_0(isa, &w, &xq, rows, k, (split, rows), &mut y);
            for (i, (a, b)) in full.iter().zip(&y).enumerate() {
                prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
            }
        }
    }
}

/// Pack inverse via normalized blocks: parse the file bytes and the packed
/// image to the same (d, qs) structure — localizes layout bugs (spec §Testing).
#[test]
fn q8_0_pack_inverse() {
    let (rows, k) = (11usize, 64usize); // partial strip, 2 blocks
    let nb = k / 32;
    let vals = pseudo(7, rows * k);
    let bytes = quant::pack(&DType::Q8_0, &vals).unwrap();
    let w = q8_0::pack_q8_0_rs8(&bytes, rows, k).unwrap();
    let p = w.as_slice();
    for r in 0..rows {
        for b in 0..nb {
            let s = (r * nb + b) * 34;
            let file_d = quant::f16_to_f32(u16::from_le_bytes([bytes[s], bytes[s + 1]]));
            let file_qs = &bytes[s + 2..s + 34];
            let g = ((r / 8) * nb + b) * 288;
            let lane = r % 8;
            let packed_d =
                f32::from_le_bytes(p[g + lane * 4..g + lane * 4 + 4].try_into().unwrap());
            let packed_qs = &p[g + 32 + lane * 32..g + 32 + (lane + 1) * 32];
            assert_eq!(packed_d.to_bits(), file_d.to_bits(), "({r},{b}) d");
            assert_eq!(packed_qs, file_qs, "({r},{b}) qs");
        }
    }
}

#[test]
fn q8_0_pack_clamps_minus_128() {
    // Hand-build one block whose qs are all -128 (hostile file).
    let mut bytes = vec![0u8; 34];
    bytes[..2].copy_from_slice(&quant::f32_to_f16(1.0).to_le_bytes());
    for b in &mut bytes[2..] {
        *b = (-128i8) as u8;
    }
    let w = q8_0::pack_q8_0_rs8(&bytes, 1, 32).unwrap();
    let p = w.as_slice();
    for i in 0..32 {
        assert_eq!(p[32 + i] as i8, -127);
    }
}

/// Max-scale block (spec edge case): every value at the block amax, so
/// quantized weights and activations all saturate to ±127.
#[test]
fn q8_0_saturated_block_matches_oracle() {
    let (rows, k) = (3usize, 32usize);
    let vals: Vec<f32> = (0..rows * k).map(|i| if i % 2 == 0 { 10.0 } else { -10.0 }).collect();
    let x: Vec<f32> = (0..k).map(|i| if i % 2 == 0 { 8.0 } else { -8.0 }).collect();
    let wbytes = quant::pack(&DType::Q8_0, &vals).unwrap();
    let w = q8_0::pack_q8_0_rs8(&wbytes, rows, k).unwrap();
    let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
    let want = oracle(&DType::Q8_0, &wbytes, rows, k, &x);
    for isa in KernelIsa::all_available() {
        let mut y = vec![f32::NAN; rows];
        gemv_q8_0(isa, &w, &xq, rows, k, (0, rows), &mut y);
        assert_close(&DType::Q8_0, &y, &want);
    }
}

#[test]
fn q8_0_pack_validates() {
    assert!(q8_0::pack_q8_0_rs8(&[0u8; 34], 1, 31).is_err()); // k not multiple of 32
    assert!(q8_0::pack_q8_0_rs8(&[0u8; 33], 1, 32).is_err()); // wrong byte count
    assert!(q8_0::pack_q8_0_rs8(&[], 0, 32).is_err());
}

/// Ignored diagnostic: prints the observed max relative error so
/// gemv_rel_tol(Q8_0) is tuned from data (AGENTS.md tolerance rule).
/// Run: cargo nextest run -p inferno-kernels --run-ignored all observed_error_q8_0 --no-capture
#[test]
#[ignore = "diagnostic; prints observed gemv error distribution"]
fn observed_error_q8_0() {
    let mut max_rel = 0f32;
    for seed in 0..500u64 {
        let (rows, k) = (16usize, 128usize);
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 99, k);
        let wbytes = quant::pack(&DType::Q8_0, &vals).unwrap();
        let w = q8_0::pack_q8_0_rs8(&wbytes, rows, k).unwrap();
        let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
        let want = oracle(&DType::Q8_0, &wbytes, rows, k, &x);
        let mut y = vec![f32::NAN; rows];
        gemv_q8_0(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut y);
        let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));
        for (g, w_) in y.iter().zip(&want) {
            max_rel = max_rel.max((g - w_).abs() / scale);
        }
    }
    println!("q8_0 observed max rel error: {max_rel:e} (tol {:e})", gemv_rel_tol(&DType::Q8_0));
}
```

Also add `f32_to_f16` to the `quant` imports used above (it is `pub` in `inferno_formats::quant`).

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p inferno-kernels rig`
Expected: compile error (`q8_0` missing).

- [ ] **Step 3: Implement `src/q8_0.rs`** (add `pub mod q8_0;` and `pub use q8_0::{inferno_gemv_q8_0_rs8_avx2, inferno_gemv_q8_0_rs8_scalar};` to lib.rs)

```rust
//! Q8_0 GEMV in the rs8 layout. Weights: ggml Q8_0 blocks (f16 scale + 32 i8)
//! repacked per strip with scales widened to f32. Activations: q8a. Integer
//! dots are exact; the f32 combine runs in block order in every variant.

use inferno_formats::quant::f16_to_f32;

use crate::act::{Q8A_BLOCK_BYTES, hsum_i32};
use crate::{AlignedBuf, KernelError, Result, STRIP};

const WBLOCK: usize = 32; // weight elements per block
const FILE_BLOCK_BYTES: usize = 34; // f16 d + 32 i8
const GROUP_BYTES: usize = 288; // 8 f32 d + 8×32 qs

pub fn packed_len_q8_0_rs8(rows: usize, k: usize) -> usize {
    rows.next_multiple_of(STRIP) / STRIP * (k / WBLOCK) * GROUP_BYTES
}

/// Repack file-order Q8_0 blocks into rs8 groups. Clamps qs `-128 → -127`
/// so the AVX2 sign-trick stays exact on hostile files (plan Deviations §5);
/// ggml's own quantizer never emits −128, so real files are unchanged.
pub fn pack_q8_0_rs8(bytes: &[u8], rows: usize, k: usize) -> Result<AlignedBuf> {
    if rows == 0 {
        return Err(KernelError::ZeroRows);
    }
    if k == 0 || k % WBLOCK != 0 {
        return Err(KernelError::BadK { k, block: WBLOCK });
    }
    let nb = k / WBLOCK;
    let expected = rows
        .checked_mul(nb)
        .and_then(|n| n.checked_mul(FILE_BLOCK_BYTES))
        .ok_or(KernelError::Overflow)?;
    if bytes.len() != expected {
        return Err(KernelError::SizeMismatch {
            what: "Q8_0 weight bytes",
            got: bytes.len(),
            expected,
        });
    }
    let mut out = AlignedBuf::zeroed(packed_len_q8_0_rs8(rows, k));
    let dst = out.as_mut_slice();
    for r in 0..rows {
        let (strip, lane) = (r / STRIP, r % STRIP);
        for b in 0..nb {
            let s = (r * nb + b) * FILE_BLOCK_BYTES;
            let g = (strip * nb + b) * GROUP_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([bytes[s], bytes[s + 1]]));
            dst[g + lane * 4..g + lane * 4 + 4].copy_from_slice(&d.to_le_bytes());
            for (i, &q) in bytes[s + 2..s + 2 + WBLOCK].iter().enumerate() {
                dst[g + 32 + lane * WBLOCK + i] = if q as i8 == i8::MIN { -127i8 as u8 } else { q };
            }
        }
    }
    Ok(out)
}

/// Unified GEMV ABI: `y[row_start..row_end] = W · dequant(x)`.
///
/// # Safety
/// - `y` valid for f32 writes at `row_start..row_end`.
/// - `x` is a q8a buffer for this `k` (from `inferno_quantize_row_q8a_*`).
/// - `w` is a `pack_q8_0_rs8` image built with this exact `k` and at least
///   `row_end` rows, 32-byte aligned.
/// - `row_start <= row_end`; `k` a positive multiple of 32.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_q8_0_rs8_scalar(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    let nb = k / WBLOCK;
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let mut acc = 0f32;
        for b in 0..nb {
            let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
            let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
            let qw = unsafe { g.add(32 + lane * WBLOCK) };
            let qx = unsafe { xb.add(4) };
            let mut isum = 0i32;
            for i in 0..WBLOCK {
                let a = i32::from(unsafe { qw.add(i).cast::<i8>().read() });
                let b_ = i32::from(unsafe { qx.add(i).cast::<i8>().read() });
                isum += a * b_;
            }
            acc = (dw * dx).mul_add(isum as f32, acc);
        }
        unsafe { y.add(r).write(acc) };
    }
}

/// # Safety
/// As [`inferno_gemv_q8_0_rs8_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_q8_0_rs8_avx2(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let nb = k / WBLOCK;
    let ones = _mm256_set1_epi16(1);
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let mut acc = 0f32;
        for b in 0..nb {
            let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
            let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
            // Aligned: group is 32-aligned, +32, lane*32.
            let wv = unsafe { _mm256_load_si256(g.add(32 + lane * WBLOCK).cast()) };
            let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
            // Sign trick (both operands in [-127,127] by pack/quantize):
            // |w| as u8 × sign-adjusted x, exact in i16/i32.
            let aw = _mm256_sign_epi8(wv, wv);
            let sx = _mm256_sign_epi8(xv, wv);
            let p16 = _mm256_maddubs_epi16(aw, sx);
            let p32 = _mm256_madd_epi16(p16, ones);
            let isum = hsum_i32(p32);
            acc = (dw * dx).mul_add(isum as f32, acc);
        }
        unsafe { y.add(r).write(acc) };
    }
}
```

`hsum_i32` moves from a private fn in `act.rs` to `pub(crate)` (it already is in the Task 3 listing).

- [ ] **Step 4: Run tests, record observed error**

```bash
cargo nextest run -p inferno-kernels
cargo nextest run -p inferno-kernels --run-ignored all -E 'test(observed_error_q8_0)' --no-capture
```

Expected: all PASS. Note the printed `observed max rel error`. If it is more than ~10× below `2e-2`, tighten `gemv_rel_tol(Q8_0)` in `tolerance.rs` to ~4× the observed max and record the observation in the comment (mirror of the `LOGIT_TIE_EPSILON` convention). Re-run `mise run test` after any tolerance change.

- [ ] **Step 5: Lint and commit**

```bash
mise run lint && mise run test
git add -A && git commit -m "feat(kernels): Q8_0 rs8 GEMV with exact integer AVX2 dots"
```

---

### Task 6: Q4_K rs8 kernels

**Files:**
- Modify: `crates/inferno-formats/src/quant.rs` (make `get_scale_min_k4` pub)
- Create: `crates/inferno-kernels/src/q4_k.rs` (add `pub mod q4_k;` + symbol re-exports to lib.rs)
- Modify: `crates/inferno-kernels/tests/rig.rs` (Q4_K section)

**Interfaces:**
- Consumes: `act::{quantize_row_q8k, Q8K_BLOCK_BYTES}`, `act::hsum_i32`, `inferno_formats::quant::{f16_to_f32, get_scale_min_k4}`.
- Produces (used by Tasks 7, 8):
  - `q4_k::packed_len_q4_k_rs8(rows, k) -> usize`
  - `q4_k::pack_q4_k_rs8(bytes: &[u8], rows: usize, k: usize) -> Result<AlignedBuf>`
  - C symbols `inferno_gemv_q4_k_rs8_{scalar,avx2}` (unified GEMV ABI; `x` is a q8k buffer)

**rs8/q4_k layout.** `k` a positive multiple of 256; `nsb = k/256`. Per (strip, super-block) group, 1216 bytes (38×32, stays aligned):

| offset | content |
|---|---|
| 0 | `d`: 8 × f32 (per lane, from file f16) |
| 32 | `dmin`: 8 × f32 |
| 64 | `sc`: 8 lanes × 8 u8 — 6-bit scales **decoded at pack time** via `get_scale_min_k4` |
| 128 | `m`: 8 lanes × 8 u8 — decoded mins |
| 192 | `qs`: 8 lanes × 128 B nibble data (file order per lane) |

Decoding the 6-bit scale packing once at pack time keeps all bit-twiddling out of the inner loop.

**Math (identical in both ISAs, bitwise):** per row and super-block, with the file's chunk structure (chunk `c ∈ 0..4` holds sub-block `2c` in low nibbles, `2c+1` in high nibbles):

```
sumd = Σ_c ( sc[2c] * dot(lo_nibbles_c, q8[c*64 .. c*64+32])
           + sc[2c+1] * dot(hi_nibbles_c, q8[c*64+32 .. c*64+64]) )   (exact i32)
summ = Σ_j m[j] * bsums[j]                                            (exact i32)
acc  = (d_w * d_x).mul_add(sumd as f32, acc)
acc  = (dmin_w * d_x).mul_add(-(summ as f32), acc)
```

Bounds: `dot ≤ 32·15·127 = 60 960`, `sumd ≤ 8·63·60 960 < 2^25`, `summ ≤ 8·63·4064 < 2^22` — comfortably i32.

- [ ] **Step 1: Make the scale decoder public** (in `crates/inferno-formats/src/quant.rs`)

Change the signature line of `get_scale_min_k4` to:

```rust
/// ggml Q4_K scale/min extraction: 8 six-bit (scale, min) pairs in 12 bytes.
/// Public because `inferno-kernels` decodes scales at pack time (M2).
pub fn get_scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
```

Run: `cargo nextest run -p inferno-formats` → PASS (visibility-only change), then both fuzz targets briefly (Global Constraints):

```bash
mise run fuzz -- gguf_parse
mise run fuzz -- safetensors_parse
```

Expected: no crashes.

- [ ] **Step 2: Write failing tests** (append to `tests/rig.rs`)

```rust
// ---------- Q4_K ----------

use inferno_kernels::q4_k;

fn gemv_q4_k(
    isa: KernelIsa,
    w: &inferno_kernels::AlignedBuf,
    xq: &[u8],
    rows: usize,
    k: usize,
    range: (usize, usize),
    y: &mut [f32],
) {
    // SAFETY: w is a pack_q4_k_rs8 image for (rows, k); xq is a q8k buffer
    // for k; y has rows elements; range within rows.
    unsafe {
        match isa {
            KernelIsa::Scalar => inferno_kernels::inferno_gemv_q4_k_rs8_scalar(
                y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, range.0, range.1,
            ),
            KernelIsa::Avx2 => inferno_kernels::inferno_gemv_q4_k_rs8_avx2(
                y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, range.0, range.1,
            ),
        }
    }
}

proptest! {
    #[test]
    fn q4_k_gemv_matches_oracle(seed in any::<u64>(), rows in 1usize..20, nsb in 1usize..3) {
        let k = nsb * 256;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 0x51ed, k);
        let wbytes = quant::pack(&DType::Q4_K, &vals).unwrap();
        let w = q4_k::pack_q4_k_rs8(&wbytes, rows, k).unwrap();
        let xq = act::quantize_row_q8k(KernelIsa::Scalar, &x).unwrap();
        let want = oracle(&DType::Q4_K, &wbytes, rows, k, &x);
        for isa in KernelIsa::all_available() {
            let mut y = vec![f32::NAN; rows];
            gemv_q4_k(isa, &w, &xq, rows, k, (0, rows), &mut y);
            assert_close(&DType::Q4_K, &y, &want);
        }
    }

    #[test]
    fn q4_k_isa_variants_bitwise_equal(seed in any::<u64>(), rows in 1usize..20) {
        if !KernelIsa::Avx2.available() { return Ok(()); }
        let k = 512usize;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 5, k);
        let w = q4_k::pack_q4_k_rs8(&quant::pack(&DType::Q4_K, &vals).unwrap(), rows, k).unwrap();
        let xq = act::quantize_row_q8k(KernelIsa::Scalar, &x).unwrap();
        let (mut a, mut b) = (vec![f32::NAN; rows], vec![f32::NAN; rows]);
        gemv_q4_k(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut a);
        gemv_q4_k(KernelIsa::Avx2, &w, &xq, rows, k, (0, rows), &mut b);
        for (i, (a, b)) in a.iter().zip(&b).enumerate() {
            prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
        }
    }

    #[test]
    fn q4_k_range_partition_bitwise(seed in any::<u64>(), rows in 2usize..24) {
        let k = 256usize;
        let split = (seed % rows as u64) as usize;
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 6, k);
        let w = q4_k::pack_q4_k_rs8(&quant::pack(&DType::Q4_K, &vals).unwrap(), rows, k).unwrap();
        let xq = act::quantize_row_q8k(KernelIsa::Scalar, &x).unwrap();
        let mut full = vec![f32::NAN; rows];
        gemv_q4_k(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut full);
        for isa in KernelIsa::all_available() {
            let mut y = vec![f32::NAN; rows];
            gemv_q4_k(isa, &w, &xq, rows, k, (0, split), &mut y);
            gemv_q4_k(isa, &w, &xq, rows, k, (split, rows), &mut y);
            for (i, (a, b)) in full.iter().zip(&y).enumerate() {
                prop_assert_eq!(a.to_bits(), b.to_bits(), "row {}", i);
            }
        }
    }
}

/// Pack inverse via normalized super-blocks (spec §Testing).
#[test]
fn q4_k_pack_inverse() {
    use inferno_formats::quant::get_scale_min_k4;
    let (rows, k) = (9usize, 256usize);
    let vals = pseudo(11, rows * k);
    let bytes = quant::pack(&DType::Q4_K, &vals).unwrap();
    let w = q4_k::pack_q4_k_rs8(&bytes, rows, k).unwrap();
    let p = w.as_slice();
    for r in 0..rows {
        let s = r * 144; // one super-block per row at k=256
        let file_d = quant::f16_to_f32(u16::from_le_bytes([bytes[s], bytes[s + 1]]));
        let file_dmin = quant::f16_to_f32(u16::from_le_bytes([bytes[s + 2], bytes[s + 3]]));
        let g = (r / 8) * 1216;
        let lane = r % 8;
        let pd = f32::from_le_bytes(p[g + lane * 4..g + lane * 4 + 4].try_into().unwrap());
        let pdmin =
            f32::from_le_bytes(p[g + 32 + lane * 4..g + 32 + lane * 4 + 4].try_into().unwrap());
        assert_eq!(pd.to_bits(), file_d.to_bits(), "row {r} d");
        assert_eq!(pdmin.to_bits(), file_dmin.to_bits(), "row {r} dmin");
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &bytes[s + 4..s + 16]);
            assert_eq!(p[g + 64 + lane * 8 + j], sc, "row {r} sc[{j}]");
            assert_eq!(p[g + 128 + lane * 8 + j], m, "row {r} m[{j}]");
        }
        assert_eq!(&p[g + 192 + lane * 128..g + 192 + (lane + 1) * 128], &bytes[s + 16..s + 144]);
    }
}

#[test]
fn q4_k_pack_validates() {
    assert!(q4_k::pack_q4_k_rs8(&[0u8; 144], 1, 255).is_err());
    assert!(q4_k::pack_q4_k_rs8(&[0u8; 143], 1, 256).is_err());
    assert!(q4_k::pack_q4_k_rs8(&[], 0, 256).is_err());
}

/// Ignored diagnostic (see observed_error_q8_0).
#[test]
#[ignore = "diagnostic; prints observed gemv error distribution"]
fn observed_error_q4_k() {
    let mut max_rel = 0f32;
    for seed in 0..500u64 {
        let (rows, k) = (16usize, 512usize);
        let vals = pseudo(seed, rows * k);
        let x = pseudo(seed ^ 77, k);
        let wbytes = quant::pack(&DType::Q4_K, &vals).unwrap();
        let w = q4_k::pack_q4_k_rs8(&wbytes, rows, k).unwrap();
        let xq = act::quantize_row_q8k(KernelIsa::Scalar, &x).unwrap();
        let want = oracle(&DType::Q4_K, &wbytes, rows, k, &x);
        let mut y = vec![f32::NAN; rows];
        gemv_q4_k(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut y);
        let scale = want.iter().fold(1f32, |m, v| m.max(v.abs()));
        for (g, w_) in y.iter().zip(&want) {
            max_rel = max_rel.max((g - w_).abs() / scale);
        }
    }
    println!("q4_k observed max rel error: {max_rel:e} (tol {:e})", gemv_rel_tol(&DType::Q4_K));
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo nextest run -p inferno-kernels rig`
Expected: compile error (`q4_k` missing).

- [ ] **Step 4: Implement `src/q4_k.rs`** (add `pub mod q4_k;` and `pub use q4_k::{inferno_gemv_q4_k_rs8_avx2, inferno_gemv_q4_k_rs8_scalar};` to lib.rs)

```rust
//! Q4_K GEMV in the rs8 layout. Weights: ggml 144-byte super-blocks with the
//! 6-bit (scale, min) packing decoded to plain u8 at pack time. Activations:
//! q8k (bsums feed the dmin correction). Integer dots exact; f32 combine in
//! fixed order — ISA variants are bit-identical.

use inferno_formats::quant::{f16_to_f32, get_scale_min_k4};

use crate::act::{Q8K_BLOCK_BYTES, hsum_i32};
use crate::{AlignedBuf, KernelError, Result, STRIP};

const WBLOCK: usize = 256; // weight elements per super-block
const FILE_SB_BYTES: usize = 144;
const GROUP_BYTES: usize = 1216; // 32 d + 32 dmin + 64 sc + 64 m + 1024 qs
const OFF_DMIN: usize = 32;
const OFF_SC: usize = 64;
const OFF_M: usize = 128;
const OFF_QS: usize = 192;

pub fn packed_len_q4_k_rs8(rows: usize, k: usize) -> usize {
    rows.next_multiple_of(STRIP) / STRIP * (k / WBLOCK) * GROUP_BYTES
}

pub fn pack_q4_k_rs8(bytes: &[u8], rows: usize, k: usize) -> Result<AlignedBuf> {
    if rows == 0 {
        return Err(KernelError::ZeroRows);
    }
    if k == 0 || k % WBLOCK != 0 {
        return Err(KernelError::BadK { k, block: WBLOCK });
    }
    let nsb = k / WBLOCK;
    let expected = rows
        .checked_mul(nsb)
        .and_then(|n| n.checked_mul(FILE_SB_BYTES))
        .ok_or(KernelError::Overflow)?;
    if bytes.len() != expected {
        return Err(KernelError::SizeMismatch {
            what: "Q4_K weight bytes",
            got: bytes.len(),
            expected,
        });
    }
    let mut out = AlignedBuf::zeroed(packed_len_q4_k_rs8(rows, k));
    let dst = out.as_mut_slice();
    for r in 0..rows {
        let (strip, lane) = (r / STRIP, r % STRIP);
        for sb in 0..nsb {
            let s = (r * nsb + sb) * FILE_SB_BYTES;
            let g = (strip * nsb + sb) * GROUP_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([bytes[s], bytes[s + 1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([bytes[s + 2], bytes[s + 3]]));
            dst[g + lane * 4..g + lane * 4 + 4].copy_from_slice(&d.to_le_bytes());
            dst[g + OFF_DMIN + lane * 4..g + OFF_DMIN + lane * 4 + 4]
                .copy_from_slice(&dmin.to_le_bytes());
            for j in 0..8 {
                let (sc, m) = get_scale_min_k4(j, &bytes[s + 4..s + 16]);
                dst[g + OFF_SC + lane * 8 + j] = sc;
                dst[g + OFF_M + lane * 8 + j] = m;
            }
            dst[g + OFF_QS + lane * 128..g + OFF_QS + (lane + 1) * 128]
                .copy_from_slice(&bytes[s + 16..s + 144]);
        }
    }
    Ok(out)
}

/// Unified GEMV ABI: `y[row_start..row_end] = W · dequant(x)`.
///
/// # Safety
/// - `y` valid for f32 writes at `row_start..row_end`.
/// - `x` is a q8k buffer for this `k` (from `inferno_quantize_row_q8k_*`).
/// - `w` is a `pack_q4_k_rs8` image built with this exact `k` and at least
///   `row_end` rows, 32-byte aligned.
/// - `row_start <= row_end`; `k` a positive multiple of 256.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_q4_k_rs8_scalar(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    let nsb = k / WBLOCK;
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let mut acc = 0f32;
        for sb in 0..nsb {
            let g = unsafe { w.add((strip * nsb + sb) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let dmin =
                f32::from_le_bytes(unsafe { g.add(OFF_DMIN + lane * 4).cast::<[u8; 4]>().read() });
            let sc = unsafe { g.add(OFF_SC + lane * 8) };
            let m = unsafe { g.add(OFF_M + lane * 8) };
            let qs = unsafe { g.add(OFF_QS + lane * 128) };
            let xb = unsafe { x.add(sb * Q8K_BLOCK_BYTES) };
            let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
            let xqs = unsafe { xb.add(4) };
            let mut summ = 0i32;
            for j in 0..8 {
                let bsum =
                    i32::from_le_bytes(unsafe { xb.add(260 + j * 4).cast::<[u8; 4]>().read_unaligned() });
                summ += i32::from(unsafe { m.add(j).read() }) * bsum;
            }
            let mut sumd = 0i32;
            for c in 0..4 {
                let (mut dlo, mut dhi) = (0i32, 0i32);
                for i in 0..32 {
                    let qb = unsafe { qs.add(c * 32 + i).read() };
                    let lo = i32::from(qb & 0xF);
                    let hi = i32::from(qb >> 4);
                    dlo += lo * i32::from(unsafe { xqs.add(c * 64 + i).cast::<i8>().read() });
                    dhi += hi * i32::from(unsafe { xqs.add(c * 64 + 32 + i).cast::<i8>().read() });
                }
                sumd += i32::from(unsafe { sc.add(2 * c).read() }) * dlo
                    + i32::from(unsafe { sc.add(2 * c + 1).read() }) * dhi;
            }
            acc = (dw * dx).mul_add(sumd as f32, acc);
            acc = (dmin * dx).mul_add(-(summ as f32), acc);
        }
        unsafe { y.add(r).write(acc) };
    }
}

/// # Safety
/// As [`inferno_gemv_q4_k_rs8_scalar`]; additionally requires AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemv_q4_k_rs8_avx2(
    y: *mut f32,
    x: *const u8,
    w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let nsb = k / WBLOCK;
    let ones = _mm256_set1_epi16(1);
    let nib = _mm256_set1_epi8(0x0F);
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        let mut acc = 0f32;
        for sb in 0..nsb {
            let g = unsafe { w.add((strip * nsb + sb) * GROUP_BYTES) };
            let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
            let dmin =
                f32::from_le_bytes(unsafe { g.add(OFF_DMIN + lane * 4).cast::<[u8; 4]>().read() });
            let sc = unsafe { g.add(OFF_SC + lane * 8) };
            let m = unsafe { g.add(OFF_M + lane * 8) };
            let qs = unsafe { g.add(OFF_QS + lane * 128) };
            let xb = unsafe { x.add(sb * Q8K_BLOCK_BYTES) };
            let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
            let xqs = unsafe { xb.add(4) };
            let mut summ = 0i32;
            for j in 0..8 {
                let bsum =
                    i32::from_le_bytes(unsafe { xb.add(260 + j * 4).cast::<[u8; 4]>().read_unaligned() });
                summ += i32::from(unsafe { m.add(j).read() }) * bsum;
            }
            let mut sumd = 0i32;
            for c in 0..4 {
                // Aligned: g 32-aligned, OFF_QS=192, lane*128, c*32.
                let qv = unsafe { _mm256_load_si256(qs.add(c * 32).cast()) };
                let lo = _mm256_and_si256(qv, nib);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qv), nib);
                let x_lo = unsafe { _mm256_loadu_si256(xqs.add(c * 64).cast()) };
                let x_hi = unsafe { _mm256_loadu_si256(xqs.add(c * 64 + 32).cast()) };
                // Nibbles are unsigned 0..=15 → valid u8 operand for maddubs;
                // pairs sum ≤ 2·15·127 < i16::MAX, no saturation.
                let dlo = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(lo, x_lo), ones));
                let dhi = hsum_i32(_mm256_madd_epi16(_mm256_maddubs_epi16(hi, x_hi), ones));
                sumd += i32::from(unsafe { sc.add(2 * c).read() }) * dlo
                    + i32::from(unsafe { sc.add(2 * c + 1).read() }) * dhi;
            }
            acc = (dw * dx).mul_add(sumd as f32, acc);
            acc = (dmin * dx).mul_add(-(summ as f32), acc);
        }
        unsafe { y.add(r).write(acc) };
    }
}
```

- [ ] **Step 5: Run tests, record observed error**

```bash
cargo nextest run -p inferno-kernels
cargo nextest run -p inferno-kernels --run-ignored all -E 'test(observed_error_q4_k)' --no-capture
```

Expected: all PASS. Tune `gemv_rel_tol(Q4_K)` from the printed observation exactly as in Task 5 Step 4.

- [ ] **Step 6: Lint, fuzz (formats was touched in Step 1 — already fuzzed there; re-run only if quant.rs changed again), and commit**

```bash
mise run lint && mise run test
git add -A && git commit -m "feat(kernels): Q4_K rs8 GEMV with pack-time scale decoding"
```

---

### Task 7: Kernel registry — validated safe wrappers + `kernels_for`

**Files:**
- Create: `crates/inferno-kernels/src/registry.rs` (add `pub mod registry;` and `pub use registry::{KernelSet, kernels_for, reference_kernels};` to lib.rs)

**Interfaces:**
- Consumes: everything from Tasks 3–6; `inferno_target::Isa`.
- Produces (used by Task 8's benches and M3's planner):
  - `pub struct KernelSet { pub dtype: DType, pub isa: KernelIsa, pub layout: &'static str, .. }` with methods `pack(&self, bytes, rows, k) -> Result<AlignedBuf>`, `quantize_row(&self, x: &[f32]) -> Result<Vec<u8>>`, `gemv(&self, y: &mut [f32], xq: &[u8], w: &AlignedBuf, rows, k, row_start, row_end) -> Result<()>`, `packed_len(&self, rows, k) -> usize`, `act_len(&self, k) -> usize`
  - `pub fn kernels_for(dtype: &DType, isa: Isa) -> Option<KernelSet>` — SIMD set; refuses when the CPU lacks the features (the only place runtime feature detection happens)
  - `pub fn reference_kernels(dtype: &DType) -> Option<KernelSet>` — scalar set, always available

- [ ] **Step 1: Write failing tests** (`#[cfg(test)]` at the bottom of `src/registry.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelIsa;
    use inferno_formats::{DType, quant};
    use inferno_target::Isa;

    fn pseudo(mut seed: u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    fn selection_rules() {
        for dtype in [DType::F32, DType::Q8_0, DType::Q4_K] {
            assert!(reference_kernels(&dtype).is_some(), "{dtype:?}");
            if KernelIsa::Avx2.available() {
                let s = kernels_for(&dtype, Isa::X86_64v3).unwrap();
                assert_eq!(s.isa, KernelIsa::Avx2);
                assert_eq!(s.layout, "rs8");
                // v4 CPUs run v3 kernels.
                assert!(kernels_for(&dtype, Isa::X86_64v4).is_some());
            }
        }
        for dtype in [DType::F16, DType::BF16, DType::Unsupported("x".into())] {
            assert!(reference_kernels(&dtype).is_none(), "{dtype:?}");
            assert!(kernels_for(&dtype, Isa::X86_64v3).is_none(), "{dtype:?}");
        }
    }

    #[test]
    fn f32_quantize_row_is_le_bytes() {
        let s = reference_kernels(&DType::F32).unwrap();
        let x = [1.5f32, -2.0];
        let b = s.quantize_row(&x).unwrap();
        assert_eq!(b, [1.5f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat());
    }

    #[test]
    fn end_to_end_matches_direct_symbols() {
        let (rows, k) = (10usize, 64usize);
        let vals = pseudo(1, rows * k);
        let file = quant::pack(&DType::Q8_0, &vals).unwrap();
        let x = pseudo(2, k);
        let s = reference_kernels(&DType::Q8_0).unwrap();
        let w = s.pack(&file, rows, k).unwrap();
        let xq = s.quantize_row(&x).unwrap();
        let mut y = vec![f32::NAN; rows];
        s.gemv(&mut y, &xq, &w, rows, k, 0, rows).unwrap();
        let mut direct = vec![f32::NAN; rows];
        // SAFETY: same validated inputs as the wrapper call above.
        unsafe {
            crate::inferno_gemv_q8_0_rs8_scalar(
                direct.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, 0, rows,
            );
        }
        for (a, b) in y.iter().zip(&direct) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn gemv_wrapper_validates_everything() {
        let (rows, k) = (4usize, 32usize);
        let vals = pseudo(3, rows * k);
        let file = quant::pack(&DType::Q8_0, &vals).unwrap();
        let s = reference_kernels(&DType::Q8_0).unwrap();
        let w = s.pack(&file, rows, k).unwrap();
        let x = pseudo(4, k);
        let xq = s.quantize_row(&x).unwrap();
        let mut y = vec![0f32; rows];
        // Good call passes.
        s.gemv(&mut y, &xq, &w, rows, k, 0, rows).unwrap();
        // y too short.
        assert!(s.gemv(&mut y[..3], &xq, &w, rows, k, 0, rows).is_err());
        // Inverted / out-of-bounds row range.
        assert!(s.gemv(&mut y, &xq, &w, rows, k, 3, 2).is_err());
        assert!(s.gemv(&mut y, &xq, &w, rows, k, 0, rows + 1).is_err());
        // Wrong activation buffer length.
        assert!(s.gemv(&mut y, &xq[..xq.len() - 1], &w, rows, k, 0, rows).is_err());
        // k not matching the packed image.
        assert!(s.gemv(&mut y, &xq, &w, rows, 64, 0, rows).is_err());
        // k not a block multiple.
        assert!(s.gemv(&mut y, &xq, &w, rows, 33, 0, rows).is_err());
        // quantize_row validates too.
        assert!(s.quantize_row(&pseudo(5, 31)).is_err());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p inferno-kernels registry`
Expected: compile error.

- [ ] **Step 3: Implement** (top of `src/registry.rs`)

```rust
//! Kernel selection + the validated safe wrappers every non-codegen caller
//! uses. The raw `extern "C"` symbols stay unchecked (M3 codegen guarantees
//! their contracts by construction); tests, benches, and the M3 planner go
//! through `KernelSet`, which validates lengths, block multiples, and row
//! ranges. This is also the single place runtime CPU-feature detection
//! happens: `kernels_for` refuses to hand out kernels the CPU can't run.

use inferno_formats::DType;
use inferno_target::Isa;

use crate::{AlignedBuf, KernelError, KernelIsa, Result, act, f32k, q4_k, q8_0};

type PackFn = fn(&[u8], usize, usize) -> Result<AlignedBuf>;
type PackedLenFn = fn(usize, usize) -> usize;
type ActLenFn = fn(usize) -> usize;
type QuantFn = unsafe extern "C" fn(*const f32, *mut u8, usize);
type GemvFn = unsafe extern "C" fn(*mut f32, *const u8, *const u8, usize, usize, usize);

pub struct KernelSet {
    pub dtype: DType,
    pub isa: KernelIsa,
    /// Packed-layout identifier; part of the symbol names ("rs8").
    pub layout: &'static str,
    wblock: usize,
    pack: PackFn,
    packed_len: PackedLenFn,
    act_len: ActLenFn,
    quantize: Option<QuantFn>, // None: activations are raw f32 LE bytes
    gemv: GemvFn,
}

impl KernelSet {
    pub fn packed_len(&self, rows: usize, k: usize) -> usize {
        (self.packed_len)(rows, k)
    }

    pub fn act_len(&self, k: usize) -> usize {
        (self.act_len)(k)
    }

    pub fn pack(&self, bytes: &[u8], rows: usize, k: usize) -> Result<AlignedBuf> {
        (self.pack)(bytes, rows, k)
    }

    pub fn quantize_row(&self, x: &[f32]) -> Result<Vec<u8>> {
        if x.is_empty() || x.len() % self.wblock != 0 {
            return Err(KernelError::BadK { k: x.len(), block: self.wblock });
        }
        match self.quantize {
            Some(f) => {
                let mut out = vec![0u8; (self.act_len)(x.len())];
                // SAFETY: x/out lengths validated against the symbol contract;
                // SIMD sets exist only when the CPU supports them.
                unsafe { f(x.as_ptr(), out.as_mut_ptr(), x.len()) };
                Ok(out)
            }
            None => Ok(x.iter().flat_map(|v| v.to_le_bytes()).collect()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemv(
        &self,
        y: &mut [f32],
        xq: &[u8],
        w: &AlignedBuf,
        rows: usize,
        k: usize,
        row_start: usize,
        row_end: usize,
    ) -> Result<()> {
        if k == 0 || k % self.wblock != 0 {
            return Err(KernelError::BadK { k, block: self.wblock });
        }
        if y.len() != rows {
            return Err(KernelError::SizeMismatch {
                what: "output rows (f32 count)",
                got: y.len(),
                expected: rows,
            });
        }
        if row_start > row_end || row_end > rows {
            return Err(KernelError::BadRowRange { row_start, row_end, rows });
        }
        if xq.len() != (self.act_len)(k) {
            return Err(KernelError::SizeMismatch {
                what: "activation buffer bytes",
                got: xq.len(),
                expected: (self.act_len)(k),
            });
        }
        if w.len() != (self.packed_len)(rows, k) {
            return Err(KernelError::SizeMismatch {
                what: "packed weight bytes",
                got: w.len(),
                expected: (self.packed_len)(rows, k),
            });
        }
        // SAFETY: every pointer/length/alignment precondition of the symbol
        // was validated above; AlignedBuf guarantees 32-byte alignment.
        unsafe { (self.gemv)(y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, row_start, row_end) };
        Ok(())
    }
}

fn set(dtype: &DType, isa: KernelIsa) -> Option<KernelSet> {
    let s = match dtype {
        DType::F32 => KernelSet {
            dtype: DType::F32,
            isa,
            layout: "rs8",
            wblock: 1,
            pack: f32k::pack_f32_rs8,
            packed_len: f32k::packed_len_f32_rs8,
            act_len: |k| k * 4,
            quantize: None,
            gemv: match isa {
                KernelIsa::Scalar => f32k::inferno_gemv_f32_rs8_scalar,
                KernelIsa::Avx2 => f32k::inferno_gemv_f32_rs8_avx2,
            },
        },
        DType::Q8_0 => KernelSet {
            dtype: DType::Q8_0,
            isa,
            layout: "rs8",
            wblock: act::Q8A_BLOCK,
            pack: q8_0::pack_q8_0_rs8,
            packed_len: q8_0::packed_len_q8_0_rs8,
            act_len: act::q8a_len,
            quantize: Some(match isa {
                KernelIsa::Scalar => act::inferno_quantize_row_q8a_scalar,
                KernelIsa::Avx2 => act::inferno_quantize_row_q8a_avx2,
            }),
            gemv: match isa {
                KernelIsa::Scalar => q8_0::inferno_gemv_q8_0_rs8_scalar,
                KernelIsa::Avx2 => q8_0::inferno_gemv_q8_0_rs8_avx2,
            },
        },
        DType::Q4_K => KernelSet {
            dtype: DType::Q4_K,
            isa,
            layout: "rs8",
            wblock: act::Q8K_BLOCK,
            pack: q4_k::pack_q4_k_rs8,
            packed_len: q4_k::packed_len_q4_k_rs8,
            act_len: act::q8k_len,
            quantize: Some(match isa {
                KernelIsa::Scalar => act::inferno_quantize_row_q8k_scalar,
                KernelIsa::Avx2 => act::inferno_quantize_row_q8k_avx2,
            }),
            gemv: match isa {
                KernelIsa::Scalar => q4_k::inferno_gemv_q4_k_rs8_scalar,
                KernelIsa::Avx2 => q4_k::inferno_gemv_q4_k_rs8_avx2,
            },
        },
        DType::F16 | DType::BF16 | DType::Unsupported(_) => return None,
    };
    Some(s)
}

/// The SIMD kernel set for a target ISA level, or None if this dtype has no
/// kernels or the *running* CPU can't execute them (spec: the registry
/// refuses; scalar fallbacks come from [`reference_kernels`]).
pub fn kernels_for(dtype: &DType, isa: Isa) -> Option<KernelSet> {
    let kisa = match isa {
        // v4 ⊇ v3: no v4-specific kernels exist in M2, v4 CPUs run the AVX2 set.
        Isa::X86_64v3 | Isa::X86_64v4 => KernelIsa::Avx2,
    };
    if !kisa.available() {
        return None;
    }
    set(dtype, kisa)
}

/// Scalar kernels — always runnable, the portable fallback and debug aid.
pub fn reference_kernels(dtype: &DType) -> Option<KernelSet> {
    set(dtype, KernelIsa::Scalar)
}
```

This requires the Task 3–6 modules to expose their symbols as module items (they do — the `pub use` re-exports at crate root are additive). The `act` quantize symbols must be reachable as `act::inferno_quantize_row_*` — they are declared in `act.rs`.

- [ ] **Step 4: Run tests**

Run: `cargo nextest run -p inferno-kernels`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
mise run lint && mise run test
git add -A && git commit -m "feat(kernels): kernel registry with validated safe wrappers"
```

---

### Task 8: Criterion benches + pinned-ggml comparison

**Files:**
- Modify: `crates/inferno-kernels/benches/gemv.rs` (replace the Task 3 placeholder)
- Modify: `devenv.nix` (export `INFERNO_GGML_CPU_LIB`)
- Modify: `mise.toml` (add `bench-kernels` task)

**Interfaces:**
- Consumes: `registry::{kernels_for, reference_kernels}`, `inferno_target::Isa`, `inferno_formats::quant::f32_to_f16`.
- Produces: `mise run bench-kernels`; criterion groups `gemv/F32`, `gemv/Q8_0`, `gemv/Q4_K` with series `inferno-scalar`, `inferno-avx2`, `ggml` (feature-gated). Throughput reported as bytes of weight data streamed — GEMV is memory-bound, that is the honest metric (spec §Benchmarks).

- [ ] **Step 1: devenv + mise wiring**

In `devenv.nix`, after the `packages` list:

```nix
  # ggml CPU backend for `mise run bench-kernels` (--features ggml-compare).
  # haswell = AVX2+FMA — the same ISA class as inferno's M2 kernels, so the
  # comparison is apples-to-apples. The per-arch backends live under bin/.
  env.INFERNO_GGML_CPU_LIB = "${pkgs.llama-cpp}/bin/libggml-cpu-haswell.so";
```

In `mise.toml`:

```toml
[tasks.bench-kernels]
description = "Kernel µbenches vs pinned ggml (run inside devenv shell, on quiet hardware)"
run = "cargo bench -p inferno-kernels --features ggml-compare"
```

Verify the library exists and exports the symbols:

```bash
devenv shell -- bash -c 'nm -D "$INFERNO_GGML_CPU_LIB" | grep -E " T (ggml_vec_dot_q4_K_q8_K|ggml_vec_dot_q8_0_q8_0|quantize_row_q8_K|quantize_row_q8_0|ggml_vec_dot_f32)$"'
```

Expected: all five symbols listed.

- [ ] **Step 2: Write the bench** (`benches/gemv.rs`, full contents)

```rust
//! GEMV throughput on real Llama-family shapes (Qwen2.5-0.5B: 896/4864/151936;
//! Llama-3-8B: 4096/14336/128256), side by side with the devenv-pinned
//! llama.cpp CPU kernels when built with --features ggml-compare.
//! Run via `mise run bench-kernels` inside the devenv shell on quiet hardware;
//! numbers from shared CI runners are noise (spec §Benchmarks).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use inferno_formats::DType;
use inferno_formats::quant::f32_to_f16;
use inferno_kernels::registry::{KernelSet, kernels_for, reference_kernels};
use inferno_target::Isa;

const SHAPES_F32: &[(usize, usize)] = &[(4096, 4096)];
const SHAPES_Q8_0: &[(usize, usize)] =
    &[(896, 896), (4864, 896), (896, 4864), (151936, 896), (4096, 4096), (14336, 4096)];
const SHAPES_Q4_K: &[(usize, usize)] =
    &[(4096, 4096), (14336, 4096), (4096, 14336), (128256, 4096)];

fn pseudo_f32(mut seed: u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn pseudo_bytes(mut seed: u64, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u8
        })
        .collect()
}

/// Plausible file-order weight bytes without quantizing gigabytes of f32:
/// random quant payloads with small fixed scales (perf is value-independent).
fn gen_weights(dtype: &DType, rows: usize, k: usize) -> Vec<u8> {
    match dtype {
        DType::F32 => pseudo_f32(1, rows * k).iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::Q8_0 => {
            let nb = rows * k / 32;
            let mut out = Vec::with_capacity(nb * 34);
            let d = f32_to_f16(0.05).to_le_bytes();
            let qs = pseudo_bytes(2, nb * 32);
            for b in 0..nb {
                out.extend_from_slice(&d);
                out.extend_from_slice(&qs[b * 32..(b + 1) * 32]);
            }
            out
        }
        DType::Q4_K => {
            let nsb = rows * k / 256;
            let mut out = Vec::with_capacity(nsb * 144);
            let d = f32_to_f16(0.05).to_le_bytes();
            let dmin = f32_to_f16(0.02).to_le_bytes();
            let payload = pseudo_bytes(3, nsb * 140);
            for b in 0..nsb {
                out.extend_from_slice(&d);
                out.extend_from_slice(&dmin);
                out.extend_from_slice(&payload[b * 140..(b + 1) * 140]);
            }
            out
        }
        _ => unreachable!("no benches for {dtype:?}"),
    }
}

fn sets_for(dtype: &DType) -> Vec<(&'static str, KernelSet)> {
    let mut v = vec![("inferno-scalar", reference_kernels(dtype).unwrap())];
    if let Some(s) = kernels_for(dtype, Isa::X86_64v3) {
        v.push(("inferno-avx2", s));
    }
    v
}

fn bench_dtype(c: &mut Criterion, dtype: DType, shapes: &[(usize, usize)]) {
    let mut group = c.benchmark_group(format!("gemv/{dtype:?}"));
    group.sample_size(20);
    for &(rows, k) in shapes {
        let file = gen_weights(&dtype, rows, k);
        let x = pseudo_f32(42, k);
        for (name, set) in sets_for(&dtype) {
            let w = set.pack(&file, rows, k).unwrap();
            let xq = set.quantize_row(&x).unwrap();
            let mut y = vec![0f32; rows];
            group.throughput(Throughput::Bytes(w.len() as u64));
            group.bench_function(BenchmarkId::new(name, format!("{rows}x{k}")), |b| {
                b.iter(|| set.gemv(&mut y, &xq, &w, rows, k, 0, rows).unwrap())
            });
        }
        #[cfg(feature = "ggml-compare")]
        ggml::bench(&mut group, &dtype, &file, &x, rows, k);
    }
    group.finish();
}

#[cfg(feature = "ggml-compare")]
mod ggml {
    //! dlopen the pinned ggml CPU backend and drive its row-dot kernels on
    //! identical data. ggml consumes file-order weights directly (no repack),
    //! so its throughput basis is the file byte count.
    use std::ffi::c_void;
    use std::sync::OnceLock;

    use criterion::{BenchmarkId, Throughput, measurement::WallTime};
    use inferno_formats::DType;

    // void ggml_vec_dot_*(int n, float *s, size_t bs, const void *x, size_t bx,
    //                     const void *y, size_t by, int nrc)
    type VecDot =
        unsafe extern "C" fn(i32, *mut f32, usize, *const c_void, usize, *const c_void, usize, i32);
    // void quantize_row_*(const float *x, void *y, int64_t k)
    type QuantRow = unsafe extern "C" fn(*const f32, *mut c_void, i64);

    fn lib() -> &'static libloading::Library {
        static LIB: OnceLock<libloading::Library> = OnceLock::new();
        LIB.get_or_init(|| {
            let path = std::env::var("INFERNO_GGML_CPU_LIB")
                .expect("INFERNO_GGML_CPU_LIB not set — run inside `devenv shell`");
            // SAFETY: loading the version-pinned ggml CPU backend for benching.
            unsafe { libloading::Library::new(&path) }
                .unwrap_or_else(|e| panic!("cannot load {path}: {e}"))
        })
    }

    pub fn bench(
        group: &mut criterion::BenchmarkGroup<'_, WallTime>,
        dtype: &DType,
        file: &[u8],
        x: &[f32],
        rows: usize,
        k: usize,
    ) {
        let (dot_sym, quant_sym, act_bytes, row_bytes): (&[u8], Option<&[u8]>, usize, usize) =
            match dtype {
                DType::F32 => (b"ggml_vec_dot_f32", None, k * 4, k * 4),
                DType::Q8_0 => {
                    (b"ggml_vec_dot_q8_0_q8_0", Some(b"quantize_row_q8_0"), k / 32 * 34, k / 32 * 34)
                }
                DType::Q4_K => {
                    (b"ggml_vec_dot_q4_K_q8_K", Some(b"quantize_row_q8_K"), k / 256 * 292, k / 256 * 144)
                }
                _ => return,
            };
        // SAFETY: signatures match the pinned ggml's headers.
        let dot: libloading::Symbol<'_, VecDot> = unsafe { lib().get(dot_sym).unwrap() };
        let xq: Vec<u8> = match quant_sym {
            Some(qs) => {
                let quant: libloading::Symbol<'_, QuantRow> = unsafe { lib().get(qs).unwrap() };
                let mut buf = vec![0u8; act_bytes];
                // SAFETY: x has k f32; buf sized per ggml's activation block.
                unsafe { quant(x.as_ptr(), buf.as_mut_ptr().cast(), k as i64) };
                buf
            }
            None => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        let mut y = vec![0f32; rows];
        group.throughput(Throughput::Bytes((rows * row_bytes) as u64));
        group.bench_function(BenchmarkId::new("ggml", format!("{rows}x{k}")), |b| {
            b.iter(|| {
                for r in 0..rows {
                    // SAFETY: file holds rows*row_bytes; xq is ggml's own
                    // activation layout for k; one row per call (nrc=1).
                    unsafe {
                        dot(
                            k as i32,
                            y.as_mut_ptr().add(r),
                            0,
                            file.as_ptr().add(r * row_bytes).cast(),
                            0,
                            xq.as_ptr().cast(),
                            0,
                            1,
                        )
                    };
                }
            })
        });
    }
}

fn benches(c: &mut Criterion) {
    bench_dtype(c, DType::F32, SHAPES_F32);
    bench_dtype(c, DType::Q8_0, SHAPES_Q8_0);
    bench_dtype(c, DType::Q4_K, SHAPES_Q4_K);
}

criterion_group!(gemv, benches);
criterion_main!(gemv);
```

- [ ] **Step 3: Smoke-test both configurations**

```bash
cargo bench -p inferno-kernels -- --test                     # no FFI, quick single-pass
devenv shell -- cargo bench -p inferno-kernels --features ggml-compare -- --test
mise run lint                                                # bench target compiles clean both ways
```

Expected: every bench runs once, no panics, ggml symbols resolve inside the shell.

- [ ] **Step 4: Full run on quiet hardware, save output**

```bash
devenv shell -- mise run bench-kernels 2>&1 | tee /tmp/bench-kernels-m2.txt
```

Expected: GB/s figures for `inferno-scalar`, `inferno-avx2`, `ggml` per shape. **Parity check (spec exit criterion):** `inferno-avx2` should be at or approaching `ggml` on the Q4_K and Q8_0 shapes. If it falls well short (< ~0.7×), profile before proceeding — likely suspects are the per-32-element `hsum` (batch the horizontal reduction across sub-blocks) or per-row group re-reads (process whole strips per pass). Tuning may land as a follow-up commit in this task; do not skip recording whatever the numbers are.

- [ ] **Step 5: Commit**

```bash
mise run lint && mise run test
git add -A && git commit -m "feat(kernels): criterion GEMV benches vs pinned ggml CPU kernels"
```

---

### Task 9: Docs, nightly CI, spec amendments, recorded data

**Files:**
- Modify: `README.md`, `ARCHITECTURE.md`, `AGENTS.md`
- Modify: `.github/workflows/nightly.yml`
- Modify: `docs/superpowers/specs/2026-07-05-m2-targets-kernels-design.md`

- [ ] **Step 1: README status**

Change the status line to:

```markdown
**Status:** pre-release, milestone M2 (targets + AVX2 quantized GEMV kernels).
```

- [ ] **Step 2: ARCHITECTURE.md**

Move `inferno-target` and `inferno-kernels` from the "Planned" list to "Present (M0–M2)", keeping their one-line descriptions, and append to "Boundary rules that aren't visible in the code":

```markdown
- Activation-side quant formats (q8a/q8k) are kernel implementation details:
  they live in `inferno-kernels` and never appear in `inferno_formats::DType`.
- Kernel ISA variants are bit-identical by construction (exact integer block
  dots, fixed f32 combine order); the rig asserts exact equality, so any
  "harmless" reassociation in a kernel is a contract break, not an optimization.
- Kernels are single-threaded and row-range partitioned; parallelism is the
  caller's job (M3 splits `row_start..row_end` across threads).
```

- [ ] **Step 3: AGENTS.md** — add to "Non-obvious constraints":

```markdown
- **`inferno-kernels` is the only crate allowed `unsafe`** (intrinsics + the
  C ABI); it opts out of the workspace lint deliberately. Scalar and SIMD
  kernel variants must stay bit-identical — the rig asserts exact equality.
- **Kernel perf numbers come only from `mise run bench-kernels`** inside the
  devenv shell on quiet hardware; CI runners are noise. Record data points in
  the M2 spec's amendments section.
- **`gemv_rel_tol`** follows the same rule as `LOGIT_TIE_EPSILON`: tuned
  against observed error distributions (the rig's ignored `observed_error_*`
  diagnostics), never to make a red test green.
```

- [ ] **Step 4: nightly.yml** — add the test-full job (enlarged proptest runs; the machine-gated detect==profile test passes vacuously here):

```yaml
  # Full suite incl. ignored/slow tests, with enlarged property-test runs.
  test-full:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jdx/mise-action@v2
      - uses: Swatinem/rust-cache@v2
      - run: mise run test-full
        env:
          PROPTEST_CASES: "1024"
```

- [ ] **Step 5: Spec amendments** — append to `docs/superpowers/specs/2026-07-05-m2-targets-kernels-design.md`:

```markdown
## Amendments (2026-07-05, during planning and implementation)

- **ggml comparison mechanism:** the pinned llama.cpp exports the needed
  kernels from its per-arch CPU backends (`bin/libggml-cpu-<arch>.so`), not
  from `libggml.so`. The bench `dlopen`s `libggml-cpu-haswell.so` (AVX2+FMA)
  via `$INFERNO_GGML_CPU_LIB` instead of link-time FFI; the `ggml_mul_mat`
  fallback was unnecessary. Verified: all five symbols export.
- **detect==profile placement:** GitHub runners are not the dev machine, so
  the equivalence test is gated on `INFERNO_EXPECT_PROFILE` (vacuous when
  unset) rather than nightly-scheduled; nightly CI instead runs the full
  suite with `PROPTEST_CASES=1024`.
- **`pack_*` is safe Rust, not a C symbol** — its only caller (M3 planner)
  is Rust. `quantize_row_*`/`gemv_*` remain `extern "C"`.
- **ISA variants are bit-identical**, not ~1e-6-close: integer block dots are
  exact and the f32 combine order is fixed. The rig asserts exact equality.
- **`pack_q8_0_rs8` clamps weight bytes −128 → −127** so the AVX2 sign-trick
  stays exact on hostile files (ggml's quantizer never emits −128).
- **Benches report GB/s only** (criterion `Throughput::Bytes` on the weight
  stream — the metric that matters for a memory-bound GEMV). GFLOPS is
  derivable as `2·rows·k / time` and was dropped rather than double-reported.

### First bench data points (dev Ryzen 9 3900, 2026-07-05)

<!-- paste the gemv/{Q4_K,Q8_0,F32} GB/s table from /tmp/bench-kernels-m2.txt,
     with the inferno-avx2 : ggml ratio per shape -->
```

Replace the HTML comment with the actual recorded numbers from Task 8 Step 4.

- [ ] **Step 6: Full verification and commit**

```bash
mise run lint
mise run test
PROPTEST_CASES=1024 mise run test-full   # includes ignored diagnostics; slow, one-off
INFERNO_EXPECT_PROFILE=ryzen-3900 cargo nextest run -p inferno-target detect_matches
git add -A && git commit -m "docs: M2 integration — architecture, agents, nightly test-full, bench data"
```

Expected: all green. This closes the plan; the milestone-exit checklist in the spec should now read as done: detect==profile passes, all three dtypes pass the rig in the blocking tier, and bench data is recorded.



