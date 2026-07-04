# Inferno M0 — Skeleton + Formats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the inferno workspace — pinned dev environment (mise + devenv.nix with LLVM/llama.cpp), repo front door, tiered CI with scanners, threat model — and ship `inferno-formats` parsing GGUF and MLX/safetensors files into a shared `ModelDesc`, exercised by an `inferno inspect` CLI and fuzz targets.

**Architecture:** Cargo workspace with two M0 crates: `inferno-formats` (library: format-agnostic `ModelDesc` + GGUF and safetensors/MLX parsers, no `unsafe`, bounded reads) and `cli` (the `inferno` binary with an `inspect` subcommand). Parsers work over `std::io::Read` so the same code path serves files, tests, and fuzz targets; tensor *data* is never read in M0, only headers/metadata. Spec: `docs/superpowers/specs/2026-07-04-inferno-v1-design.md`.

**Tech Stack:** Rust (mise-pinned stable, edition 2024), thiserror, serde/serde_json, clap, insta (snapshots), assert_cmd (CLI tests), cargo-fuzz/libfuzzer-sys (nightly, CI), cargo-nextest, cargo-audit, cargo-deny, gitleaks, lefthook, devenv.nix (LLVM 18, llama.cpp), GitHub Actions.

## Global Constraints

- **No `unsafe` anywhere in `inferno-formats`** — enforced with `#![forbid(unsafe_code)]` (spec: Security / threat model).
- **Every mise tool entry is `--pin`ned to an exact version** — an unpinned entry is a reproducibility bug.
- **Native deps (LLVM, llama.cpp) live only in `devenv.nix`**, never installed ad hoc; LLVM major = 18 to match `inkwell`'s `llvm18-1` feature used in M3. Export `LLVM_SYS_180_PREFIX` in the devenv shell.
- **All parser reads are bounded**: limits in `crates/inferno-formats/src/limits.rs`; all arithmetic on attacker-controlled sizes uses `checked_add`/`checked_mul`.
- **`ModelDesc` is format-agnostic**: nothing outside `inferno-formats` may learn which file format a model came from.
- **Tensor shapes are normalized to row-major, outermost dimension first** (GGUF stores fastest-varying first — reverse its dims on ingest).
- **README/docs reference mise task *names*, never re-spell commands** (single source of truth).
- **Blocking CI wall-clock budget: ≤ 5 minutes.** Fuzz, mutants, semgrep, real-model tests, onboarding job are nightly.
- Commit after every task (steps below include the commits).

## File Structure

```
mise.toml                      # pinned tools + named tasks (Task 1)
rustfmt.toml, deny.toml, lefthook.yml, .gitignore   (Task 1)
Cargo.toml                     # workspace root (Task 1)
devenv.yaml, devenv.nix, devenv.lock                (Task 2)
crates/inferno-formats/
  Cargo.toml
  src/lib.rs                   # load_desc() entry, module wiring (Tasks 3,8)
  src/error.rs                 # FormatError (Task 3)
  src/limits.rs                # parser limits (Task 3)
  src/desc.rs                  # ModelDesc, TensorDesc, DType, HyperParams, Architecture (Task 3)
  src/read.rs                  # bounded primitive readers over io::Read (Task 4)
  src/gguf/mod.rs              # GGUF header parse → ModelDesc (Task 5)
  src/gguf/value.rs            # GGUF metadata value tree (Task 4)
  src/safetensors.rs           # safetensors header parse (Task 6)
  src/mlx.rs                   # MLX dir/config.json loader (Task 7)
  src/fixtures.rs              # tiny in-memory model builders (Task 5; used by tests, fuzz seeds, CLI tests, M1)
  examples/gen_fixtures.rs     # writes committed fixture files (Task 8)
  tests/fixtures/…             # committed tiny.gguf + mlx/ dir (Task 8)
  tests/snapshot_desc.rs       # insta snapshots of parsed fixtures (Task 8)
fuzz/Cargo.toml
fuzz/fuzz_targets/{gguf_parse,safetensors_parse}.rs  (Task 10)
cli/Cargo.toml
cli/src/main.rs                # clap entry (Task 9)
cli/src/inspect.rs             # inspect rendering (Task 9)
cli/tests/inspect.rs           # assert_cmd + insta (Task 9)
README.md, AGENTS.md, CLAUDE.md, ARCHITECTURE.md, docs/threat-model.md (Task 11)
.github/workflows/ci.yml, .github/workflows/nightly.yml              (Task 12)
```

---

### Task 1: Toolchain pins + workspace skeleton

**Files:**
- Create: `mise.toml`, `Cargo.toml`, `rustfmt.toml`, `deny.toml`, `lefthook.yml`
- Create: `crates/inferno-formats/Cargo.toml`, `crates/inferno-formats/src/lib.rs`
- Create: `cli/Cargo.toml`, `cli/src/main.rs`
- Modify: `.gitignore`

**Interfaces:**
- Produces: mise tasks `test`, `test-full`, `lint`, `fmt`, `audit`, `fuzz` — every later task and CI invoke these names; workspace crates `inferno-formats` (lib) and `inferno` (bin, in `cli/`).

- [ ] **Step 1: Pin tools with mise**

Run (each writes an exact resolved version into `mise.toml` — that exactness is the requirement; the versions shown later are what resolution looked like at planning time):

```bash
cd /workspace
mise use --pin rust@latest
mise use --pin "cargo:cargo-nextest@latest" "cargo:cargo-audit@latest" "cargo:cargo-deny@latest" "cargo:cargo-insta@latest"
mise use --pin gitleaks@latest lefthook@latest
mise install
```

Expected: `mise.toml` gains a `[tools]` section with exact versions (e.g. `rust = "1.93.0"`, not `"latest"`); `cargo --version` then works. If any `cargo:` backend entry is slow to compile, that is expected on first install (they compile from source and are cached after).

- [ ] **Step 2: Add named tasks and env to mise.toml**

Append to `mise.toml` (keep the generated `[tools]` block above it):

```toml
[env]
CARGO_TERM_COLOR = "always"

[tasks.test]
description = "Blocking-tier tests (fast; what PR CI runs)"
run = "cargo nextest run --workspace"

[tasks.test-full]
description = "Full test suite incl. ignored/slow tests"
run = "cargo nextest run --workspace --run-ignored all"

[tasks.lint]
description = "Format check + clippy (deny warnings)"
run = [
  "cargo fmt --all --check",
  "cargo clippy --workspace --all-targets -- -D warnings",
]

[tasks.fmt]
description = "Auto-format"
run = "cargo fmt --all"

[tasks.audit]
description = "Supply-chain gate: RustSec advisories + cargo-deny policy"
run = ["cargo audit", "cargo deny check advisories bans sources licenses"]

[tasks.fuzz]
description = "Run one fuzz target briefly (nightly rustc): mise run fuzz -- gguf_parse"
run = "mise exec rust@nightly -- cargo fuzz run --fuzz-dir fuzz {{arg(name='target')}} -- -max_total_time=60"
```

- [ ] **Step 3: Create the workspace**

`Cargo.toml` (root):

```toml
[workspace]
resolver = "3"
members = ["crates/inferno-formats", "cli"]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "Apache-2.0"
repository = "https://github.com/rahulmutt/inferno"

[workspace.dependencies]
inferno-formats = { path = "crates/inferno-formats" }
thiserror = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
insta = { version = "1", features = ["yaml"] }
assert_cmd = "2"

[workspace.lints.rust]
unsafe_code = "deny"
```

`crates/inferno-formats/Cargo.toml`:

```toml
[package]
name = "inferno-formats"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
thiserror.workspace = true
serde.workspace = true
serde_json.workspace = true

[dev-dependencies]
insta.workspace = true

[lints]
workspace = true
```

`crates/inferno-formats/src/lib.rs`:

```rust
//! Model-file parsing: GGUF and MLX/safetensors → format-agnostic [`ModelDesc`].
//!
//! Parsers treat every input byte as untrusted (see docs/threat-model.md):
//! bounded reads, checked arithmetic, and no `unsafe` anywhere in this crate.
#![forbid(unsafe_code)]
```

`cli/Cargo.toml`:

```toml
[package]
name = "inferno"
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "inferno"
path = "src/main.rs"

[dependencies]
inferno-formats.workspace = true
clap.workspace = true

[dev-dependencies]
assert_cmd.workspace = true
insta.workspace = true

[lints]
workspace = true
```

`cli/src/main.rs` (placeholder that compiles; Task 9 replaces it):

```rust
fn main() {
    println!("inferno: no subcommands yet (M0 in progress)");
}
```

`rustfmt.toml`:

```toml
edition = "2024"
```

`deny.toml`:

```toml
[advisories]
version = 2

[licenses]
version = 2
allow = ["Apache-2.0", "MIT", "Unicode-3.0", "BSD-3-Clause", "ISC", "Zlib"]

[bans]
multiple-versions = "warn"

[sources]
unknown-registry = "deny"
unknown-git = "deny"
```

`lefthook.yml`:

```yaml
pre-commit:
  parallel: true
  commands:
    gitleaks:
      run: gitleaks protect --staged --redact
    fmt:
      glob: "*.rs"
      run: cargo fmt --all --check
```

Append to `.gitignore`:

```
/target
.devenv*
.direnv
```

- [ ] **Step 4: Verify the skeleton**

```bash
mise run lint && mise run test && lefthook install
```

Expected: lint passes; nextest reports `0 tests run` (no tests yet) with exit 0; lefthook prints `sync hooks: ✔️`. If `cargo nextest` errors with "no tests to run", pass — that is exit 0 on current nextest; if your nextest version exits non-zero for zero tests, add `--no-tests=pass` to the `test` task and keep going.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: workspace skeleton, pinned toolchain, named tasks"
```

---

### Task 2: devenv.nix for native dependencies (LLVM + llama.cpp)

**Files:**
- Create: `devenv.yaml`, `devenv.nix` (and generated `devenv.lock`, `.envrc` if direnv chosen)

**Interfaces:**
- Produces: `devenv shell` exposing `llvm-config` (major 18), `LLVM_SYS_180_PREFIX`, and `llama-cli`/`llama-bench` on PATH. M3 (`inferno-codegen`) and M4 (bench protocol) consume these; nothing in M0 compiles against LLVM.

- [ ] **Step 1: Author devenv files**

`devenv.yaml`:

```yaml
inputs:
  nixpkgs:
    url: github:NixOS/nixpkgs/nixpkgs-unstable
```

`devenv.nix`:

```nix
# Native dependencies mise cannot supply (see AGENTS.md / developer-environment
# skill). Everything else — rust, cargo tools — is pinned in mise.toml.
{ pkgs, ... }:

{
  packages = [
    # LLVM for inferno-codegen (llvm-sys/inkwell). Major version MUST match
    # the inkwell feature flag in crates/inferno-codegen (llvm18-1).
    pkgs.llvmPackages_18.llvm.dev
    pkgs.libffi
    pkgs.libxml2
    pkgs.zlib
    # Pinned benchmark opponent for `mise run bench` (M4).
    pkgs.llama-cpp
  ];

  env.LLVM_SYS_180_PREFIX = "${pkgs.llvmPackages_18.llvm.dev}";

  enterShell = ''
    echo "inferno devenv: LLVM $(llvm-config --version)"
  '';
}
```

- [ ] **Step 2: Verify the shell provides the native deps**

```bash
cd /workspace && devenv shell -- bash -c 'llvm-config --version && echo "prefix=$LLVM_SYS_180_PREFIX" && command -v llama-cli'
```

Expected: prints `18.x.y`, a `/nix/store/...` prefix path, and a `/nix/store/.../bin/llama-cli` path. First run downloads/builds packages (minutes). If nixpkgs-unstable has moved past LLVM 18 as default, this still works — we reference `llvmPackages_18` explicitly. Commit the generated `devenv.lock`: it is the pin.

- [ ] **Step 3: Commit**

```bash
git add devenv.yaml devenv.nix devenv.lock && git commit -m "chore: devenv.nix native deps — LLVM 18 (for codegen), llama.cpp (bench opponent)"
```

---

### Task 3: ModelDesc core types, errors, limits

**Files:**
- Create: `crates/inferno-formats/src/desc.rs`, `src/error.rs`, `src/limits.rs`
- Modify: `crates/inferno-formats/src/lib.rs`

**Interfaces:**
- Produces (used by every later task):
  - `DType::{F32, F16, BF16, Q8_0, Q4_K, Unsupported(String)}`; `DType::byte_len(n_elems: u64) -> Option<u64>`
  - `Architecture::{Llama, Qwen2, Qwen3, Mistral, Unknown(String)}`; `Architecture::from_id(&str) -> Architecture`
  - `HyperParams { vocab_size, hidden_size, n_layers, n_heads, n_kv_heads, ffn_hidden_size: u64, rope_theta: f32, norm_eps: f32, context_length: u64 }`
  - `TensorDesc { name: String, shape: Vec<u64>, dtype: DType, file_index: u32, data_offset: u64, data_len: Option<u64> }`
  - `ModelDesc { architecture, name: Option<String>, hyperparams, tensors: Vec<TensorDesc>, weight_files: Vec<PathBuf>, data_section_offsets: Vec<u64> }`
  - `FormatError` (+ `pub type Result<T>`), limit constants in `limits`

- [ ] **Step 1: Write failing unit tests** (bottom of `desc.rs`, shown with the implementation below)

Tests to include:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_len_simple_dtypes() {
        assert_eq!(DType::F32.byte_len(10), Some(40));
        assert_eq!(DType::F16.byte_len(10), Some(20));
        assert_eq!(DType::BF16.byte_len(10), Some(20));
    }

    #[test]
    fn byte_len_block_quants() {
        // Q8_0: 34-byte blocks of 32 elements; Q4_K: 144-byte blocks of 256.
        assert_eq!(DType::Q8_0.byte_len(64), Some(68));
        assert_eq!(DType::Q4_K.byte_len(512), Some(288));
        // Not a multiple of the block size → not computable.
        assert_eq!(DType::Q8_0.byte_len(33), None);
        // Unsupported dtype → not computable.
        assert_eq!(DType::Unsupported("ggml:26".into()).byte_len(32), None);
    }

    #[test]
    fn byte_len_overflow_is_none() {
        assert_eq!(DType::F32.byte_len(u64::MAX), None);
    }

    #[test]
    fn architecture_from_id() {
        assert_eq!(Architecture::from_id("llama"), Architecture::Llama);
        assert_eq!(Architecture::from_id("qwen2"), Architecture::Qwen2);
        assert_eq!(
            Architecture::from_id("mamba"),
            Architecture::Unknown("mamba".into())
        );
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p inferno-formats`
Expected: compile FAILURE — `DType` not defined.

- [ ] **Step 3: Implement**

`crates/inferno-formats/src/desc.rs` (above the tests from Step 1):

```rust
use std::path::PathBuf;

use serde::Serialize;

/// Tensor element type. Quant formats are first-class dtypes (spec §Graph IR).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_K,
    /// Parsed but not supported by inferno v1 (e.g. "ggml:26", "U32").
    Unsupported(String),
}

impl DType {
    /// (bytes per block, elements per block), when the layout is known.
    fn block_layout(&self) -> Option<(u64, u64)> {
        match self {
            DType::F32 => Some((4, 1)),
            DType::F16 | DType::BF16 => Some((2, 1)),
            DType::Q8_0 => Some((34, 32)),
            DType::Q4_K => Some((144, 256)),
            DType::Unsupported(_) => None,
        }
    }

    /// Byte length of `n_elems` elements, if the dtype's layout is known and
    /// `n_elems` is block-aligned. Overflow-safe.
    pub fn byte_len(&self, n_elems: u64) -> Option<u64> {
        let (block_bytes, block_elems) = self.block_layout()?;
        if n_elems % block_elems != 0 {
            return None;
        }
        (n_elems / block_elems).checked_mul(block_bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum Architecture {
    Llama,
    Qwen2,
    Qwen3,
    Mistral,
    Unknown(String),
}

impl Architecture {
    pub fn from_id(id: &str) -> Self {
        match id {
            "llama" => Self::Llama,
            "qwen2" => Self::Qwen2,
            "qwen3" => Self::Qwen3,
            "mistral" => Self::Mistral,
            other => Self::Unknown(other.to_string()),
        }
    }
}

/// Llama-family transformer hyperparameters (spec §Graph IR).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HyperParams {
    pub vocab_size: u64,
    pub hidden_size: u64,
    pub n_layers: u64,
    pub n_heads: u64,
    pub n_kv_heads: u64,
    pub ffn_hidden_size: u64,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub context_length: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TensorDesc {
    pub name: String,
    /// Row-major, outermost dimension first (GGUF dims arrive reversed).
    pub shape: Vec<u64>,
    pub dtype: DType,
    /// Index into [`ModelDesc::weight_files`].
    pub file_index: u32,
    /// Byte offset within that file's data section.
    pub data_offset: u64,
    /// Byte length, when computable for the dtype.
    pub data_len: Option<u64>,
}

/// Format-agnostic model description. Downstream crates must not be able to
/// tell which file format this came from.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelDesc {
    pub architecture: Architecture,
    pub name: Option<String>,
    pub hyperparams: HyperParams,
    pub tensors: Vec<TensorDesc>,
    /// Files holding tensor data (absolute paths; not serialized — machine-specific).
    #[serde(skip)]
    pub weight_files: Vec<PathBuf>,
    /// Byte offset of the data section in each weight file (parallel array).
    pub data_section_offsets: Vec<u64>,
}
```

`crates/inferno-formats/src/error.rs`:

```rust
/// Errors from parsing model files. Every variant is a *rejection* of
/// untrusted input, not a panic — parsers must be total over arbitrary bytes.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a recognized model file: {0}")]
    UnknownFormat(String),
    #[error("bad magic bytes (expected {expected})")]
    BadMagic { expected: &'static str },
    #[error("unsupported gguf version {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),
    #[error("malformed {context}: {detail}")]
    Malformed {
        context: &'static str,
        detail: String,
    },
    #[error("{what} ({got}) exceeds limit ({limit})")]
    LimitExceeded {
        what: &'static str,
        got: u64,
        limit: u64,
    },
    #[error("missing required metadata: {0}")]
    MissingKey(String),
    #[error("invalid json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, FormatError>;
```

`crates/inferno-formats/src/limits.rs`:

```rust
//! Parser limits (spec §Security: allocation limits on untrusted input).
//! Sized generously above real models (vocab ~300k, merges ~500k) but far
//! below anything that could exhaust memory from a header alone.

pub const MAX_TENSORS: u64 = 65_536;
pub const MAX_KV_PAIRS: u64 = 65_536;
pub const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_ARRAY_ELEMS: u64 = 10_000_000;
pub const MAX_DIMS: u32 = 8;
pub const MAX_ARRAY_DEPTH: u32 = 4;
/// safetensors spec caps the JSON header at 100 MB.
pub const MAX_ST_HEADER_BYTES: u64 = 100 * 1024 * 1024;
```

Update `crates/inferno-formats/src/lib.rs`:

```rust
//! Model-file parsing: GGUF and MLX/safetensors → format-agnostic [`ModelDesc`].
//!
//! Parsers treat every input byte as untrusted (see docs/threat-model.md):
//! bounded reads, checked arithmetic, and no `unsafe` anywhere in this crate.
#![forbid(unsafe_code)]

mod desc;
mod error;
pub mod limits;

pub use desc::{Architecture, DType, HyperParams, ModelDesc, TensorDesc};
pub use error::{FormatError, Result};
```

- [ ] **Step 4: Run tests**

Run: `cargo nextest run -p inferno-formats`
Expected: 4 tests PASS.

- [ ] **Step 5: Lint + commit**

```bash
mise run lint && git add -A && git commit -m "feat(formats): ModelDesc core types, error taxonomy, parser limits"
```

---

### Task 4: Bounded readers + GGUF metadata values

**Files:**
- Create: `crates/inferno-formats/src/read.rs`, `crates/inferno-formats/src/gguf/value.rs`, `crates/inferno-formats/src/gguf/mod.rs` (module shell)
- Modify: `crates/inferno-formats/src/lib.rs`

**Interfaces:**
- Consumes: `FormatError`, `limits` (Task 3)
- Produces:
  - `read::{read_u8, read_u16, read_u32, read_u64, read_i8, read_i16, read_i32, read_i64, read_f32, read_f64, read_bool, read_string}` — all `pub(crate) fn(&mut impl Read, ...) -> Result<T>`, strings limited by `limits::MAX_STRING_BYTES`
  - `gguf::value::GgufValue` enum with `parse(&mut impl Read, type_id: u32, depth: u32) -> Result<GgufValue>` and accessors `as_u64() -> Option<u64>`, `as_f32() -> Option<f32>`, `as_str() -> Option<&str>`, `array_len() -> Option<u64>`

- [ ] **Step 1: Write failing tests** (in `gguf/value.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse_bytes(type_id: u32, bytes: &[u8]) -> crate::Result<GgufValue> {
        GgufValue::parse(&mut Cursor::new(bytes), type_id, 0)
    }

    #[test]
    fn scalar_values() {
        assert_eq!(parse_bytes(4, &7u32.to_le_bytes()).unwrap().as_u64(), Some(7));
        assert_eq!(parse_bytes(10, &9u64.to_le_bytes()).unwrap().as_u64(), Some(9));
        assert_eq!(parse_bytes(6, &1.5f32.to_le_bytes()).unwrap().as_f32(), Some(1.5));
        assert_eq!(parse_bytes(7, &[1]).unwrap(), GgufValue::Bool(true));
    }

    #[test]
    fn string_value() {
        // GGUF string: u64 LE length + UTF-8 bytes.
        let mut b = 5u64.to_le_bytes().to_vec();
        b.extend_from_slice(b"hello");
        assert_eq!(parse_bytes(8, &b).unwrap().as_str(), Some("hello"));
    }

    #[test]
    fn string_over_limit_rejected() {
        let b = (crate::limits::MAX_STRING_BYTES + 1).to_le_bytes();
        assert!(matches!(
            parse_bytes(8, &b),
            Err(crate::FormatError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn array_of_u32() {
        // array: u32 elem type + u64 count + payload
        let mut b = 4u32.to_le_bytes().to_vec(); // elem type = u32
        b.extend_from_slice(&3u64.to_le_bytes()); // count = 3
        for v in [1u32, 2, 3] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let v = parse_bytes(9, &b).unwrap();
        assert_eq!(v.array_len(), Some(3));
    }

    #[test]
    fn array_count_over_limit_rejected() {
        let mut b = 4u32.to_le_bytes().to_vec();
        b.extend_from_slice(&(crate::limits::MAX_ARRAY_ELEMS + 1).to_le_bytes());
        assert!(matches!(
            parse_bytes(9, &b),
            Err(crate::FormatError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn deep_array_nesting_rejected() {
        // arrays-of-arrays beyond MAX_ARRAY_DEPTH must be rejected, not recurse.
        // Build: array(elem=array(elem=array(...))) with depth 5, count 1 each.
        let mut b: Vec<u8> = Vec::new();
        for _ in 0..5 {
            b.extend_from_slice(&9u32.to_le_bytes()); // elem type = array
            b.extend_from_slice(&1u64.to_le_bytes()); // count = 1
        }
        assert!(parse_bytes(9, &b).is_err());
    }

    #[test]
    fn truncated_input_is_error_not_panic() {
        assert!(parse_bytes(4, &[0x01]).is_err()); // u32 needs 4 bytes
    }

    #[test]
    fn unknown_type_id_rejected() {
        assert!(parse_bytes(99, &[]).is_err());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p inferno-formats`
Expected: compile FAILURE — `GgufValue` not defined.

- [ ] **Step 3: Implement**

`crates/inferno-formats/src/read.rs`:

```rust
//! Bounded little-endian primitive readers over `io::Read`.
//! Truncation maps to `Malformed`, never a panic; strings are length-limited.

use std::io::Read;

use crate::{FormatError, Result, limits};

fn fill<R: Read>(r: &mut R, buf: &mut [u8], context: &'static str) -> Result<()> {
    r.read_exact(buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            FormatError::Malformed { context, detail: "unexpected end of input".into() }
        } else {
            FormatError::Io(e)
        }
    })
}

macro_rules! reader {
    ($name:ident, $ty:ty) => {
        pub(crate) fn $name<R: Read>(r: &mut R) -> Result<$ty> {
            let mut buf = [0u8; size_of::<$ty>()];
            fill(r, &mut buf, stringify!($ty))?;
            Ok(<$ty>::from_le_bytes(buf))
        }
    };
}

reader!(read_u8, u8);
reader!(read_u16, u16);
reader!(read_u32, u32);
reader!(read_u64, u64);
reader!(read_i8, i8);
reader!(read_i16, i16);
reader!(read_i32, i32);
reader!(read_i64, i64);
reader!(read_f32, f32);
reader!(read_f64, f64);

pub(crate) fn read_bool<R: Read>(r: &mut R) -> Result<bool> {
    match read_u8(r)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(FormatError::Malformed {
            context: "bool",
            detail: format!("expected 0 or 1, got {other}"),
        }),
    }
}

/// GGUF string: u64 LE byte length + UTF-8 payload.
pub(crate) fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)?;
    if len > limits::MAX_STRING_BYTES {
        return Err(FormatError::LimitExceeded {
            what: "string length",
            got: len,
            limit: limits::MAX_STRING_BYTES,
        });
    }
    let mut buf = vec![0u8; len as usize];
    fill(r, &mut buf, "string payload")?;
    String::from_utf8(buf).map_err(|_| FormatError::Malformed {
        context: "string payload",
        detail: "invalid utf-8".into(),
    })
}
```

`crates/inferno-formats/src/gguf/value.rs` (above the Step 1 tests):

```rust
//! GGUF metadata value tree (types 0–12 of the GGUF spec).

use std::io::Read;

use crate::read::*;
use crate::{FormatError, Result, limits};

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn parse<R: Read>(r: &mut R, type_id: u32, depth: u32) -> Result<Self> {
        Ok(match type_id {
            0 => Self::U8(read_u8(r)?),
            1 => Self::I8(read_i8(r)?),
            2 => Self::U16(read_u16(r)?),
            3 => Self::I16(read_i16(r)?),
            4 => Self::U32(read_u32(r)?),
            5 => Self::I32(read_i32(r)?),
            6 => Self::F32(read_f32(r)?),
            7 => Self::Bool(read_bool(r)?),
            8 => Self::String(read_string(r)?),
            9 => {
                if depth >= limits::MAX_ARRAY_DEPTH {
                    return Err(FormatError::LimitExceeded {
                        what: "array nesting depth",
                        got: u64::from(depth) + 1,
                        limit: u64::from(limits::MAX_ARRAY_DEPTH),
                    });
                }
                let elem_type = read_u32(r)?;
                let count = read_u64(r)?;
                if count > limits::MAX_ARRAY_ELEMS {
                    return Err(FormatError::LimitExceeded {
                        what: "array element count",
                        got: count,
                        limit: limits::MAX_ARRAY_ELEMS,
                    });
                }
                // No preallocation from the untrusted count: elements are
                // parsed one at a time, so a lying count hits EOF cheaply.
                let mut items = Vec::new();
                for _ in 0..count {
                    items.push(Self::parse(r, elem_type, depth + 1)?);
                }
                Self::Array(items)
            }
            10 => Self::U64(read_u64(r)?),
            11 => Self::I64(read_i64(r)?),
            12 => Self::F64(read_f64(r)?),
            other => {
                return Err(FormatError::Malformed {
                    context: "metadata value type",
                    detail: format!("unknown type id {other}"),
                });
            }
        })
    }

    /// Widen any integer value to u64 (metadata writers vary integer widths).
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v.into()),
            Self::U16(v) => Some(v.into()),
            Self::U32(v) => Some(v.into()),
            Self::U64(v) => Some(v),
            Self::I8(v) => u64::try_from(v).ok(),
            Self::I16(v) => u64::try_from(v).ok(),
            Self::I32(v) => u64::try_from(v).ok(),
            Self::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn array_len(&self) -> Option<u64> {
        match self {
            Self::Array(v) => Some(v.len() as u64),
            _ => None,
        }
    }
}
```

`crates/inferno-formats/src/gguf/mod.rs` (shell for now):

```rust
//! GGUF parsing. `value` holds the metadata tree; header parsing lands next.

pub(crate) mod value;
```

Add to `lib.rs` after `pub mod limits;`:

```rust
pub mod gguf;
mod read;
```

- [ ] **Step 4: Run tests**

Run: `cargo nextest run -p inferno-formats`
Expected: all tests PASS (12 total across Tasks 3–4).

- [ ] **Step 5: Lint + commit**

```bash
mise run lint && git add -A && git commit -m "feat(formats): bounded readers and GGUF metadata value parsing"
```

---

### Task 5: GGUF header → ModelDesc, plus the tiny-model fixture builder

**Files:**
- Create: `crates/inferno-formats/src/fixtures.rs`
- Modify: `crates/inferno-formats/src/gguf/mod.rs`, `crates/inferno-formats/src/lib.rs`

**Interfaces:**
- Consumes: `GgufValue` (Task 4), `ModelDesc`/`TensorDesc`/`DType` (Task 3)
- Produces:
  - `gguf::parse(r: &mut impl Read) -> Result<ModelDesc>` — parses header + tensor infos, never tensor data; returned desc has empty `weight_files` (the caller in Task 8 fills path + copies `data_section_offsets[0]` from the parse)
  - `fixtures::tiny_llama_gguf() -> Vec<u8>` — a complete, valid ~2 KB GGUF v3 model (llama arch, 2 layers, hidden 8, heads 2, kv-heads 1, ffn 16, vocab 32, F32 tensors)
  - `fixtures::TINY_HP: fn() -> HyperParams` equivalent values for assertions

- [ ] **Step 1: Write the fixture builder** (needed to express the tests)

`crates/inferno-formats/src/fixtures.rs`:

```rust
//! Tiny in-memory models for tests, fuzz corpus seeds, and CLI snapshots.
//! Also consumed by later milestones (M1 interpreter tests). Not a public
//! stability surface.

use crate::HyperParams;

pub fn tiny_hyperparams() -> HyperParams {
    HyperParams {
        vocab_size: 32,
        hidden_size: 8,
        n_layers: 2,
        n_heads: 2,
        n_kv_heads: 1,
        ffn_hidden_size: 16,
        rope_theta: 10000.0,
        norm_eps: 1e-5,
        context_length: 128,
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_kv_u32(out: &mut Vec<u8>, key: &str, v: u32) {
    put_str(out, key);
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_kv_f32(out: &mut Vec<u8>, key: &str, v: f32) {
    put_str(out, key);
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_kv_str(out: &mut Vec<u8>, key: &str, v: &str) {
    put_str(out, key);
    out.extend_from_slice(&8u32.to_le_bytes());
    put_str(out, v);
}

/// Tensor list for the tiny llama: (name, row-major shape).
/// GGUF stores dims fastest-first, so the writer reverses these.
pub fn tiny_tensor_shapes() -> Vec<(String, Vec<u64>)> {
    let hp = tiny_hyperparams();
    let (v, h, f) = (hp.vocab_size, hp.hidden_size, hp.ffn_hidden_size);
    let head_dim = h / hp.n_heads; // 4
    let kv_dim = head_dim * hp.n_kv_heads; // 4
    let mut t = vec![
        ("token_embd.weight".into(), vec![v, h]),
        ("output_norm.weight".into(), vec![h]),
        ("output.weight".into(), vec![v, h]),
    ];
    for i in 0..hp.n_layers {
        for (suffix, shape) in [
            ("attn_norm.weight", vec![h]),
            ("attn_q.weight", vec![h, h]),
            ("attn_k.weight", vec![kv_dim, h]),
            ("attn_v.weight", vec![kv_dim, h]),
            ("attn_output.weight", vec![h, h]),
            ("ffn_norm.weight", vec![h]),
            ("ffn_gate.weight", vec![f, h]),
            ("ffn_up.weight", vec![f, h]),
            ("ffn_down.weight", vec![h, f]),
        ] {
            t.push((format!("blk.{i}.{suffix}"), shape));
        }
    }
    t
}

/// A complete, valid GGUF v3 file (F32 tensors, data zero-filled).
pub fn tiny_llama_gguf() -> Vec<u8> {
    let hp = tiny_hyperparams();
    let tensors = tiny_tensor_shapes();
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes()); // version
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&10u64.to_le_bytes()); // kv count — keep in sync below!

    put_kv_str(&mut out, "general.architecture", "llama");
    put_kv_str(&mut out, "general.name", "tiny-llama-test");
    put_kv_u32(&mut out, "general.alignment", 32);
    put_kv_u32(&mut out, "llama.block_count", hp.n_layers as u32);
    put_kv_u32(&mut out, "llama.embedding_length", hp.hidden_size as u32);
    put_kv_u32(&mut out, "llama.attention.head_count", hp.n_heads as u32);
    put_kv_u32(&mut out, "llama.attention.head_count_kv", hp.n_kv_heads as u32);
    put_kv_u32(&mut out, "llama.feed_forward_length", hp.ffn_hidden_size as u32);
    put_kv_u32(&mut out, "llama.context_length", hp.context_length as u32);
    put_kv_f32(&mut out, "llama.attention.layer_norm_rms_epsilon", hp.norm_eps);
    // vocab_size key deliberately omitted: exercises the token_embd fallback.

    // Tensor infos. Offsets are relative to the (32-aligned) data section.
    let mut offset = 0u64;
    for (name, shape) in &tensors {
        put_str(&mut out, name);
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in shape.iter().rev() {
            // fastest-first on disk
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&0u32.to_le_bytes()); // ggml type 0 = F32
        out.extend_from_slice(&offset.to_le_bytes());
        let n: u64 = shape.iter().product();
        offset += (n * 4).next_multiple_of(32);
    }

    // Data section: align, then zero-fill.
    while out.len() % 32 != 0 {
        out.push(0);
    }
    out.resize(out.len() + offset as usize, 0);
    out
}
```

Register in `lib.rs`: add `pub mod fixtures;`.

- [ ] **Step 2: Write failing parse tests** (append to `gguf/mod.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Architecture, DType, fixtures};
    use std::io::Cursor;

    #[test]
    fn parses_tiny_llama() {
        let bytes = fixtures::tiny_llama_gguf();
        let desc = parse(&mut Cursor::new(&bytes)).unwrap();
        assert_eq!(desc.architecture, Architecture::Llama);
        assert_eq!(desc.name.as_deref(), Some("tiny-llama-test"));
        assert_eq!(desc.hyperparams, fixtures::tiny_hyperparams());
        assert_eq!(desc.tensors.len(), fixtures::tiny_tensor_shapes().len());

        let embd = &desc.tensors[0];
        assert_eq!(embd.name, "token_embd.weight");
        assert_eq!(embd.shape, vec![32, 8]); // row-major: [vocab, hidden]
        assert_eq!(embd.dtype, DType::F32);
        assert_eq!(embd.data_len, Some(32 * 8 * 4));
        assert_eq!(desc.data_section_offsets.len(), 1);
        assert_eq!(desc.data_section_offsets[0] % 32, 0);
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(matches!(
            parse(&mut Cursor::new(b"GGML........")),
            Err(crate::FormatError::BadMagic { .. })
        ));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&99u32.to_le_bytes());
        b.extend_from_slice(&[0u8; 16]);
        assert!(matches!(
            parse(&mut Cursor::new(&b)),
            Err(crate::FormatError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn rejects_huge_tensor_count() {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // tensor count
        b.extend_from_slice(&0u64.to_le_bytes()); // kv count
        assert!(matches!(
            parse(&mut Cursor::new(&b)),
            Err(crate::FormatError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn truncated_tensor_info_is_error_not_panic() {
        let bytes = fixtures::tiny_llama_gguf();
        // Cut the file mid-way through the tensor-info block.
        let cut = &bytes[..bytes.len() / 3];
        assert!(parse(&mut Cursor::new(cut)).is_err());
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo nextest run -p inferno-formats`
Expected: compile FAILURE — `gguf::parse` not defined.

- [ ] **Step 4: Implement the parser** (replace `gguf/mod.rs` contents above the tests)

```rust
//! GGUF header parsing (versions 2 and 3): magic, metadata KVs, tensor infos.
//! Tensor *data* is never read here.

pub(crate) mod value;

use std::collections::BTreeMap;
use std::io::Read;

use value::GgufValue;

use crate::read::*;
use crate::{
    Architecture, DType, FormatError, HyperParams, ModelDesc, Result, TensorDesc, limits,
};

/// `io::Read` wrapper that tracks the byte position, so we can compute where
/// the aligned data section starts without requiring `Seek`.
struct Counting<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

fn dtype_from_ggml(type_id: u32) -> DType {
    match type_id {
        0 => DType::F32,
        1 => DType::F16,
        8 => DType::Q8_0,
        12 => DType::Q4_K,
        30 => DType::BF16,
        other => DType::Unsupported(format!("ggml:{other}")),
    }
}

pub fn parse<R: Read>(r: &mut R) -> Result<ModelDesc> {
    let mut r = Counting { inner: r, pos: 0 };

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|_| FormatError::BadMagic { expected: "GGUF" })?;
    if &magic != b"GGUF" {
        return Err(FormatError::BadMagic { expected: "GGUF" });
    }
    let version = read_u32(&mut r)?;
    if !(2..=3).contains(&version) {
        return Err(FormatError::UnsupportedVersion(version));
    }

    let tensor_count = read_u64(&mut r)?;
    if tensor_count > limits::MAX_TENSORS {
        return Err(FormatError::LimitExceeded {
            what: "tensor count",
            got: tensor_count,
            limit: limits::MAX_TENSORS,
        });
    }
    let kv_count = read_u64(&mut r)?;
    if kv_count > limits::MAX_KV_PAIRS {
        return Err(FormatError::LimitExceeded {
            what: "metadata kv count",
            got: kv_count,
            limit: limits::MAX_KV_PAIRS,
        });
    }

    let mut meta = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_string(&mut r)?;
        let type_id = read_u32(&mut r)?;
        let value = GgufValue::parse(&mut r, type_id, 0)?;
        meta.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count.min(1024) as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut r)?;
        let n_dims = read_u32(&mut r)?;
        if n_dims > limits::MAX_DIMS {
            return Err(FormatError::LimitExceeded {
                what: "tensor rank",
                got: n_dims.into(),
                limit: limits::MAX_DIMS.into(),
            });
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(read_u64(&mut r)?);
        }
        shape.reverse(); // GGUF stores fastest-varying first; we are row-major.
        let dtype = dtype_from_ggml(read_u32(&mut r)?);
        let data_offset = read_u64(&mut r)?;

        let n_elems = shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| FormatError::Malformed {
                context: "tensor shape",
                detail: format!("{name}: element count overflows u64"),
            })?;
        let data_len = dtype.byte_len(n_elems);

        tensors.push(TensorDesc {
            name,
            shape,
            dtype,
            file_index: 0,
            data_offset,
            data_len,
        });
    }

    let alignment = meta
        .get("general.alignment")
        .and_then(GgufValue::as_u64)
        .unwrap_or(32);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(FormatError::Malformed {
            context: "general.alignment",
            detail: format!("{alignment} is not a power of two"),
        });
    }
    for t in &tensors {
        if t.data_offset % alignment != 0 {
            return Err(FormatError::Malformed {
                context: "tensor offset",
                detail: format!("{}: offset {} not {}-aligned", t.name, t.data_offset, alignment),
            });
        }
    }
    let data_section = r.pos.next_multiple_of(alignment);

    let (architecture, name, hyperparams) = extract_hyperparams(&meta, &tensors)?;

    Ok(ModelDesc {
        architecture,
        name,
        hyperparams,
        tensors,
        weight_files: Vec::new(), // caller (load_desc) records the path
        data_section_offsets: vec![data_section],
    })
}

fn get_u64(meta: &BTreeMap<String, GgufValue>, key: &str) -> Result<u64> {
    meta.get(key)
        .and_then(GgufValue::as_u64)
        .ok_or_else(|| FormatError::MissingKey(key.to_string()))
}

fn extract_hyperparams(
    meta: &BTreeMap<String, GgufValue>,
    tensors: &[TensorDesc],
) -> Result<(Architecture, Option<String>, HyperParams)> {
    let arch_id = meta
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .ok_or_else(|| FormatError::MissingKey("general.architecture".into()))?;
    let architecture = Architecture::from_id(arch_id);
    let name = meta
        .get("general.name")
        .and_then(GgufValue::as_str)
        .map(str::to_string);

    let k = |suffix: &str| format!("{arch_id}.{suffix}");
    let n_heads = get_u64(meta, &k("attention.head_count"))?;
    let vocab_size = match get_u64(meta, &k("vocab_size")) {
        Ok(v) => v,
        // Fallbacks: tokenizer vocab length, then token_embd row count.
        Err(_) => meta
            .get("tokenizer.ggml.tokens")
            .and_then(GgufValue::array_len)
            .or_else(|| {
                tensors
                    .iter()
                    .find(|t| t.name == "token_embd.weight")
                    .and_then(|t| t.shape.first().copied())
            })
            .ok_or_else(|| FormatError::MissingKey(k("vocab_size")))?,
    };

    Ok((
        architecture,
        name,
        HyperParams {
            vocab_size,
            hidden_size: get_u64(meta, &k("embedding_length"))?,
            n_layers: get_u64(meta, &k("block_count"))?,
            n_heads,
            n_kv_heads: get_u64(meta, &k("attention.head_count_kv")).unwrap_or(n_heads),
            ffn_hidden_size: get_u64(meta, &k("feed_forward_length"))?,
            rope_theta: meta
                .get(&k("rope.freq_base"))
                .and_then(GgufValue::as_f32)
                .unwrap_or(10000.0),
            norm_eps: meta
                .get(&k("attention.layer_norm_rms_epsilon"))
                .and_then(GgufValue::as_f32)
                .unwrap_or(1e-5),
            context_length: get_u64(meta, &k("context_length")).unwrap_or(0),
        },
    ))
}
```

- [ ] **Step 5: Run tests**

Run: `cargo nextest run -p inferno-formats`
Expected: all PASS, including the 5 new gguf tests.

- [ ] **Step 6: Lint + commit**

```bash
mise run lint && git add -A && git commit -m "feat(formats): GGUF header parser and tiny-model fixture builder"
```

---

### Task 6: safetensors header parsing

**Files:**
- Create: `crates/inferno-formats/src/safetensors.rs`
- Modify: `crates/inferno-formats/src/lib.rs` (add `pub mod safetensors;`)

**Interfaces:**
- Consumes: `read::read_u64`, `TensorDesc`, `DType`, `FormatError`, `limits`
- Produces: `safetensors::parse(r: &mut impl Read, file_index: u32) -> Result<(Vec<TensorDesc>, u64)>` — the `u64` is the data-section offset (8 + header length). Task 7 (MLX) and Task 10 (fuzz) consume this.

- [ ] **Step 1: Write failing tests** (bottom of `safetensors.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use std::io::Cursor;

    fn wrap(json: &str) -> Vec<u8> {
        let mut b = (json.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(json.as_bytes());
        b
    }

    #[test]
    fn parses_two_tensors() {
        let json = r#"{
            "model.embed_tokens.weight": {"dtype":"F32","shape":[32,8],"data_offsets":[0,1024]},
            "model.norm.weight": {"dtype":"BF16","shape":[8],"data_offsets":[1024,1040]},
            "__metadata__": {"format":"mlx"}
        }"#;
        let (tensors, data_off) = parse(&mut Cursor::new(wrap(json)), 0).unwrap();
        assert_eq!(data_off, 8 + json.len() as u64);
        assert_eq!(tensors.len(), 2); // __metadata__ skipped
        let e = tensors.iter().find(|t| t.name.ends_with("embed_tokens.weight")).unwrap();
        assert_eq!(e.dtype, DType::F32);
        assert_eq!(e.shape, vec![32, 8]);
        assert_eq!(e.data_offset, 0);
        assert_eq!(e.data_len, Some(1024));
    }

    #[test]
    fn unknown_dtype_is_unsupported_not_error() {
        let json = r#"{"w": {"dtype":"U32","shape":[4],"data_offsets":[0,16]}}"#;
        let (tensors, _) = parse(&mut Cursor::new(wrap(json)), 0).unwrap();
        assert_eq!(tensors[0].dtype, DType::Unsupported("U32".into()));
        assert_eq!(tensors[0].data_len, Some(16)); // trusted from offsets
    }

    #[test]
    fn rejects_length_mismatch() {
        // F32 [4] must span 16 bytes, not 15.
        let json = r#"{"w": {"dtype":"F32","shape":[4],"data_offsets":[0,15]}}"#;
        assert!(parse(&mut Cursor::new(wrap(json)), 0).is_err());
    }

    #[test]
    fn rejects_reversed_offsets() {
        let json = r#"{"w": {"dtype":"F32","shape":[4],"data_offsets":[16,0]}}"#;
        assert!(parse(&mut Cursor::new(wrap(json)), 0).is_err());
    }

    #[test]
    fn rejects_header_over_limit() {
        let mut b = (crate::limits::MAX_ST_HEADER_BYTES + 1).to_le_bytes().to_vec();
        b.extend_from_slice(b"{}");
        assert!(matches!(
            parse(&mut Cursor::new(b), 0),
            Err(crate::FormatError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn rejects_garbage_json() {
        assert!(parse(&mut Cursor::new(wrap("not json")), 0).is_err());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p inferno-formats`
Expected: compile FAILURE — `safetensors::parse` not defined.

- [ ] **Step 3: Implement** (top of `safetensors.rs`)

```rust
//! safetensors header parsing: u64 LE header length + JSON tensor table.
//! Used directly for MLX model files. Tensor data is never read.

use std::io::Read;

use serde::Deserialize;

use crate::read::read_u64;
use crate::{DType, FormatError, Result, TensorDesc, limits};

#[derive(Deserialize)]
struct StEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

fn dtype_from_st(s: &str) -> DType {
    match s {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        other => DType::Unsupported(other.to_string()),
    }
}

pub fn parse<R: Read>(r: &mut R, file_index: u32) -> Result<(Vec<TensorDesc>, u64)> {
    let header_len = read_u64(r)?;
    if header_len > limits::MAX_ST_HEADER_BYTES {
        return Err(FormatError::LimitExceeded {
            what: "safetensors header length",
            got: header_len,
            limit: limits::MAX_ST_HEADER_BYTES,
        });
    }
    let mut json = vec![0u8; header_len as usize];
    r.read_exact(&mut json).map_err(|_| FormatError::Malformed {
        context: "safetensors header",
        detail: "truncated before header end".into(),
    })?;

    let table: std::collections::BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&json)?;

    let mut tensors = Vec::new();
    for (name, v) in table {
        if name == "__metadata__" {
            continue;
        }
        let entry: StEntry =
            serde_json::from_value(v).map_err(FormatError::Json)?;
        let [start, end] = entry.data_offsets;
        if end < start {
            return Err(FormatError::Malformed {
                context: "safetensors data_offsets",
                detail: format!("{name}: end {end} < start {start}"),
            });
        }
        let span = end - start;
        let dtype = dtype_from_st(&entry.dtype);
        let n_elems = entry
            .shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| FormatError::Malformed {
                context: "safetensors shape",
                detail: format!("{name}: element count overflows u64"),
            })?;
        if let Some(expect) = dtype.byte_len(n_elems) {
            if expect != span {
                return Err(FormatError::Malformed {
                    context: "safetensors data_offsets",
                    detail: format!("{name}: span {span} != dtype size {expect}"),
                });
            }
        }
        tensors.push(TensorDesc {
            name,
            shape: entry.shape,
            dtype,
            file_index,
            data_offset: start,
            data_len: Some(span),
        });
    }
    Ok((tensors, 8 + header_len))
}
```

- [ ] **Step 4: Run tests**

Run: `cargo nextest run -p inferno-formats`
Expected: all PASS.

- [ ] **Step 5: Lint + commit**

```bash
mise run lint && git add -A && git commit -m "feat(formats): safetensors header parsing with offset validation"
```

---

### Task 7: MLX directory loader (config.json + shards)

**Files:**
- Create: `crates/inferno-formats/src/mlx.rs`
- Modify: `crates/inferno-formats/src/lib.rs` (add `pub(crate) mod mlx;`), `crates/inferno-formats/src/fixtures.rs`

**Interfaces:**
- Consumes: `safetensors::parse` (Task 6), desc types (Task 3)
- Produces:
  - `mlx::load_dir(dir: &Path) -> Result<ModelDesc>` — reads `config.json` + all `*.safetensors` (sorted by filename; `file_index` = sort position)
  - `fixtures::tiny_llama_safetensors() -> Vec<u8>` and `fixtures::tiny_llama_config_json() -> String` (same hyperparams as the GGUF fixture)

- [ ] **Step 1: Extend fixtures** (append to `fixtures.rs`)

```rust
/// The tiny llama as a single MLX-style safetensors file (F32, zero data).
pub fn tiny_llama_safetensors() -> Vec<u8> {
    let mut entries = Vec::new();
    let mut offset = 0u64;
    for (name, shape) in tiny_tensor_shapes() {
        // HF/MLX naming differs from GGUF naming; that mapping is M1's
        // problem (graph builder). M0 records names verbatim.
        let n: u64 = shape.iter().product();
        let end = offset + n * 4;
        entries.push(format!(
            r#""{name}": {{"dtype":"F32","shape":[{}],"data_offsets":[{offset},{end}]}}"#,
            shape.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
        ));
        offset = end;
    }
    let json = format!("{{{}}}", entries.join(","));
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(json.as_bytes());
    out.resize(out.len() + offset as usize, 0);
    out
}

/// Matching MLX config.json (HF-style keys).
pub fn tiny_llama_config_json() -> String {
    let hp = tiny_hyperparams();
    format!(
        r#"{{
  "model_type": "llama",
  "hidden_size": {},
  "num_hidden_layers": {},
  "num_attention_heads": {},
  "num_key_value_heads": {},
  "intermediate_size": {},
  "vocab_size": {},
  "rope_theta": {},
  "rms_norm_eps": {},
  "max_position_embeddings": {}
}}"#,
        hp.hidden_size,
        hp.n_layers,
        hp.n_heads,
        hp.n_kv_heads,
        hp.ffn_hidden_size,
        hp.vocab_size,
        hp.rope_theta,
        hp.norm_eps,
        hp.context_length
    )
}
```

- [ ] **Step 2: Write failing tests** (bottom of `mlx.rs`; uses a temp dir built from fixtures)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Architecture, fixtures};

    fn write_tiny_mlx_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("inferno-mlx-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), fixtures::tiny_llama_config_json()).unwrap();
        std::fs::write(dir.join("model.safetensors"), fixtures::tiny_llama_safetensors()).unwrap();
        dir
    }

    #[test]
    fn loads_tiny_mlx_dir() {
        let dir = write_tiny_mlx_dir();
        let desc = load_dir(&dir).unwrap();
        assert_eq!(desc.architecture, Architecture::Llama);
        assert_eq!(desc.hyperparams, fixtures::tiny_hyperparams());
        assert_eq!(desc.tensors.len(), fixtures::tiny_tensor_shapes().len());
        assert_eq!(desc.weight_files, vec![dir.join("model.safetensors")]);
        assert_eq!(desc.data_section_offsets.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_config_is_clear_error() {
        let dir = std::env::temp_dir().join(format!("inferno-mlx-noconf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), fixtures::tiny_llama_safetensors()).unwrap();
        let err = load_dir(&dir).unwrap_err().to_string();
        assert!(err.contains("config.json"), "unhelpful error: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_safetensors_is_clear_error() {
        let dir = std::env::temp_dir().join(format!("inferno-mlx-nost-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), fixtures::tiny_llama_config_json()).unwrap();
        assert!(load_dir(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo nextest run -p inferno-formats`
Expected: compile FAILURE — `mlx::load_dir` not defined.

- [ ] **Step 4: Implement** (top of `mlx.rs`)

```rust
//! MLX model directories: HF-style config.json + one or more .safetensors.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{Architecture, FormatError, HyperParams, ModelDesc, Result, safetensors};

#[derive(Deserialize)]
struct MlxConfig {
    model_type: String,
    hidden_size: u64,
    num_hidden_layers: u64,
    num_attention_heads: u64,
    num_key_value_heads: Option<u64>,
    intermediate_size: u64,
    vocab_size: u64,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_norm_eps")]
    rms_norm_eps: f32,
    #[serde(default)]
    max_position_embeddings: u64,
}

fn default_rope_theta() -> f32 {
    10000.0
}
fn default_norm_eps() -> f32 {
    1e-5
}

pub fn load_dir(dir: &Path) -> Result<ModelDesc> {
    let config_path = dir.join("config.json");
    let config_file = File::open(&config_path).map_err(|e| FormatError::Malformed {
        context: "mlx model directory",
        detail: format!("cannot open {}: {e}", config_path.display()),
    })?;
    let config: MlxConfig = serde_json::from_reader(BufReader::new(config_file))?;

    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shard_paths.sort();
    if shard_paths.is_empty() {
        return Err(FormatError::Malformed {
            context: "mlx model directory",
            detail: format!("no .safetensors files in {}", dir.display()),
        });
    }

    let mut tensors = Vec::new();
    let mut data_section_offsets = Vec::new();
    for (i, path) in shard_paths.iter().enumerate() {
        let mut reader = BufReader::new(File::open(path)?);
        let (mut shard_tensors, data_off) = safetensors::parse(&mut reader, i as u32)?;
        tensors.append(&mut shard_tensors);
        data_section_offsets.push(data_off);
    }

    Ok(ModelDesc {
        architecture: Architecture::from_id(&config.model_type),
        name: dir.file_name().map(|n| n.to_string_lossy().into_owned()),
        hyperparams: HyperParams {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            n_layers: config.num_hidden_layers,
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads.unwrap_or(config.num_attention_heads),
            ffn_hidden_size: config.intermediate_size,
            rope_theta: config.rope_theta,
            norm_eps: config.rms_norm_eps,
            context_length: config.max_position_embeddings,
        },
        tensors,
        weight_files: shard_paths,
        data_section_offsets,
    })
}
```

- [ ] **Step 5: Run tests**

Run: `cargo nextest run -p inferno-formats`
Expected: all PASS.

- [ ] **Step 6: Lint + commit**

```bash
mise run lint && git add -A && git commit -m "feat(formats): MLX directory loader (config.json + safetensors shards)"
```

---

### Task 8: Unified `load_desc` entry point + committed fixtures + snapshots

**Files:**
- Create: `crates/inferno-formats/examples/gen_fixtures.rs`, `crates/inferno-formats/tests/snapshot_desc.rs`
- Create (generated, committed): `crates/inferno-formats/tests/fixtures/tiny.gguf`, `tests/fixtures/mlx/config.json`, `tests/fixtures/mlx/model.safetensors`
- Modify: `crates/inferno-formats/src/lib.rs`

**Interfaces:**
- Consumes: `gguf::parse`, `mlx::load_dir` (Tasks 5, 7)
- Produces: `inferno_formats::load_desc(path: &Path) -> Result<ModelDesc>` — the ONLY model-loading entry point for the CLI (Task 9) and all later milestones. Directory → MLX; `.gguf` file or GGUF magic → GGUF; `.safetensors` file → MLX using the file's parent for `config.json`.

- [ ] **Step 1: Write failing integration test** (`tests/snapshot_desc.rs`)

```rust
use std::path::Path;

fn fixture(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(rel)
}

#[test]
fn gguf_desc_snapshot() {
    let desc = inferno_formats::load_desc(&fixture("tiny.gguf")).unwrap();
    insta::assert_yaml_snapshot!("tiny_gguf_desc", desc);
}

#[test]
fn mlx_desc_snapshot() {
    let desc = inferno_formats::load_desc(&fixture("mlx")).unwrap();
    insta::assert_yaml_snapshot!("tiny_mlx_desc", desc);
}

#[test]
fn gguf_weight_file_and_offset_recorded() {
    let path = fixture("tiny.gguf");
    let desc = inferno_formats::load_desc(&path).unwrap();
    assert_eq!(desc.weight_files, vec![path]);
    assert_eq!(desc.data_section_offsets.len(), 1);
}

#[test]
fn unknown_format_is_clear_error() {
    let err = inferno_formats::load_desc(Path::new("Cargo.toml")).unwrap_err();
    assert!(matches!(err, inferno_formats::FormatError::UnknownFormat(_)));
}
```

- [ ] **Step 2: Write the fixture generator** (`examples/gen_fixtures.rs`)

```rust
//! Regenerates the committed test fixtures and fuzz corpus seeds.
//! Run: cargo run -p inferno-formats --example gen_fixtures

use std::fs;
use std::path::Path;

use inferno_formats::fixtures;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fix = root.join("tests/fixtures");
    fs::create_dir_all(fix.join("mlx")).unwrap();
    fs::write(fix.join("tiny.gguf"), fixtures::tiny_llama_gguf()).unwrap();
    fs::write(fix.join("mlx/config.json"), fixtures::tiny_llama_config_json()).unwrap();
    fs::write(fix.join("mlx/model.safetensors"), fixtures::tiny_llama_safetensors()).unwrap();

    // Fuzz corpus seeds (fuzz/ is created in the fuzz task; ignore if absent).
    let corpus = root.join("../../fuzz/corpus");
    if corpus.parent().is_some_and(|p| p.join("Cargo.toml").exists()) {
        fs::create_dir_all(corpus.join("gguf_parse")).unwrap();
        fs::create_dir_all(corpus.join("safetensors_parse")).unwrap();
        fs::write(corpus.join("gguf_parse/tiny.gguf"), fixtures::tiny_llama_gguf()).unwrap();
        fs::write(
            corpus.join("safetensors_parse/tiny.safetensors"),
            fixtures::tiny_llama_safetensors(),
        )
        .unwrap();
    }
    println!("fixtures written under {}", fix.display());
}
```

- [ ] **Step 3: Implement `load_desc`** (append to `lib.rs`, plus module line `pub(crate) mod mlx;` if not present)

```rust
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Load a model description from a GGUF file, an MLX directory, or a
/// single .safetensors file (with sibling config.json).
pub fn load_desc(path: &Path) -> Result<ModelDesc> {
    if path.is_dir() {
        return mlx::load_dir(path);
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "safetensors" {
        let dir = path.parent().ok_or_else(|| {
            FormatError::UnknownFormat(format!("{}: no parent directory", path.display()))
        })?;
        return mlx::load_dir(dir);
    }

    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic == b"GGUF" {
        // Re-open so the parser sees the magic too (readers are not Seek).
        let mut reader = BufReader::new(File::open(path)?);
        let mut desc = gguf::parse(&mut reader)?;
        desc.weight_files = vec![path.to_path_buf()];
        return Ok(desc);
    }
    Err(FormatError::UnknownFormat(path.display().to_string()))
}
```

- [ ] **Step 4: Generate fixtures, run tests, review snapshots**

```bash
cargo run -p inferno-formats --example gen_fixtures
cargo nextest run -p inferno-formats
cargo insta review
```

Expected: generator prints the fixtures path; first nextest run creates 2 pending snapshots; `cargo insta review` shows YAML containing `architecture: Llama`, `vocab_size: 32`, and 21 tensors — **read both snapshots fully before accepting** (this is the review-every-snapshot rule). After accepting, `cargo nextest run -p inferno-formats` passes clean.

- [ ] **Step 5: Lint + commit (fixtures and snapshots included)**

```bash
mise run lint
git add -A && git commit -m "feat(formats): unified load_desc entry, committed fixtures, desc snapshots"
```

---

### Task 9: `inferno inspect` CLI

**Files:**
- Create: `cli/src/inspect.rs`, `cli/tests/inspect.rs`
- Modify: `cli/src/main.rs`

**Interfaces:**
- Consumes: `inferno_formats::load_desc` (Task 8)
- Produces: `inferno inspect <MODEL_PATH> [--tensors N]` (N defaults to 10; `--tensors 0` = hide tensor list). `inspect::render(desc: &ModelDesc, max_tensors: usize) -> String` is pure so tests snapshot it without terminal concerns.

- [ ] **Step 1: Write failing CLI test** (`cli/tests/inspect.rs`)

```rust
use assert_cmd::Command;

fn fixture(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../crates/inferno-formats/tests/fixtures")
        .join(rel)
}

#[test]
fn inspect_gguf_snapshot() {
    let out = Command::cargo_bin("inferno")
        .unwrap()
        .args(["inspect", fixture("tiny.gguf").to_str().unwrap(), "--tensors", "3"])
        .assert()
        .success();
    let stdout = String::from_utf8(out.get_output().stdout.clone()).unwrap();
    insta::assert_snapshot!("inspect_gguf", stdout);
}

#[test]
fn inspect_mlx_dir() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args(["inspect", fixture("mlx").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicates::str::contains("architecture: llama"));
}

#[test]
fn inspect_missing_file_fails_cleanly() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args(["inspect", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("error"));
}
```

Add to `cli/Cargo.toml` `[dev-dependencies]`: `predicates = "3"` (and add `predicates = "3"` to `[workspace.dependencies]`, referenced as `predicates.workspace = true`).

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p inferno`
Expected: FAIL — binary has no `inspect` subcommand yet.

- [ ] **Step 3: Implement**

`cli/src/main.rs`:

```rust
mod inspect;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "inferno", about = "CPU-first LLM inference engine", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show a model file's architecture, hyperparameters, and tensors.
    Inspect {
        /// Path to a .gguf file, an MLX directory, or a .safetensors file.
        model: PathBuf,
        /// How many tensors to list (0 = none).
        #[arg(long, default_value_t = 10)]
        tensors: usize,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { model, tensors } => match inferno_formats::load_desc(&model) {
            Ok(desc) => {
                print!("{}", inspect::render(&desc, tensors));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
    }
}
```

`cli/src/inspect.rs`:

```rust
use inferno_formats::{Architecture, ModelDesc};

fn arch_label(a: &Architecture) -> String {
    match a {
        Architecture::Llama => "llama".into(),
        Architecture::Qwen2 => "qwen2".into(),
        Architecture::Qwen3 => "qwen3".into(),
        Architecture::Mistral => "mistral".into(),
        Architecture::Unknown(s) => format!("unknown ({s})"),
    }
}

pub fn render(desc: &ModelDesc, max_tensors: usize) -> String {
    let hp = &desc.hyperparams;
    let mut out = String::new();
    if let Some(name) = &desc.name {
        out.push_str(&format!("model: {name}\n"));
    }
    out.push_str(&format!("architecture: {}\n", arch_label(&desc.architecture)));
    out.push_str(&format!(
        "hyperparams: layers={} hidden={} heads={} kv_heads={} ffn={} vocab={} ctx={} rope_theta={} norm_eps={}\n",
        hp.n_layers, hp.hidden_size, hp.n_heads, hp.n_kv_heads,
        hp.ffn_hidden_size, hp.vocab_size, hp.context_length, hp.rope_theta, hp.norm_eps,
    ));
    out.push_str(&format!("tensors: {}\n", desc.tensors.len()));
    for t in desc.tensors.iter().take(max_tensors) {
        let shape = t.shape.iter().map(u64::to_string).collect::<Vec<_>>().join("x");
        out.push_str(&format!("  {:<40} {:>12} {:?}\n", t.name, shape, t.dtype));
    }
    if desc.tensors.len() > max_tensors && max_tensors > 0 {
        out.push_str(&format!("  … and {} more\n", desc.tensors.len() - max_tensors));
    }
    out
}
```

- [ ] **Step 4: Run tests, review snapshot**

```bash
cargo nextest run -p inferno && cargo insta review
```

Expected: `inspect_gguf` snapshot pending → review (must show `model: tiny-llama-test`, `architecture: llama`, 3 tensor lines + "… and 18 more") → accept → re-run passes.

- [ ] **Step 5: Manual verification (the deliverable works end-to-end)**

```bash
cargo run -p inferno -- inspect crates/inferno-formats/tests/fixtures/tiny.gguf
```

Expected output (exact):

```
model: tiny-llama-test
architecture: llama
hyperparams: layers=2 hidden=8 heads=2 kv_heads=1 ffn=16 vocab=32 ctx=128 rope_theta=10000 norm_eps=0.00001
tensors: 21
  token_embd.weight                                32x8 F32
  output_norm.weight                                  8 F32
  output.weight                                    32x8 F32
  blk.0.attn_norm.weight                              8 F32
  blk.0.attn_q.weight                               8x8 F32
  blk.0.attn_k.weight                               4x8 F32
  blk.0.attn_v.weight                               4x8 F32
  blk.0.attn_output.weight                          8x8 F32
  blk.0.ffn_norm.weight                               8 F32
  blk.0.ffn_gate.weight                            16x8 F32
  … and 11 more
```

(Float formatting may differ slightly — trust the reviewed snapshot, not this transcript.)

- [ ] **Step 6: Lint + commit**

```bash
mise run lint && git add -A && git commit -m "feat(cli): inferno inspect subcommand"
```

---

### Task 10: Fuzz targets for both parsers

**Files:**
- Create: `fuzz/Cargo.toml`, `fuzz/fuzz_targets/gguf_parse.rs`, `fuzz/fuzz_targets/safetensors_parse.rs`
- Create (generated, committed): `fuzz/corpus/gguf_parse/tiny.gguf`, `fuzz/corpus/safetensors_parse/tiny.safetensors`

**Interfaces:**
- Consumes: `gguf::parse`, `safetensors::parse` (public APIs, Tasks 5–6)
- Produces: `cargo fuzz` targets named `gguf_parse` and `safetensors_parse`; nightly CI (Task 12) runs them by these names via `mise run fuzz -- <name>`.

- [ ] **Step 1: Create the fuzz crate**

`fuzz/Cargo.toml`:

```toml
[package]
name = "inferno-fuzz"
version = "0.0.0"
edition = "2024"
publish = false

[package.metadata]
cargo-fuzz = true

[dependencies]
libfuzzer-sys = "0.4"
inferno-formats = { path = "../crates/inferno-formats" }

[[bin]]
name = "gguf_parse"
path = "fuzz_targets/gguf_parse.rs"
test = false
doc = false
bench = false

[[bin]]
name = "safetensors_parse"
path = "fuzz_targets/safetensors_parse.rs"
test = false
doc = false
bench = false
```

Note: `fuzz/` is intentionally NOT a workspace member (nightly-only deps must not infect the stable build). Add to the root `Cargo.toml` `[workspace]` section:

```toml
exclude = ["fuzz"]
```

`fuzz/fuzz_targets/gguf_parse.rs`:

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;

// The parser must be total over arbitrary bytes: any panic is a finding.
fuzz_target!(|data: &[u8]| {
    let _ = inferno_formats::gguf::parse(&mut std::io::Cursor::new(data));
});
```

`fuzz/fuzz_targets/safetensors_parse.rs`:

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = inferno_formats::safetensors::parse(&mut std::io::Cursor::new(data), 0);
});
```

- [ ] **Step 2: Seed the corpus and install prerequisites**

```bash
mise use --pin "cargo:cargo-fuzz@latest"
mise exec rust@nightly -- rustc --version   # downloads nightly on first use
cargo run -p inferno-formats --example gen_fixtures
git add fuzz/corpus
```

Expected: nightly rustc version prints; `fuzz/corpus/{gguf_parse,safetensors_parse}/` now contain the seed files.

- [ ] **Step 3: Smoke-run both targets (60s each)**

```bash
mise run fuzz -- gguf_parse
mise run fuzz -- safetensors_parse
```

Expected: each runs ~60s and exits 0 with `Done ...: 0 crashes`. Any crash file under `fuzz/artifacts/` is a real parser bug: minimize (`cargo fuzz tmin`), fix the parser with a regression unit test, and re-run before proceeding.

- [ ] **Step 4: Verify the stable build is unaffected**

```bash
mise run lint && mise run test
```

Expected: both pass with stable rustc (fuzz crate excluded from workspace).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test(formats): libFuzzer targets for gguf and safetensors parsers"
```

---

### Task 11: Repo front door + threat model

**Files:**
- Create: `AGENTS.md`, `CLAUDE.md`, `ARCHITECTURE.md`, `docs/threat-model.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: task names from Task 1; spec + crate layout (already committed)
- Produces: the onboarding sequence that Task 12's CI onboarding job executes verbatim.

- [ ] **Step 1: Write README.md** (replace existing)

```markdown
# inferno

CPU-first LLM inference engine. Inferno compiles each model for the exact
machine it runs on — hardware-detected code generation via LLVM, with memory
layout, quantization format, and thread partitioning specialized at
setup time — then caches the compiled artifact for instant reloads.

Loads GGUF and MLX (safetensors) models. No GPU: the goal is maximum speed on
commodity hardware, laptops to phones. Written in Rust.

**Status:** pre-release, milestone M0 (model-file parsing + tooling).

## Quickstart

Requires [devenv](https://devenv.sh) (native deps: LLVM, llama.cpp) and
[mise](https://mise.jdx.dev) (Rust toolchain + dev tools; task runner):

    devenv shell        # native deps
    mise install        # pinned toolchain
    mise run test       # fast test suite
    lefthook install    # pre-commit hooks (gitleaks, fmt)

Try it:

    cargo run -p inferno -- inspect crates/inferno-formats/tests/fixtures/tiny.gguf

## Common tasks

Run `mise tasks` for the authoritative list — `test`, `test-full`, `lint`,
`fmt`, `audit`, `fuzz`.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — crate map and why the boundaries fall
  where they do
- [docs/superpowers/specs/](docs/superpowers/specs/) — design specs
- [docs/threat-model.md](docs/threat-model.md) — what we defend against
```

- [ ] **Step 2: Write AGENTS.md**

```markdown
# Agent instructions — inferno

Everything derivable from code is not repeated here. Read
[ARCHITECTURE.md](ARCHITECTURE.md) for the crate map and
[docs/superpowers/specs/2026-07-04-inferno-v1-design.md](docs/superpowers/specs/2026-07-04-inferno-v1-design.md)
for the v1 design.

## Non-obvious constraints

- **Workflows are mise tasks** (`mise tasks`): use `mise run test` / `lint` /
  `audit` / `fuzz` — CI runs the same names. Don't hand-roll cargo invocations
  in docs or CI.
- **Toolchain:** rust + dev tools are mise-pinned (`mise.toml`); native deps
  (LLVM, llama.cpp) come ONLY from `devenv.nix`. The LLVM major version there
  (18) must match the `inkwell` feature flag in `inferno-codegen` (M3+).
- **`inferno-formats` must stay `#![forbid(unsafe_code)]`** and every parser
  read bounded — model files are untrusted input (see
  [docs/threat-model.md](docs/threat-model.md)). Touching parser code means
  running `mise run fuzz -- gguf_parse` / `-- safetensors_parse` locally.
- **`ModelDesc` is format-agnostic:** never let a downstream crate learn
  which file format a model came from.
- **Tensor shapes are row-major, outermost first** everywhere in inferno;
  GGUF stores dims reversed and the GGUF parser normalizes them on ingest.
- **Snapshots (insta):** review with `cargo insta review`; never blind-accept.
- Fixture files under `tests/fixtures/` and `fuzz/corpus/` are generated —
  regenerate with `cargo run -p inferno-formats --example gen_fixtures`,
  don't hand-edit.
```

`CLAUDE.md`:

```markdown
See [AGENTS.md](AGENTS.md).
```

- [ ] **Step 3: Write ARCHITECTURE.md**

```markdown
# Architecture

Data flow (v1): model file → `ModelDesc` → graph IR → target-aware plan →
LLVM codegen → cached native artifact → runtime executes it.

## Crates

Present (M0):

- `crates/inferno-formats` — GGUF + MLX/safetensors parsing into a
  format-agnostic `ModelDesc`. Deliberately dumb: no graph knowledge, no
  `unsafe`, every read bounded (untrusted input). Downstream code must not
  be able to tell which format a model came from — that's why hyperparams
  are normalized here and not in the graph builder.
- `cli` — the `inferno` binary. Thin; all real logic lives in library crates.

Planned (M1–M4, see the spec for details):

- `inferno-graph` — graph IR + per-architecture builders + the scalar
  reference interpreter that serves as the correctness oracle for all
  compiled code.
- `inferno-target` — `TargetDesc` (ISA, caches, topology): always an explicit
  input to planning/codegen, never re-probed downstream. A detected target
  and a named-profile target are the same struct — that equivalence is the
  future cross-compile interface.
- `inferno-plan` — fusion islands, weight-layout repacking, static memory plan.
- `inferno-kernels` — hand-tuned matmul microkernels behind a fixed C ABI,
  selected by symbol from generated code.
- `inferno-codegen` — loop IR → LLVM IR (inkwell); JIT + artifact cache.
  The only crate that links LLVM.
- `inferno-runtime` — KV cache, tokenizer, sampling, generation loop.
- `inferno-core` — the embeddable public API.

## Boundary rules that aren't visible in the code

- Quantization formats are dtypes, not ops: dequant is always fused into the
  consuming kernel, so no crate ever materializes a dequantized weight tensor.
- Shapes are row-major outermost-first everywhere; only the GGUF parser knows
  GGUF stores them reversed.
- `fuzz/` is excluded from the workspace so nightly-only deps can't infect
  the stable build.
```

- [ ] **Step 4: Write docs/threat-model.md**

```markdown
# Threat model — inferno v1

**Assets:** the user's machine (inference runs locally with the user's
privileges) and the integrity of inference results.

**Adversary:** whoever produced a model file the user downloads. Model files
(GGUF, safetensors, config.json) are **untrusted input** — the primary
boundary. A malicious file must at worst produce a clean error, never memory
corruption, panic-DoS from a tiny crafted input, or resource exhaustion from
lying headers.

**Controls at this boundary** (`crates/inferno-formats`):
- `#![forbid(unsafe_code)]` in the parsing crate.
- Bounded reads everywhere: explicit limits (`src/limits.rs`) on string
  lengths, tensor/KV counts, array sizes and nesting depth, header sizes; no
  allocation sized by an unvalidated length (arrays parse element-by-element
  so a lying count hits EOF cheaply).
- Checked arithmetic on all attacker-controlled sizes and offsets;
  offset/length consistency validated (safetensors spans vs dtype size,
  GGUF offset alignment).
- Continuous fuzzing of both parsers (nightly CI, `fuzz/`): any panic is a
  bug by definition.

**Second boundary: the artifact cache** (`~/.cache/inferno/`, from M3).
Compiled artifacts are native code loaded via `dlopen` — running a cached
artifact is running code from that directory. Artifacts are looked up and
verified by content hash (model × target × inferno version); a hash mismatch
means recompile, never load. The cache directory has user-only permissions.

**Explicitly out of scope (v1):**
- Sandboxing generated code against a hostile *local* user (same trust
  domain as the inferno process itself).
- Network boundaries — v1 has no server and makes no network calls.
- Tokenizer-content attacks on downstream systems (prompt/output content is
  the embedding application's concern).

**Supply chain:** `cargo audit` + `cargo deny` gate CI and run on a weekly
schedule; `Cargo.lock` committed; `gitleaks` on pre-commit and CI.

Revisit when: the HTTP server lands (v2), AOT artifact *distribution* lands
(signing story needed), or any parser gains `unsafe`.
```

- [ ] **Step 5: Verify docs against reality, then commit**

Run each command the README/AGENTS.md name, exactly as written there:

```bash
mise tasks && mise run test && cargo run -p inferno -- inspect crates/inferno-formats/tests/fixtures/tiny.gguf
```

Expected: all succeed (this is the "onboarding you can actually run" check; CI repeats it from a clean clone in Task 12).

```bash
git add -A && git commit -m "docs: repo front door (README, AGENTS, ARCHITECTURE) and threat model"
```

---

### Task 12: CI — blocking tier, nightly tier, scanners, onboarding job

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/nightly.yml`

**Interfaces:**
- Consumes: mise task names (Task 1), fuzz target names (Task 10), README quickstart (Task 11)
- Produces: the blocking gate (`ci.yml`) every PR must pass; scheduled deep checks (`nightly.yml`).

- [ ] **Step 1: Write the blocking workflow** (`.github/workflows/ci.yml`)

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

# Blocking tier — budget ≤ 5 min wall-clock (spec §Testing).
jobs:
  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jdx/mise-action@v2 # installs mise.toml pins, incl. rust
      - uses: Swatinem/rust-cache@v2
      - run: mise run lint
      - run: mise run test
      - run: mise run audit
```

- [ ] **Step 2: Write the nightly workflow** (`.github/workflows/nightly.yml`)

```yaml
name: nightly
on:
  schedule:
    - cron: "0 3 * * *"
  workflow_dispatch:

jobs:
  fuzz:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target: [gguf_parse, safetensors_parse]
    steps:
      - uses: actions/checkout@v4
      - uses: jdx/mise-action@v2
      - uses: Swatinem/rust-cache@v2
      - run: mise exec rust@nightly -- rustc --version
      - run: mise exec rust@nightly -- cargo fuzz run --fuzz-dir fuzz ${{ matrix.target }} -- -max_total_time=600
      - uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: fuzz-crashes-${{ matrix.target }}
          path: fuzz/artifacts/

  semgrep:
    runs-on: ubuntu-latest
    container: semgrep/semgrep
    steps:
      - uses: actions/checkout@v4
      - run: semgrep scan --config p/rust --error

  mutants:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jdx/mise-action@v2
      - uses: Swatinem/rust-cache@v2
      - run: mise use --pin "cargo:cargo-mutants@latest" && mise install
      - run: cargo mutants --package inferno-formats --timeout 60 --in-place
        continue-on-error: true # advisory: audits assertion strength, doesn't gate

  # Weekly-fresh supply-chain scan (new CVEs land on old code).
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jdx/mise-action@v2
      - run: mise run audit

  # Onboarding-you-can-actually-run: the README quickstart from a bare clone.
  onboarding:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: DeterminateSystems/nix-installer-action@v16
      - uses: DeterminateSystems/magic-nix-cache-action@v9
      - run: nix profile install nixpkgs#devenv
      - uses: jdx/mise-action@v2
      - run: devenv shell -- bash -c 'llvm-config --version && command -v llama-cli'
      - run: mise run test
      - run: cargo run -p inferno -- inspect crates/inferno-formats/tests/fixtures/tiny.gguf
```

- [ ] **Step 3: Validate workflow syntax locally**

```bash
mise exec "aqua:rhysd/actionlint@latest" -- actionlint .github/workflows/*.yml
```

Expected: no output (clean). If actionlint flags a real error, fix it; do not pin actionlint into mise.toml (one-off tool).

- [ ] **Step 4: Commit and verify on GitHub**

```bash
git add .github && git commit -m "ci: blocking tier (lint/test/audit/gitleaks) + nightly tier (fuzz/semgrep/mutants/onboarding)"
git push origin main
gh run watch --exit-status || gh run list --limit 5
gh workflow run nightly && gh run watch --exit-status
```

Expected: the `ci` workflow goes green well inside the 5-minute budget; the manually-dispatched `nightly` run goes green (fuzz jobs take ~10 min; onboarding job proves clone-to-test works on a fresh machine). If a job fails, fix it now — a red nightly on day one defeats its purpose.

---

## Verification (whole-milestone)

After all tasks: `mise run lint && mise run test-full && mise run audit` all green; `inferno inspect` works on both fixture formats; both fuzz targets smoke-ran ≥ 60s crash-free; CI green on both workflows including onboarding; all docs commands are copy-paste runnable.

## Self-Review Notes

- Spec coverage: workspace ✅(T1) devenv/LLVM ✅(T2) formats+ModelDesc ✅(T3–8) fuzz ✅(T10) CLI ✅(T9) front door ✅(T11) threat model ✅(T11) CI tiers/scanners/onboarding ✅(T12). Tokenizer, graph IR, kernels: deliberately M1+ (per spec milestones).
- `predicates` dependency introduced in Task 9 Step 1 with explicit workspace wiring.
- Type consistency: `gguf::parse` leaves `weight_files` empty and `load_desc` fills it (T5 ↔ T8 documented both sides); `safetensors::parse` returns `(Vec<TensorDesc>, u64)` consumed identically in T7 and fuzz T10.
```
