# M1 — Graph IR + Interpreter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** First tokens out — a Llama-family graph builder, a scalar reference interpreter over all five v1 dtypes, native + HF tokenizers, greedy sampling, a real `inferno run`, and a nightly teacher-forced differential against llama.cpp.

**Architecture:** `inferno-formats` gains canonical tensor names, a `TokenizerSpec`, a bounded tensor-data reader, and scalar quant codecs. New crate `inferno-graph` holds the graph IR, the data-driven builder, the scalar interpreter, and the per-dtype tolerances. New crate `inferno-runtime` holds the `Tokenizer` trait (native BPE/SPM + `tokenizers`-crate wrapper), greedy `Sampler`, `Generator` loop, and the teacher-forced diff. The CLI gains `run` and a hidden `diff`.

**Tech Stack:** Rust (edition 2024, mise-pinned 1.96.0), cargo workspace, `thiserror`, `insta`, `proptest` (new dev-dep), `tokenizers` (new, default-features off), `fancy-regex` (new; BPE pre-tokenizer patterns need lookahead), `assert_cmd`.

**Spec:** `docs/superpowers/specs/2026-07-05-m1-graph-interpreter-design.md`

## Global Constraints

- All workflows via mise tasks: `mise run test` / `lint` / `fmt` / `fuzz -- <target>`. CI runs the same names.
- `inferno-formats` stays `#![forbid(unsafe_code)]`; every read bounded; model files are untrusted input **even past parsing** — hyperparam-derived allocations are limit-checked, hostile values surface as typed errors, never panics.
- No new crates beyond `inferno-graph`, `inferno-runtime`. New deps limited to: `tokenizers` (default-features = false), `fancy-regex`, `proptest` (dev-only). Everything else uses existing workspace deps.
- Tensor shapes are row-major, outermost first, everywhere.
- Snapshots: `cargo insta review` — never blind-accept. Fixtures regenerate via `cargo run -p inferno-formats --example gen_fixtures` — never hand-edit files under `tests/fixtures/` or `fuzz/corpus/`.
- After touching parser code (Tasks 3–5): run `mise run fuzz -- gguf_parse` and `mise run fuzz -- safetensors_parse` locally.
- Per-dtype tolerances and the logit tie epsilon are defined **once** in `inferno-graph::tolerance` and imported everywhere.
- Out of scope for M1 (documented, not built): rope scaling (Llama-3.x long-context factors are ignored; short-context output is unaffected), chat templates, non-greedy sampling, mmap, `inferno-core`.
- Blocking tier stays ≤5 minutes and never touches the network.

## File Structure

```
crates/inferno-formats/src/
  quant.rs        (new)  scalar pack/dequant: F16, BF16, Q8_0, Q4_K
  data.rs         (new)  read_tensor_bytes: bounded raw-tensor reads
  names.rs        (new)  GGUF/HF → canonical tensor-name mapping
  desc.rs         (mod)  + TokenizerSpec, TokenizerKind, SpecialTokens, RopeStyle
  gguf/mod.rs     (mod)  canonical names, rope style, tokenizer extraction
  gguf/value.rs   (mod)  + as_bool, as_i32, array accessors
  mlx.rs          (mod)  canonical names, rope style, tokenizer.json detection
  safetensors.rs  (mod)  canonical names applied per-entry
  fixtures.rs     (mod)  real weights (all 5 dtypes), tokenizer metadata, tied embeddings
  examples/gen_fixtures.rs (mod) also writes mlx/tokenizer.json
crates/inferno-graph/src/
  lib.rs error.rs tolerance.rs ir.rs build.rs ops.rs interp.rs   (all new)
crates/inferno-graph/tests/
  quant_roundtrip.rs snapshot_ir.rs differential.rs              (all new)
crates/inferno-runtime/src/
  lib.rs error.rs sampler.rs generate.rs diff.rs                 (all new)
  tokenizer/mod.rs tokenizer/bytes.rs tokenizer/bpe.rs tokenizer/spm.rs tokenizer/hf.rs (all new)
cli/src/
  run.rs diff.rs  (new)   main.rs (mod)
scripts/nightly-differential.sh (new)
mise.toml (mod: differential task)  .github/workflows/nightly.yml (mod: differential job)
ARCHITECTURE.md AGENTS.md (mod)
```

Dependency graph: `inferno-runtime` → `inferno-graph` → `inferno-formats`. The quant *codecs* live in `inferno-formats` (it owns `DType` block layouts); their round-trip *property tests* live in `inferno-graph/tests/` because the tolerances live there and `inferno-formats` cannot depend on `inferno-graph`.

---

### Task 1: Scalar quant codecs in `inferno-formats`

**Files:**
- Create: `crates/inferno-formats/src/quant.rs`
- Modify: `crates/inferno-formats/src/lib.rs` (add `pub mod quant;`)

**Interfaces:**
- Consumes: `DType`, `FormatError`, `Result` from the crate root.
- Produces (used by Tasks 5, 9):
  - `quant::f16_to_f32(u16) -> f32`, `quant::f32_to_f16(f32) -> u16`
  - `quant::bf16_to_f32(u16) -> f32`, `quant::f32_to_bf16(f32) -> u16`
  - `quant::dequant(dtype: &DType, bytes: &[u8], n_elems: usize) -> Result<Vec<f32>>`
  - `quant::pack(dtype: &DType, values: &[f32]) -> Result<Vec<u8>>`
  - `pack` is a *simple reference quantizer* (per-block min/max), not ggml's optimizer; `dequant` implements the exact ggml block layouts so real model files decode correctly.

- [ ] **Step 1: Write failing unit tests** (bottom of the new `quant.rs`, `#[cfg(test)]`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn f16_known_vectors() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // f16 max
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8); // smallest subnormal
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF);
        assert_eq!(f32_to_f16(1e6), 0x7C00); // overflow → +inf
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn bf16_known_vectors() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC040), -3.0);
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        // RNE: 1.0039063 is exactly between 0x3F80 and 0x3F81 → even (0x3F80)
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn q8_0_roundtrip_exactish() {
        // 32 values in [-1, 1]; max abs error after roundtrip ≤ d/2 = amax/254.
        let vals: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) / 15.5).collect();
        let packed = pack(&DType::Q8_0, &vals).unwrap();
        assert_eq!(packed.len(), 34);
        let out = dequant(&DType::Q8_0, &packed, 32).unwrap();
        for (a, b) in vals.iter().zip(&out) {
            assert!((a - b).abs() <= 1.0 / 254.0 + 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn q4_k_roundtrip_block() {
        let vals: Vec<f32> = (0..256).map(|i| ((i * 37 % 256) as f32 / 128.0) - 1.0).collect();
        let packed = pack(&DType::Q4_K, &vals).unwrap();
        assert_eq!(packed.len(), 144);
        let out = dequant(&DType::Q4_K, &packed, 256).unwrap();
        // Simple min/max quantizer worst case: 4-bit step ≤ 2·amax/15 (half-
        // step error ~6.7% of amax) plus 6-bit scale quantization → 11%.
        let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (a, b) in vals.iter().zip(&out) {
            assert!((a - b).abs() <= 0.11 * amax, "{a} vs {b}");
        }
    }

    #[test]
    fn dequant_rejects_bad_lengths() {
        assert!(dequant(&DType::F32, &[0u8; 7], 2).is_err()); // 2 f32 = 8 bytes
        assert!(dequant(&DType::Q8_0, &[0u8; 34], 31).is_err()); // not block-aligned
        assert!(pack(&DType::Q4_K, &[0f32; 100]).is_err()); // not multiple of 256
        assert!(dequant(&DType::Unsupported("x".into()), &[], 0).is_err());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p inferno-formats quant`
Expected: compile error — `quant` module does not exist.

- [ ] **Step 3: Implement `quant.rs`**

```rust
//! Scalar quant codecs: the reference implementations of the v1 dtypes.
//! `dequant` follows the exact ggml block layouts (so real files decode);
//! `pack` is a *simple* min/max reference quantizer for fixtures and tests,
//! not ggml's error-minimizing quantizer. Both are the semantic ground truth
//! that M2 kernels are property-tested against.

use crate::{DType, FormatError, Result};

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15);
    let exp = u32::from((h >> 10) & 0x1F);
    let man = u32::from(h & 0x3FF);
    let bits = match (exp, man) {
        (0, 0) => sign << 31,
        (0, mut m) => {
            // Subnormal: renormalize into f32.
            let mut e: i32 = 113; // 127 - 15 + 1
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3FF) << 13)
        }
        (0x1F, 0) => (sign << 31) | 0x7F80_0000,
        (0x1F, m) => (sign << 31) | 0x7F80_0000 | (m << 13),
        (e, m) => (sign << 31) | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7F_FFFF;
    if exp == 0xFF {
        // Inf/NaN; keep NaN-ness with a quiet bit.
        return sign | 0x7C00 | u16::from(man != 0) << 9;
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00; // overflow → inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow → signed zero
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let midpoint = 1u32 << (shift - 1);
        let round = u32::from(rem > midpoint || (rem == midpoint && half & 1 == 1));
        return sign | (half + round) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    let round = u32::from(rem > 0x1000 || (rem == 0x1000 && half & 1 == 1));
    sign | (half + round) as u16 // rounding carry correctly bumps the exponent
}

pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040; // quiet NaN
    }
    let half = bits >> 16;
    let rem = bits & 0xFFFF;
    let round = u32::from(rem > 0x8000 || (rem == 0x8000 && half & 1 == 1));
    (half + round) as u16
}

fn bad(detail: String) -> FormatError {
    FormatError::Malformed { context: "quant data", detail }
}

/// ggml Q4_K scale/min extraction: 8 six-bit (scale, min) pairs in 12 bytes.
fn get_scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}

pub fn dequant(dtype: &DType, bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    let expected = dtype
        .byte_len(n_elems as u64)
        .ok_or_else(|| bad(format!("{dtype:?}: {n_elems} elements not representable")))?;
    if bytes.len() as u64 != expected {
        return Err(bad(format!(
            "{dtype:?}: got {} bytes, expected {expected} for {n_elems} elements",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(n_elems);
    match dtype {
        DType::F32 => {
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
        }
        DType::F16 => {
            for c in bytes.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())));
            }
        }
        DType::BF16 => {
            for c in bytes.chunks_exact(2) {
                out.push(bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())));
            }
        }
        DType::Q8_0 => {
            // 34-byte block: f16 scale d, then 32 × i8. y = d * q.
            for block in bytes.chunks_exact(34) {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for &q in &block[2..34] {
                    out.push(d * f32::from(q as i8));
                }
            }
        }
        DType::Q4_K => {
            // 144-byte super-block: f16 d, f16 dmin, 12 bytes of 6-bit
            // (scale, min) pairs, 128 bytes of 4-bit quants (256 elements).
            // y = d*sc*q - dmin*m, in chunks of 64 (32 low nibbles then 32 high).
            for block in bytes.chunks_exact(144) {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales = &block[4..16];
                let qs = &block[16..144];
                let mut is = 0;
                let mut qoff = 0;
                for _ in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let (d1, min1) = (d * f32::from(sc1), dmin * f32::from(m1));
                    let (d2, min2) = (d * f32::from(sc2), dmin * f32::from(m2));
                    for l in 0..32 {
                        out.push(d1 * f32::from(qs[qoff + l] & 0xF) - min1);
                    }
                    for l in 0..32 {
                        out.push(d2 * f32::from(qs[qoff + l] >> 4) - min2);
                    }
                    qoff += 32;
                    is += 2;
                }
            }
        }
        DType::Unsupported(s) => return Err(bad(format!("unsupported dtype {s}"))),
    }
    Ok(out)
}

pub fn pack(dtype: &DType, values: &[f32]) -> Result<Vec<u8>> {
    let expected = dtype
        .byte_len(values.len() as u64)
        .ok_or_else(|| bad(format!("{dtype:?}: {} elements not packable", values.len())))?;
    let mut out = Vec::with_capacity(expected as usize);
    match dtype {
        DType::F32 => {
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DType::F16 => {
            for v in values {
                out.extend_from_slice(&f32_to_f16(*v).to_le_bytes());
            }
        }
        DType::BF16 => {
            for v in values {
                out.extend_from_slice(&f32_to_bf16(*v).to_le_bytes());
            }
        }
        DType::Q8_0 => {
            for block in values.chunks_exact(32) {
                let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let dh = f32_to_f16(d);
                out.extend_from_slice(&dh.to_le_bytes());
                let d = f16_to_f32(dh); // quantize against the stored scale
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for v in block {
                    out.push((v * inv).round().clamp(-127.0, 127.0) as i8 as u8);
                }
            }
        }
        DType::Q4_K => {
            for sb in values.chunks_exact(256) {
                // Per 32-elem sub-block: value = d*sc*q - dmin*m with q ∈ 0..=15.
                let mut effs = [0f32; 8]; // effective scale per sub-block
                let mut mins = [0f32; 8]; // effective (positive) min offset
                for (j, blk) in sb.chunks_exact(32).enumerate() {
                    let mn = blk.iter().fold(f32::INFINITY, |m, v| m.min(*v));
                    let mx = blk.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
                    mins[j] = (-mn).max(0.0);
                    effs[j] = (mx + mins[j]).max(0.0) / 15.0;
                }
                let dsup = effs.iter().fold(0f32, |m, v| m.max(*v)) / 63.0;
                let msup = mins.iter().fold(0f32, |m, v| m.max(*v)) / 63.0;
                let dh = f32_to_f16(dsup);
                let mh = f32_to_f16(msup);
                out.extend_from_slice(&dh.to_le_bytes());
                out.extend_from_slice(&mh.to_le_bytes());
                let (dsup, msup) = (f16_to_f32(dh), f16_to_f32(mh));
                let q6 = |x: f32, s: f32| -> u8 {
                    if s > 0.0 { (x / s).round().clamp(0.0, 63.0) as u8 } else { 0 }
                };
                let lsc: Vec<u8> = effs.iter().map(|&e| q6(e, dsup)).collect();
                let lm: Vec<u8> = mins.iter().map(|&m| q6(m, msup)).collect();
                // Inverse of get_scale_min_k4.
                let mut scales = [0u8; 12];
                for j in 0..4 {
                    scales[j] = lsc[j];
                    scales[j + 4] = lm[j];
                }
                for j in 4..8 {
                    scales[j + 4] = (lsc[j] & 0xF) | ((lm[j] & 0xF) << 4);
                    scales[j - 4] |= (lsc[j] >> 4) << 6;
                    scales[j] |= (lm[j] >> 4) << 6;
                }
                out.extend_from_slice(&scales);
                // Quantize elements, packing nibbles in ggml's chunk-of-64 order.
                let quant = |j: usize, x: f32| -> u8 {
                    let sc = dsup * f32::from(lsc[j]);
                    let m = msup * f32::from(lm[j]);
                    if sc > 0.0 { ((x + m) / sc).round().clamp(0.0, 15.0) as u8 } else { 0 }
                };
                for pair in 0..4 {
                    let (j1, j2) = (pair * 2, pair * 2 + 1);
                    for l in 0..32 {
                        let lo = quant(j1, sb[j1 * 32 + l]);
                        let hi = quant(j2, sb[j2 * 32 + l]);
                        out.push(lo | (hi << 4));
                    }
                }
            }
        }
        DType::Unsupported(s) => return Err(bad(format!("unsupported dtype {s}"))),
    }
    Ok(out)
}
```

In `lib.rs` add after `pub mod limits;`:

```rust
pub mod quant;
```

- [ ] **Step 4: Run tests**

Run: `cargo nextest run -p inferno-formats quant`
Expected: all 5 tests PASS.

- [ ] **Step 5: Lint and commit**

```bash
mise run lint
git add crates/inferno-formats/src/quant.rs crates/inferno-formats/src/lib.rs
git commit -m "feat(formats): scalar quant codecs for F16/BF16/Q8_0/Q4_K"
```

---

### Task 2: Bounded tensor-data reader

**Files:**
- Create: `crates/inferno-formats/src/data.rs`
- Modify: `crates/inferno-formats/src/lib.rs` (add `mod data;` + re-export)

**Interfaces:**
- Consumes: `ModelDesc`, `TensorDesc`, `FormatError`.
- Produces (used by Task 9's interpreter): `pub fn read_tensor_bytes(desc: &ModelDesc, tensor: &TensorDesc) -> Result<Vec<u8>>` — reads the tensor's raw bytes from its weight file, every bound checked against the actual file length.

- [ ] **Step 1: Write failing test** (`#[cfg(test)]` in `data.rs`; uses the existing zero-filled fixture — real weights arrive in Task 5)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_desc;
    use std::path::Path;

    fn fixture_gguf() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.gguf")
    }

    #[test]
    fn reads_full_tensor_span() {
        let desc = load_desc(&fixture_gguf()).unwrap();
        let t = &desc.tensors[0];
        let bytes = read_tensor_bytes(&desc, t).unwrap();
        assert_eq!(bytes.len() as u64, t.data_len.unwrap());
    }

    #[test]
    fn rejects_out_of_range_offset() {
        let desc = load_desc(&fixture_gguf()).unwrap();
        let mut t = desc.tensors[0].clone();
        t.data_offset = u64::MAX - 4; // hostile header value
        assert!(read_tensor_bytes(&desc, &t).is_err());
    }

    #[test]
    fn rejects_bad_file_index() {
        let desc = load_desc(&fixture_gguf()).unwrap();
        let mut t = desc.tensors[0].clone();
        t.file_index = 7;
        assert!(read_tensor_bytes(&desc, &t).is_err());
    }

    #[test]
    fn rejects_unknown_data_len() {
        let desc = load_desc(&fixture_gguf()).unwrap();
        let mut t = desc.tensors[0].clone();
        t.data_len = None;
        assert!(read_tensor_bytes(&desc, &t).is_err());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo nextest run -p inferno-formats data`
Expected: compile error — module does not exist.

- [ ] **Step 3: Implement `data.rs`**

```rust
//! Raw tensor-byte access. Offsets and lengths come from an untrusted header,
//! so every value is validated against the real file length before reading.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::{FormatError, ModelDesc, Result, TensorDesc};

pub fn read_tensor_bytes(desc: &ModelDesc, tensor: &TensorDesc) -> Result<Vec<u8>> {
    let malformed = |detail: String| FormatError::Malformed { context: "tensor data", detail };
    let idx = tensor.file_index as usize;
    let (path, base) = match (desc.weight_files.get(idx), desc.data_section_offsets.get(idx)) {
        (Some(p), Some(b)) => (p, *b),
        _ => return Err(malformed(format!("{}: file index {idx} out of range", tensor.name))),
    };
    let len = tensor
        .data_len
        .ok_or_else(|| malformed(format!("{}: unknown data length", tensor.name)))?;
    let start = base
        .checked_add(tensor.data_offset)
        .ok_or_else(|| malformed(format!("{}: offset overflow", tensor.name)))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| malformed(format!("{}: length overflow", tensor.name)))?;

    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    if end > file_len {
        return Err(malformed(format!(
            "{}: span {start}..{end} exceeds file length {file_len}",
            tensor.name
        )));
    }
    file.seek(SeekFrom::Start(start))?;
    let mut buf = vec![0u8; len as usize];
    file.read_exact(&mut buf)?;
    Ok(buf)
}
```

In `lib.rs`: add `mod data;` next to the other modules and `pub use data::read_tensor_bytes;` next to the other re-exports.

- [ ] **Step 4: Run tests**

Run: `cargo nextest run -p inferno-formats data`
Expected: 4 tests PASS.

Note: `vec![0u8; len]` allocates from a header-derived value, but only after `end <= file_len` — allocation is bounded by the actual file size, which matches the threat model's "allocation limits derived from file size".

- [ ] **Step 5: Lint and commit**

```bash
mise run lint
git add crates/inferno-formats/src/data.rs crates/inferno-formats/src/lib.rs
git commit -m "feat(formats): bounded raw tensor-byte reader"
```

---

### Task 3: Canonical tensor naming

**Files:**
- Create: `crates/inferno-formats/src/names.rs`
- Modify: `crates/inferno-formats/src/gguf/mod.rs` (rename tensors after parsing)
- Modify: `crates/inferno-formats/src/mlx.rs` and `crates/inferno-formats/src/safetensors.rs` (rename per entry)
- Modify: existing snapshots via `cargo insta review`

**Interfaces:**
- Produces (used by Task 7's builder, which looks tensors up **only** by these names):

```
token_embed.weight
output_norm.weight
lm_head.weight                       (absent when embeddings are tied)
layers.{i}.attn_norm.weight
layers.{i}.attn.{q,k,v}_proj.{weight,bias}
layers.{i}.attn.o_proj.weight
layers.{i}.attn.{q,k}_norm.weight    (Qwen3)
layers.{i}.ffn_norm.weight
layers.{i}.ffn.{gate,up,down}_proj.weight
```

- `pub(crate) fn names::canonical_gguf(raw: &str) -> Option<String>`
- `pub(crate) fn names::canonical_hf(raw: &str) -> Option<String>`
- Unmapped names pass through unchanged (`None` → keep raw); the builder ignores them.

- [ ] **Step 1: Write failing tests** (`#[cfg(test)]` in `names.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_names_map() {
        assert_eq!(canonical_gguf("token_embd.weight").as_deref(), Some("token_embed.weight"));
        assert_eq!(canonical_gguf("output.weight").as_deref(), Some("lm_head.weight"));
        assert_eq!(canonical_gguf("output_norm.weight").as_deref(), Some("output_norm.weight"));
        assert_eq!(
            canonical_gguf("blk.3.attn_q.weight").as_deref(),
            Some("layers.3.attn.q_proj.weight")
        );
        assert_eq!(
            canonical_gguf("blk.0.attn_k.bias").as_deref(),
            Some("layers.0.attn.k_proj.bias")
        );
        assert_eq!(
            canonical_gguf("blk.12.ffn_down.weight").as_deref(),
            Some("layers.12.ffn.down_proj.weight")
        );
        assert_eq!(
            canonical_gguf("blk.0.attn_q_norm.weight").as_deref(),
            Some("layers.0.attn.q_norm.weight")
        );
        assert_eq!(canonical_gguf("rope_freqs.weight"), None); // unmapped → raw
    }

    #[test]
    fn hf_names_map() {
        assert_eq!(
            canonical_hf("model.embed_tokens.weight").as_deref(),
            Some("token_embed.weight")
        );
        assert_eq!(canonical_hf("model.norm.weight").as_deref(), Some("output_norm.weight"));
        assert_eq!(canonical_hf("lm_head.weight").as_deref(), Some("lm_head.weight"));
        assert_eq!(
            canonical_hf("model.layers.3.self_attn.q_proj.weight").as_deref(),
            Some("layers.3.attn.q_proj.weight")
        );
        assert_eq!(
            canonical_hf("model.layers.0.input_layernorm.weight").as_deref(),
            Some("layers.0.attn_norm.weight")
        );
        assert_eq!(
            canonical_hf("model.layers.0.post_attention_layernorm.weight").as_deref(),
            Some("layers.0.ffn_norm.weight")
        );
        assert_eq!(
            canonical_hf("model.layers.1.mlp.gate_proj.weight").as_deref(),
            Some("layers.1.ffn.gate_proj.weight")
        );
        assert_eq!(
            canonical_hf("model.layers.1.self_attn.q_norm.weight").as_deref(),
            Some("layers.1.attn.q_norm.weight")
        );
        assert_eq!(canonical_hf("model.rotary_emb.inv_freq"), None);
    }
}
```

- [ ] **Step 2: Run to verify failure** — `cargo nextest run -p inferno-formats names` → compile error.

- [ ] **Step 3: Implement `names.rs`**

```rust
//! Canonical tensor names. Parsers map format-specific names to one scheme at
//! the edge so nothing downstream can tell which file format a model came
//! from (ARCHITECTURE.md boundary rule). Unmapped names pass through raw.

/// GGUF suffix (after `blk.{i}.`) → canonical suffix (after `layers.{i}.`).
const GGUF_LAYER: &[(&str, &str)] = &[
    ("attn_norm", "attn_norm"),
    ("attn_q_norm", "attn.q_norm"),
    ("attn_k_norm", "attn.k_norm"),
    ("attn_q", "attn.q_proj"),
    ("attn_k", "attn.k_proj"),
    ("attn_v", "attn.v_proj"),
    ("attn_output", "attn.o_proj"),
    ("ffn_norm", "ffn_norm"),
    ("ffn_gate", "ffn.gate_proj"),
    ("ffn_up", "ffn.up_proj"),
    ("ffn_down", "ffn.down_proj"),
];

/// HF infix (after `model.layers.{i}.`) → canonical suffix.
const HF_LAYER: &[(&str, &str)] = &[
    ("input_layernorm", "attn_norm"),
    ("self_attn.q_norm", "attn.q_norm"),
    ("self_attn.k_norm", "attn.k_norm"),
    ("self_attn.q_proj", "attn.q_proj"),
    ("self_attn.k_proj", "attn.k_proj"),
    ("self_attn.v_proj", "attn.v_proj"),
    ("self_attn.o_proj", "attn.o_proj"),
    ("post_attention_layernorm", "ffn_norm"),
    ("mlp.gate_proj", "ffn.gate_proj"),
    ("mlp.up_proj", "ffn.up_proj"),
    ("mlp.down_proj", "ffn.down_proj"),
];

/// Split "name.weight" / "name.bias" → (name, param). Longest-match tables
/// above are ordered so prefixes (attn_q vs attn_q_norm) resolve correctly.
fn split_param(raw: &str) -> Option<(&str, &str)> {
    raw.rsplit_once('.')
        .filter(|(_, p)| *p == "weight" || *p == "bias")
}

fn map_layer(table: &[(&str, &str)], stem: &str) -> Option<&'static str> {
    table.iter().find(|(from, _)| *from == stem).map(|(_, to)| *to)
}

pub(crate) fn canonical_gguf(raw: &str) -> Option<String> {
    let (stem, param) = split_param(raw)?;
    match stem {
        "token_embd" => return Some(format!("token_embed.{param}")),
        "output" => return Some(format!("lm_head.{param}")),
        "output_norm" => return Some(format!("output_norm.{param}")),
        _ => {}
    }
    let rest = stem.strip_prefix("blk.")?;
    let (idx, suffix) = rest.split_once('.')?;
    let idx: u64 = idx.parse().ok()?;
    let mapped = map_layer(GGUF_LAYER, suffix)?;
    Some(format!("layers.{idx}.{mapped}.{param}"))
}

pub(crate) fn canonical_hf(raw: &str) -> Option<String> {
    let (stem, param) = split_param(raw)?;
    match stem {
        "model.embed_tokens" => return Some(format!("token_embed.{param}")),
        "model.norm" => return Some(format!("output_norm.{param}")),
        "lm_head" => return Some(format!("lm_head.{param}")),
        _ => {}
    }
    let rest = stem.strip_prefix("model.layers.")?;
    let (idx, suffix) = rest.split_once('.')?;
    let idx: u64 = idx.parse().ok()?;
    let mapped = map_layer(HF_LAYER, suffix)?;
    Some(format!("layers.{idx}.{mapped}.{param}"))
}
```

Add `mod names;` to `lib.rs`.

Wire into **`gguf/mod.rs`** — in `parse()`, immediately after the tensor-info loop completes (before the alignment check):

```rust
    for t in &mut tensors {
        if let Some(canon) = crate::names::canonical_gguf(&t.name) {
            t.name = canon;
        }
    }
```

Note: the existing `vocab_size` fallback in `extract_hyperparams` looks up `token_embd.weight` — update that string to `token_embed.weight` (the rename happens before extraction is called).

Wire into **`safetensors.rs`** — where each `TensorDesc` is built from a JSON entry, canonicalize:

```rust
        let name = crate::names::canonical_hf(&name).unwrap_or(name);
```

(Adapt to the local variable: apply just before constructing `TensorDesc`.)

- [ ] **Step 4: Run the full formats suite, review snapshots**

Run: `cargo nextest run -p inferno-formats`
Expected: `names` tests pass; snapshot tests (`snapshot_desc`) FAIL because tensor names changed. Then:

Run: `cargo insta review` — verify the only change is names becoming canonical (`blk.0.attn_q.weight` → `layers.0.attn.q_proj.weight`, `model.embed_tokens.weight` → `token_embed.weight`, etc.), accept. Also fix the `parses_tiny_llama` unit test assertions in `gguf/mod.rs` (`token_embd.weight` → `token_embed.weight`) and any name assertions in `mlx.rs`/`safetensors.rs` tests.

Run: `cargo nextest run -p inferno-formats` again — all PASS.

- [ ] **Step 5: Fuzz, lint, commit**

```bash
mise run fuzz -- gguf_parse
mise run fuzz -- safetensors_parse
mise run lint
git add -A crates/inferno-formats
git commit -m "feat(formats): canonical tensor names at the parser edge"
```

---

### Task 4: `TokenizerSpec` and `RopeStyle` in `ModelDesc`

**Files:**
- Modify: `crates/inferno-formats/src/desc.rs` (new types + fields)
- Modify: `crates/inferno-formats/src/gguf/value.rs` (accessors)
- Modify: `crates/inferno-formats/src/gguf/mod.rs` (extract tokenizer + rope style)
- Modify: `crates/inferno-formats/src/mlx.rs` (tokenizer.json path + rope style)
- Modify: `crates/inferno-formats/src/fixtures.rs` (only: add the two new fields where `HyperParams`/`ModelDesc` are constructed, to keep compiling; real tokenizer fixtures are Task 5)

**Interfaces:**
- Produces (consumed by Tasks 7, 10):

```rust
// desc.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RopeStyle {
    /// GGML "NORM": rotate adjacent pairs (x[2i], x[2i+1]). GGUF llama-arch
    /// files use this because conversion permutes Q/K weight rows.
    Interleaved,
    /// GPT-NeoX / HF: rotate half-split pairs (x[i], x[i+head_dim/2]).
    HalfSplit,
}
// HyperParams gains: pub rope_style: RopeStyle,

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind { Bpe, Spm }

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpecialTokens { pub bos: Option<u32>, pub eos: Option<u32> }

#[derive(Debug, Clone, PartialEq)]
pub enum TokenizerSpec {
    Embedded {
        kind: TokenizerKind,
        tokens: Vec<String>,
        scores: Vec<f32>,       // empty for BPE
        token_types: Vec<i32>,  // GGUF type ids (1=normal 3=control 6=byte); empty if absent
        merges: Vec<String>,    // "left right" pairs; empty for SPM
        pre: Option<String>,    // GGUF pre-tokenizer id, e.g. "qwen2"
        special: SpecialTokens,
        add_bos: bool,
    },
    HfJson { path: PathBuf },
}
// ModelDesc gains: #[serde(skip)] pub tokenizer: Option<TokenizerSpec>,
```

- `GgufValue` gains: `as_bool() -> Option<bool>`, `as_i64() -> Option<i64>`, `as_array() -> Option<&[GgufValue]>`.

- [ ] **Step 1: Write failing tests**

In `gguf/mod.rs` tests (uses the Task-5 fixture builder once it exists; for now hand-build metadata — this test constructs a GGUF with tokenizer keys):

```rust
    #[test]
    fn extracts_bpe_tokenizer_spec() {
        // fixtures::tiny_llama_gguf() gains tokenizer keys in Task 5; until
        // then, hand-assemble a minimal GGUF with the fixture KV helpers.
        use crate::desc::{TokenizerKind, TokenizerSpec};
        use crate::fixtures::{put_kv_str, put_kv_str_array, put_kv_u32};
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // tensors
        out.extend_from_slice(&8u64.to_le_bytes()); // kv count
        put_kv_str(&mut out, "general.architecture", "llama");
        put_kv_u32(&mut out, "llama.block_count", 1);
        put_kv_u32(&mut out, "llama.embedding_length", 8);
        put_kv_u32(&mut out, "llama.attention.head_count", 2);
        put_kv_u32(&mut out, "llama.feed_forward_length", 16);
        put_kv_str(&mut out, "tokenizer.ggml.model", "gpt2");
        put_kv_str_array(&mut out, "tokenizer.ggml.tokens", &["a".into(), "b".into()]);
        put_kv_str_array(&mut out, "tokenizer.ggml.merges", &["a b".into()]);
        let desc = parse(&mut Cursor::new(&out)).unwrap();
        let Some(TokenizerSpec::Embedded { kind, tokens, merges, add_bos, .. }) = desc.tokenizer
        else {
            panic!("expected embedded tokenizer");
        };
        assert_eq!(kind, TokenizerKind::Bpe);
        assert_eq!(tokens, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(merges, vec!["a b".to_string()]);
        assert!(!add_bos); // BPE default when key absent
    }

    #[test]
    fn rope_style_by_architecture() {
        // llama-arch GGUF → Interleaved (conversion permutes Q/K).
        let desc = parse(&mut Cursor::new(&fixtures::tiny_llama_gguf())).unwrap();
        assert_eq!(desc.hyperparams.rope_style, crate::RopeStyle::Interleaved);
    }
```

This test needs two `fixtures.rs` changes **in this task** (Task 5 builds on them): change the existing `put_str`/`put_kv_u32`/`put_kv_f32`/`put_kv_str` helpers from private to `pub(crate)`, and add:

```rust
pub(crate) fn put_kv_str_array(out: &mut Vec<u8>, key: &str, items: &[String]) {
    put_str(out, key);
    out.extend_from_slice(&9u32.to_le_bytes()); // array
    out.extend_from_slice(&8u32.to_le_bytes()); // elem: string
    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for s in items {
        put_str(out, s);
    }
}
```

In `mlx.rs` tests:

```rust
    #[test]
    fn detects_tokenizer_json_and_halfsplit_rope() {
        let dir = write_tiny_mlx_dir();
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        let desc = load_dir(dir.path()).unwrap();
        assert_eq!(desc.hyperparams.rope_style, crate::RopeStyle::HalfSplit);
        match desc.tokenizer {
            Some(crate::desc::TokenizerSpec::HfJson { path }) => {
                assert_eq!(path, dir.path().join("tokenizer.json"));
            }
            other => panic!("expected HfJson, got {other:?}"),
        }
    }

    #[test]
    fn no_tokenizer_json_means_none() {
        let dir = write_tiny_mlx_dir();
        assert!(load_dir(dir.path()).unwrap().tokenizer.is_none());
    }
```

- [ ] **Step 2: Run to verify failure** — `cargo nextest run -p inferno-formats` → compile errors (missing types).

- [ ] **Step 3: Implement**

`desc.rs`: add the types exactly as in **Interfaces** above (plus `use std::path::PathBuf;` already present). Add fields:

```rust
// in HyperParams (derives stay as-is; RopeStyle derives Serialize):
    pub rope_style: RopeStyle,
// in ModelDesc:
    /// Not serialized: vocab-scale payload would bloat snapshots.
    #[serde(skip)]
    pub tokenizer: Option<TokenizerSpec>,
```

`gguf/value.rs` accessors:

```rust
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::U8(v) => Some(v.into()),
            Self::U16(v) => Some(v.into()),
            Self::U32(v) => Some(v.into()),
            Self::U64(v) => i64::try_from(v).ok(),
            Self::I8(v) => Some(v.into()),
            Self::I16(v) => Some(v.into()),
            Self::I32(v) => Some(v.into()),
            Self::I64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
```

`gguf/mod.rs` — add extraction (called from `parse()` right before `Ok(ModelDesc {...})`, passing `&meta`; store result in the new `tokenizer` field):

```rust
fn extract_tokenizer(meta: &BTreeMap<String, GgufValue>) -> Option<crate::desc::TokenizerSpec> {
    use crate::desc::{SpecialTokens, TokenizerKind, TokenizerSpec};
    let kind = match meta.get("tokenizer.ggml.model").and_then(GgufValue::as_str)? {
        "gpt2" => TokenizerKind::Bpe,
        "llama" => TokenizerKind::Spm,
        _ => return None, // unsupported tokenizer family → model parses, can't run
    };
    let str_array = |key: &str| -> Vec<String> {
        meta.get(key)
            .and_then(GgufValue::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default()
    };
    let tokens = str_array("tokenizer.ggml.tokens");
    if tokens.is_empty() {
        return None;
    }
    let scores = meta
        .get("tokenizer.ggml.scores")
        .and_then(GgufValue::as_array)
        .map(|a| a.iter().filter_map(GgufValue::as_f32).collect())
        .unwrap_or_default();
    let token_types = meta
        .get("tokenizer.ggml.token_type")
        .and_then(GgufValue::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_i64().map(|i| i as i32)).collect())
        .unwrap_or_default();
    let get_id = |key: &str| meta.get(key).and_then(GgufValue::as_u64).map(|v| v as u32);
    Some(TokenizerSpec::Embedded {
        kind,
        merges: str_array("tokenizer.ggml.merges"),
        pre: meta.get("tokenizer.ggml.pre").and_then(GgufValue::as_str).map(str::to_string),
        special: SpecialTokens {
            bos: get_id("tokenizer.ggml.bos_token_id"),
            eos: get_id("tokenizer.ggml.eos_token_id"),
        },
        add_bos: meta
            .get("tokenizer.ggml.add_bos_token")
            .and_then(GgufValue::as_bool)
            .unwrap_or(kind == TokenizerKind::Spm), // SPM models add BOS by default
        tokens,
        scores,
        token_types,
    })
}
```

Rope style in `extract_hyperparams` (add to the `HyperParams` construction):

```rust
            rope_style: match architecture {
                // Qwen2/Qwen3 GGUFs keep HF half-split layout; llama-arch
                // GGUFs (Llama, Mistral) had Q/K rows permuted at conversion.
                Architecture::Qwen2 | Architecture::Qwen3 => RopeStyle::HalfSplit,
                _ => RopeStyle::Interleaved,
            },
```

(`architecture` is computed before the struct literal; hoist it if needed. Import `RopeStyle` at the top.)

`mlx.rs` — in `load_dir`, add to `HyperParams`: `rope_style: crate::RopeStyle::HalfSplit,` and before `Ok(ModelDesc {...})`:

```rust
    let tokenizer_json = dir.join("tokenizer.json");
    let tokenizer = tokenizer_json
        .is_file()
        .then_some(crate::desc::TokenizerSpec::HfJson { path: tokenizer_json });
```

and `tokenizer,` in the struct. Also add `tokenizer: None,` where `gguf/mod.rs` first builds `ModelDesc` (then overwrite with `extract_tokenizer(&meta)` result — simplest: `tokenizer: extract_tokenizer(&meta),` directly in the literal).

`fixtures.rs`: add `rope_style: crate::RopeStyle::Interleaved,` to `tiny_hyperparams()`.

Re-export in `lib.rs`: `pub use desc::{…, RopeStyle, SpecialTokens, TokenizerKind, TokenizerSpec};`

- [ ] **Step 4: Run tests, review snapshots**

Run: `cargo nextest run -p inferno-formats`
Expected: new tests PASS; `ModelDesc` snapshots change (new `rope_style` field) → `cargo insta review`, verify only `rope_style` was added, accept. Re-run: all PASS.

- [ ] **Step 5: Fuzz, lint, commit**

```bash
mise run fuzz -- gguf_parse
mise run lint
git add -A crates/inferno-formats
git commit -m "feat(formats): TokenizerSpec and RopeStyle in ModelDesc"
```

---

### Task 5: Fixture upgrade — real weights, all five dtypes, tokenizer, tied embeddings

**Files:**
- Modify: `crates/inferno-formats/src/fixtures.rs` (rewrite most of it)
- Modify: `crates/inferno-formats/examples/gen_fixtures.rs` (also write `mlx/tokenizer.json`)
- Regenerate: `crates/inferno-formats/tests/fixtures/*`, `fuzz/corpus/*` (via the example)
- Modify: snapshots + any size-asserting tests via `cargo insta review`

**Interfaces:**
- Consumes: `quant::pack/dequant` (Task 1).
- Produces (used by Tasks 7, 9, 11, 13 tests):
  - `fixtures::tiny_hyperparams() -> HyperParams` — **new sizes**: vocab 260, hidden 64, layers 2, heads 2, kv_heads 1, ffn 256, theta 10000, eps 1e-5, ctx 128. (head_dim 32, kv_dim 32.)
  - `fixtures::tiny_llama_gguf() -> Vec<u8>` — GGUF-named tensors, **Q/K rows permuted** (Interleaved rope), mixed dtypes, embedded BPE tokenizer metadata, **tied embeddings** (no `output.weight`).
  - `fixtures::tiny_llama_safetensors() -> Vec<u8>` — HF-named, unpermuted, same *effective* weights (quantized tensors stored as dequantized F32; F16/BF16 stored as identical F16/BF16 bytes), also tied.
  - `fixtures::tiny_llama_config_json() -> String` (updated sizes)
  - `fixtures::tiny_tokenizer_json() -> String` — HF tokenizer.json equivalent of the embedded vocab.
  - `fixtures::tiny_vocab() -> (Vec<String>, Vec<String>)` — (tokens, merges): 256 byte-level tokens + `<|bos|>`(256, control) + `<|eos|>`(257, control) + `th`(258) + `the`(259); merges `["t h", "th e"]`; `add_bos=false`, `pre="default"`.
- Dtype assignment (identical effective weights in both files — quantization is per-row-block so row permutation commutes with packing):

| canonical tensor | dtype (GGUF) | dtype (MLX safetensors) |
|---|---|---|
| token_embed.weight [260,64] | F32 | F32 |
| layers.*.attn.{q,k,o}_proj [.,64] | Q8_0 | F32 (dequantized values) |
| layers.*.attn.v_proj [32,64] | F16 | F16 (same bytes) |
| layers.*.ffn.gate_proj [256,64] | F16 | F16 |
| layers.*.ffn.up_proj [256,64] | BF16 | BF16 |
| layers.*.ffn.down_proj [64,256] | Q4_K | F32 (dequantized values) |
| all norms [64] | F32 | F32 |

- [ ] **Step 1: Write failing tests** (append to `fixtures.rs` tests or create `#[cfg(test)]` module)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, load_desc, quant};
    use std::io::Cursor;

    #[test]
    fn gguf_fixture_is_tied_quantized_and_tokenized() {
        let desc = crate::gguf::parse(&mut Cursor::new(&tiny_llama_gguf())).unwrap();
        assert!(desc.tensors.iter().all(|t| t.name != "lm_head.weight")); // tied
        let down = desc
            .tensors
            .iter()
            .find(|t| t.name == "layers.0.ffn.down_proj.weight")
            .unwrap();
        assert_eq!(down.dtype, DType::Q4_K);
        assert!(desc.tokenizer.is_some());
    }

    #[test]
    fn gguf_and_mlx_effective_weights_match() {
        // Same value stream: GGUF stores packed (and Q/K-permuted) weights,
        // MLX stores the dequantized (unpermuted) values. Dequantizing the
        // GGUF v_proj (F16, never permuted) must equal the MLX v_proj bytes.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiny.gguf"), tiny_llama_gguf()).unwrap();
        std::fs::write(dir.path().join("config.json"), tiny_llama_config_json()).unwrap();
        std::fs::write(dir.path().join("model.safetensors"), tiny_llama_safetensors()).unwrap();
        let g = load_desc(&dir.path().join("tiny.gguf")).unwrap();
        let m = load_desc(dir.path()).unwrap();
        for name in ["layers.0.attn.v_proj.weight", "layers.1.ffn.up_proj.weight"] {
            let gt = g.tensors.iter().find(|t| t.name == name).unwrap();
            let mt = m.tensors.iter().find(|t| t.name == name).unwrap();
            let gv = quant::dequant(
                &gt.dtype,
                &crate::read_tensor_bytes(&g, gt).unwrap(),
                gt.shape.iter().product::<u64>() as usize,
            )
            .unwrap();
            let mv = quant::dequant(
                &mt.dtype,
                &crate::read_tensor_bytes(&m, mt).unwrap(),
                mt.shape.iter().product::<u64>() as usize,
            )
            .unwrap();
            assert_eq!(gv, mv, "{name}");
        }
    }

    #[test]
    fn weights_are_not_degenerate() {
        let desc = crate::gguf::parse(&mut Cursor::new(&tiny_llama_gguf())).unwrap();
        let embd = desc.tensors.iter().find(|t| t.name == "token_embed.weight").unwrap();
        // Data written into the in-memory image, non-zero and deterministic.
        let bytes = tiny_llama_gguf();
        let start = desc.data_section_offsets[0] + embd.data_offset;
        let b = &bytes[start as usize..(start + 16) as usize];
        assert_ne!(b, &[0u8; 16]);
    }
}
```

- [ ] **Step 2: Run to verify failure** — `cargo nextest run -p inferno-formats fixtures` → fails (no tokenizer metadata, zero weights, `output.weight` still present, F32-only).

- [ ] **Step 3: Rewrite `fixtures.rs`**

Replace the tensor/writer sections (keep and extend the `put_*` helpers; make `put_str`, `put_kv_u32`, `put_kv_f32`, `put_kv_str` `pub(crate)` for the Task-4 test):

```rust
use crate::{DType, HyperParams, quant};

pub fn tiny_hyperparams() -> HyperParams {
    HyperParams {
        vocab_size: 260,
        hidden_size: 64,
        n_layers: 2,
        n_heads: 2,
        n_kv_heads: 1,
        ffn_hidden_size: 256,
        rope_theta: 10000.0,
        norm_eps: 1e-5,
        context_length: 128,
        rope_style: crate::RopeStyle::Interleaved,
    }
}

/// Deterministic xorshift64* stream; weights in [-0.125, 0.125).
fn weight_stream(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((r >> 40) as f32 / 16_777_216.0 - 0.5) * 0.25
        })
        .collect()
}

/// HF half-split → GGML interleaved row order for rope'd projections
/// (convert_hf_to_gguf.py LlamaModel.permute): within each head, source row
/// s*half+j2 (s ∈ {0,1}) moves to row 2*j2+s.
fn permute_rows(w: &[f32], rows: usize, cols: usize, n_head: usize) -> Vec<f32> {
    let hd = rows / n_head;
    let half = hd / 2;
    let mut out = vec![0.0; w.len()];
    for h in 0..n_head {
        for j in 0..hd {
            let dst = h * hd + if j < half { 2 * j } else { 2 * (j - half) + 1 };
            let src = h * hd + j;
            out[dst * cols..(dst + 1) * cols].copy_from_slice(&w[src * cols..(src + 1) * cols]);
        }
    }
    out
}

pub struct FixtureTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

/// Table: (gguf name, hf name, shape, gguf dtype, permute heads (0 = no)).
fn tensor_table() -> Vec<(String, String, Vec<u64>, DType, usize)> {
    let hp = tiny_hyperparams();
    let (v, h, f) = (hp.vocab_size, hp.hidden_size, hp.ffn_hidden_size);
    let kv = h / hp.n_heads * hp.n_kv_heads; // 32
    let mut t: Vec<(String, String, Vec<u64>, DType, usize)> = vec![
        ("token_embd.weight".into(), "model.embed_tokens.weight".into(), vec![v, h], DType::F32, 0),
        ("output_norm.weight".into(), "model.norm.weight".into(), vec![h], DType::F32, 0),
        // NOTE: no output.weight / lm_head.weight — embeddings are tied.
    ];
    for i in 0..hp.n_layers {
        let g = |s: &str| format!("blk.{i}.{s}");
        let m = |s: &str| format!("model.layers.{i}.{s}");
        t.extend([
            (g("attn_norm.weight"), m("input_layernorm.weight"), vec![h], DType::F32, 0),
            (g("attn_q.weight"), m("self_attn.q_proj.weight"), vec![h, h], DType::Q8_0,
             hp.n_heads as usize),
            (g("attn_k.weight"), m("self_attn.k_proj.weight"), vec![kv, h], DType::Q8_0,
             hp.n_kv_heads as usize),
            (g("attn_v.weight"), m("self_attn.v_proj.weight"), vec![kv, h], DType::F16, 0),
            (g("attn_output.weight"), m("self_attn.o_proj.weight"), vec![h, h], DType::Q8_0, 0),
            (g("ffn_norm.weight"), m("post_attention_layernorm.weight"), vec![h], DType::F32, 0),
            (g("ffn_gate.weight"), m("mlp.gate_proj.weight"), vec![f, h], DType::F16, 0),
            (g("ffn_up.weight"), m("mlp.up_proj.weight"), vec![f, h], DType::BF16, 0),
            (g("ffn_down.weight"), m("mlp.down_proj.weight"), vec![h, f], DType::Q4_K, 0),
        ]);
    }
    t
}

/// GGUF-side tensors: packed in `dtype`, Q/K rows permuted (Interleaved rope).
pub fn tiny_tensors_gguf() -> Vec<FixtureTensor> {
    tensor_table()
        .into_iter()
        .enumerate()
        .map(|(seed, (gname, _, shape, dtype, permute_heads))| {
            let n: usize = shape.iter().product::<u64>() as usize;
            let mut w = weight_stream(0xF17E + seed as u64, n);
            if permute_heads > 0 {
                let cols = *shape.last().unwrap() as usize;
                w = permute_rows(&w, n / cols, cols, permute_heads);
            }
            let data = quant::pack(&dtype, &w).unwrap();
            FixtureTensor { name: gname, shape, dtype, data }
        })
        .collect()
}

/// MLX-side tensors: same effective values, HF names, unpermuted, quantized
/// dtypes materialized as F32 (safetensors has no Q8_0/Q4_K).
pub fn tiny_tensors_hf() -> Vec<FixtureTensor> {
    tensor_table()
        .into_iter()
        .enumerate()
        .map(|(seed, (_, hname, shape, dtype, _))| {
            let n: usize = shape.iter().product::<u64>() as usize;
            let w = weight_stream(0xF17E + seed as u64, n);
            // Effective value = dequant(pack(w)); per-row blocks make this
            // independent of the GGUF-side row permutation.
            let eff = quant::dequant(&dtype, &quant::pack(&dtype, &w).unwrap(), n).unwrap();
            let (dtype, data) = match dtype {
                DType::F16 | DType::BF16 => (dtype.clone(), quant::pack(&dtype, &eff).unwrap()),
                _ => (DType::F32, quant::pack(&DType::F32, &eff).unwrap()),
            };
            FixtureTensor { name: hname, shape, dtype, data }
        })
        .collect()
}

/// GPT-2 byte↔unicode table (duplicated in inferno-runtime's BPE tokenizer;
/// kept private here — fixtures are not a stability surface).
fn byte_unicode(b: u8) -> char {
    let printable = (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
    if printable {
        char::from_u32(u32::from(b)).unwrap()
    } else {
        // Non-printables map to 256+n in first-seen order, matching GPT-2.
        let mut n = 0;
        for x in 0u16..u16::from(b) {
            let x8 = x as u8;
            let p = (b'!'..=b'~').contains(&x8) || (0xA1..=0xAC).contains(&x8) || x8 >= 0xAE;
            if x < 256 && !p {
                n += 1;
            }
        }
        char::from_u32(256 + n).unwrap()
    }
}

/// (tokens, merges): 256 byte tokens, <|bos|>=256, <|eos|>=257, "th"=258, "the"=259.
pub fn tiny_vocab() -> (Vec<String>, Vec<String>) {
    let mut tokens: Vec<String> = (0u16..256).map(|b| byte_unicode(b as u8).to_string()).collect();
    tokens.push("<|bos|>".into());
    tokens.push("<|eos|>".into());
    tokens.push("th".into());
    tokens.push("the".into());
    (tokens, vec!["t h".into(), "th e".into()])
}

fn ggml_dtype_id(d: &DType) -> u32 {
    match d {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::Q8_0 => 8,
        DType::Q4_K => 12,
        DType::BF16 => 30,
        DType::Unsupported(_) => unreachable!("fixtures use supported dtypes"),
    }
}
```

GGUF writer (replaces `tiny_llama_gguf`; kv section built as a counted list so the count can't drift):

```rust
pub fn tiny_llama_gguf() -> Vec<u8> {
    let hp = tiny_hyperparams();
    let tensors = tiny_tensors_gguf();
    let (tokens, merges) = tiny_vocab();

    // Each entry in `kvs` is one serialized KV pair; the count is
    // kvs.len() by construction, so it can never drift out of sync.
    let mut token_types = vec![1i32; 256];
    token_types.extend([3, 3, 1, 1]); // bos/eos control, merged tokens normal
    let one = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = Vec::new();
        f(&mut b);
        b
    };
    let kvs: Vec<Vec<u8>> = vec![
        one(&|o| put_kv_str(o, "general.architecture", "llama")),
        one(&|o| put_kv_str(o, "general.name", "tiny-llama-test")),
        one(&|o| put_kv_u32(o, "general.alignment", 32)),
        one(&|o| put_kv_u32(o, "llama.block_count", hp.n_layers as u32)),
        one(&|o| put_kv_u32(o, "llama.embedding_length", hp.hidden_size as u32)),
        one(&|o| put_kv_u32(o, "llama.attention.head_count", hp.n_heads as u32)),
        one(&|o| put_kv_u32(o, "llama.attention.head_count_kv", hp.n_kv_heads as u32)),
        one(&|o| put_kv_u32(o, "llama.feed_forward_length", hp.ffn_hidden_size as u32)),
        one(&|o| put_kv_u32(o, "llama.context_length", hp.context_length as u32)),
        one(&|o| put_kv_f32(o, "llama.attention.layer_norm_rms_epsilon", hp.norm_eps)),
        one(&|o| put_kv_str(o, "tokenizer.ggml.model", "gpt2")),
        one(&|o| put_kv_str(o, "tokenizer.ggml.pre", "default")),
        one(&|o| put_kv_str_array(o, "tokenizer.ggml.tokens", &tokens)),
        one(&|o| put_kv_str_array(o, "tokenizer.ggml.merges", &merges)),
        one(&|o| put_kv_i32_array(o, "tokenizer.ggml.token_type", &token_types)),
        one(&|o| put_kv_u32(o, "tokenizer.ggml.bos_token_id", 256)),
        one(&|o| put_kv_u32(o, "tokenizer.ggml.eos_token_id", 257)),
        one(&|o| put_kv_bool(o, "tokenizer.ggml.add_bos_token", false)),
    ];

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for kv in &kvs {
        out.extend_from_slice(kv);
    }

    // Tensor infos: offsets relative to the 32-aligned data section.
    let mut offset = 0u64;
    for t in &tensors {
        put_str(&mut out, &t.name);
        out.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
        for d in t.shape.iter().rev() {
            out.extend_from_slice(&d.to_le_bytes()); // fastest-first on disk
        }
        out.extend_from_slice(&ggml_dtype_id(&t.dtype).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += (t.data.len() as u64).next_multiple_of(32);
    }
    while out.len() % 32 != 0 {
        out.push(0);
    }
    for t in &tensors {
        out.extend_from_slice(&t.data);
        while out.len() % 32 != 0 {
            out.push(0);
        }
    }
    out
}
```

New KV helpers alongside the existing ones (`put_kv_str_array` already landed in Task 4):

```rust
pub(crate) fn put_kv_i32_array(out: &mut Vec<u8>, key: &str, items: &[i32]) {
    put_str(out, key);
    out.extend_from_slice(&9u32.to_le_bytes());
    out.extend_from_slice(&5u32.to_le_bytes()); // elem: i32
    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
    for v in items {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

pub(crate) fn put_kv_bool(out: &mut Vec<u8>, key: &str, v: bool) {
    put_str(out, key);
    out.extend_from_slice(&7u32.to_le_bytes());
    out.push(u8::from(v));
}
```

Safetensors writer (replaces `tiny_llama_safetensors`):

```rust
pub fn tiny_llama_safetensors() -> Vec<u8> {
    let tensors = tiny_tensors_hf();
    let mut entries = Vec::new();
    let mut offset = 0u64;
    for t in &tensors {
        let end = offset + t.data.len() as u64;
        let dtype = match t.dtype {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            _ => unreachable!("hf fixture tensors are float dtypes"),
        };
        entries.push(format!(
            r#""{}": {{"dtype":"{dtype}","shape":[{}],"data_offsets":[{offset},{end}]}}"#,
            t.name,
            t.shape.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
        ));
        offset = end;
    }
    let json = format!("{{{}}}", entries.join(","));
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(json.as_bytes());
    for t in &tensors {
        out.extend_from_slice(&t.data);
    }
    out
}
```

`tiny_llama_config_json()`: unchanged shape, values now come from the new `tiny_hyperparams()` (it already interpolates — just verify it emits the new sizes).

`tiny_tokenizer_json()` (HF equivalent; ByteLevel BPE):

```rust
pub fn tiny_tokenizer_json() -> String {
    let (tokens, merges) = tiny_vocab();
    let vocab: Vec<String> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| format!(r#""{}": {i}"#, t.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect();
    let merges: Vec<String> = merges.iter().map(|m| format!(r#""{m}""#)).collect();
    format!(
        r#"{{
  "version": "1.0",
  "added_tokens": [
    {{"id": 256, "content": "<|bos|>", "single_word": false, "lstrip": false,
      "rstrip": false, "normalized": false, "special": true}},
    {{"id": 257, "content": "<|eos|>", "single_word": false, "lstrip": false,
      "rstrip": false, "normalized": false, "special": true}}
  ],
  "normalizer": null,
  "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}},
  "post_processor": null,
  "decoder": {{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}},
  "model": {{
    "type": "BPE",
    "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
    "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
    "vocab": {{{vocab}}},
    "merges": [{merges}]
  }}
}}"#,
        vocab = vocab.join(", "),
        merges = merges.join(", ")
    )
}
```

`examples/gen_fixtures.rs` — add after the config.json write:

```rust
    fs::write(
        fix.join("mlx/tokenizer.json"),
        fixtures::tiny_tokenizer_json(),
    )
    .unwrap();
```

- [ ] **Step 4: Regenerate fixtures, run everything, review snapshots**

```bash
cargo run -p inferno-formats --example gen_fixtures
cargo nextest run -p inferno-formats
cargo insta review    # ModelDesc snapshots: new sizes, dtypes, canonical names
cargo nextest run --workspace   # cli inspect tests may assert old sizes — fix if so
```

Expected: Task-5 tests pass; snapshots change (sizes, dtypes, tied embeddings). The `truncated_tensor_info_is_error_not_panic` test in `gguf/mod.rs` still holds (its cut logic is size-independent). Any test asserting `vocab=32`/`hidden=8` gets updated to the new hyperparams.

- [ ] **Step 5: Fuzz (corpus seeds changed), lint, commit**

```bash
mise run fuzz -- gguf_parse
mise run fuzz -- safetensors_parse
mise run lint
git add -A crates/inferno-formats fuzz/corpus
git commit -m "feat(formats): real-weight fixtures with all five dtypes, tokenizer, tied embeddings"
```

---

### Task 6: `inferno-graph` crate — IR types, tolerances, quant round-trip properties

**Files:**
- Create: `crates/inferno-graph/Cargo.toml`, `src/lib.rs`, `src/error.rs`, `src/tolerance.rs`, `src/ir.rs`
- Create: `crates/inferno-graph/tests/quant_roundtrip.rs`
- Modify: `Cargo.toml` (workspace members + deps: `inferno-graph` path dep, `proptest = "1"` dev-dep entry)

**Interfaces:**
- Consumes: `inferno_formats::{DType, ModelDesc, RopeStyle}`.
- Produces (used by Tasks 7–9, 13, 15):

```rust
// error.rs
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("tensor {name}: expected shape {expected:?}, got {got:?}")]
    ShapeMismatch { name: String, expected: Vec<u64>, got: Vec<u64> },
    #[error("invalid hyperparameters: {0}")]
    BadHyperParams(String),
    #[error("token id {id} out of range (vocab {vocab})")]
    TokenOutOfRange { id: u32, vocab: usize },
    #[error("sequence length {got} exceeds capacity {max}")]
    SeqTooLong { got: usize, max: usize },
    #[error(transparent)]
    Format(#[from] inferno_formats::FormatError),
}
pub type Result<T> = std::result::Result<T, GraphError>;

// tolerance.rs — THE single home for numeric comparison constants (spec).
pub fn roundtrip_rel_tol(dtype: &DType) -> f32;   // F32:0.0 F16:1e-3 BF16:8e-3 Q8_0:8e-3 Q4_K:1.1e-1
pub fn logits_abs_tol(dtype: &DType) -> f32;      // widest weight dtype → logit tolerance; quantized 1e-2, float 1e-4
pub const LOGIT_TIE_EPSILON: f32 = 0.05;          // teacher-forced diff tie threshold (nightly may tune; see AGENTS.md)

// ir.rs
pub type ValueId = usize;                          // 0 = tokens input; node i outputs i+1
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim { Const(u64), Seq }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<Dim>);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRef(pub usize);                   // index into ModelDesc::tensors
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Embed { weight: TensorRef },
    MatMul { weight: TensorRef, bias: Option<TensorRef> },
    RmsNorm { weight: TensorRef, eps: f32, head_dim: Option<u64> }, // Some = per-head (Qwen3 q/k norm)
    Rope { theta: f32, style: RopeStyle, n_heads: u64, head_dim: u64 },
    Attention { layer: usize, n_heads: u64, n_kv_heads: u64, head_dim: u64 },
    SwiGlu,
    Add,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Node { pub op: Op, pub inputs: Vec<ValueId>, pub out_shape: Shape, pub label: String }
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub output: ValueId,       // logits [Seq, vocab]
    pub n_layers: u64,
    pub n_kv_heads: u64,
    pub head_dim: u64,
}
impl Graph { pub fn dump(&self, desc: &ModelDesc) -> String }  // stable text for insta
```

- [ ] **Step 1: Scaffold the crate and workspace wiring**

`crates/inferno-graph/Cargo.toml`:

```toml
[package]
name = "inferno-graph"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
inferno-formats.workspace = true
thiserror.workspace = true

[dev-dependencies]
insta.workspace = true
proptest.workspace = true
tempfile.workspace = true

[lints]
workspace = true
```

Workspace `Cargo.toml`: add `"crates/inferno-graph"` to `members`; add to `[workspace.dependencies]`:

```toml
inferno-graph = { path = "crates/inferno-graph" }
proptest = "1"
```

`src/lib.rs`:

```rust
//! Graph IR, Llama-family builder, and the scalar reference interpreter —
//! the correctness oracle every compiled path is measured against.

mod error;
pub mod ir;
pub mod tolerance;

pub use error::{GraphError, Result};
pub use ir::{Dim, Graph, Node, Op, Shape, TensorRef, ValueId};
```

(`build`/`ops`/`interp` modules join in Tasks 7–9.)

- [ ] **Step 2: Write failing tests**

`quant_roundtrip.rs` (integration test — codecs from formats, tolerances from graph):

```rust
//! Property tests: pack → dequant stays within the per-dtype tolerance
//! defined in inferno_graph::tolerance (the single home for these numbers).

use inferno_formats::{DType, quant};
use inferno_graph::tolerance::roundtrip_rel_tol;
use proptest::prelude::*;

fn check_roundtrip(dtype: &DType, vals: &[f32]) {
    let packed = quant::pack(dtype, vals).unwrap();
    let out = quant::dequant(dtype, &packed, vals.len()).unwrap();
    let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    let tol = roundtrip_rel_tol(dtype) * amax;
    for (i, (a, b)) in vals.iter().zip(&out).enumerate() {
        assert!((a - b).abs() <= tol, "{dtype:?}[{i}]: {a} vs {b} (tol {tol})");
    }
}

proptest! {
    #[test]
    fn f16_roundtrip(vals in proptest::collection::vec(-100f32..100.0, 1..64)) {
        check_roundtrip(&DType::F16, &vals);
    }
    #[test]
    fn bf16_roundtrip(vals in proptest::collection::vec(-100f32..100.0, 1..64)) {
        check_roundtrip(&DType::BF16, &vals);
    }
    #[test]
    fn q8_0_roundtrip(vals in proptest::collection::vec(-10f32..10.0, 1..8)) {
        let vals: Vec<f32> = vals.into_iter().cycle().take(64).collect(); // 2 blocks
        check_roundtrip(&DType::Q8_0, &vals);
    }
    #[test]
    fn q4_k_roundtrip(vals in proptest::collection::vec(-10f32..10.0, 256..=512)) {
        let n = (vals.len() / 256) * 256;
        check_roundtrip(&DType::Q4_K, &vals[..n]);
    }
}
```

IR dump test (in `ir.rs` `#[cfg(test)]`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use inferno_formats::RopeStyle;

    #[test]
    fn dump_is_stable_and_readable() {
        use inferno_formats::fixtures;
        let desc =
            inferno_formats::gguf::parse(&mut std::io::Cursor::new(&fixtures::tiny_llama_gguf()))
                .unwrap();
        let embed_idx =
            desc.tensors.iter().position(|t| t.name == "token_embed.weight").unwrap();
        let g = Graph {
            nodes: vec![Node {
                op: Op::Embed { weight: TensorRef(embed_idx) },
                inputs: vec![0],
                out_shape: Shape(vec![Dim::Seq, Dim::Const(64)]),
                label: "embed".into(),
            }],
            output: 1,
            n_layers: 2,
            n_kv_heads: 1,
            head_dim: 32,
        };
        let dump = g.dump(&desc);
        assert!(dump.contains("%1 = embed(%0, @token_embed.weight[260x64:F32]) : [seq,64]"));
        let _ = RopeStyle::HalfSplit; // silence unused import if assertions change
    }
}
```

- [ ] **Step 3: Run to verify failure** — `cargo nextest run -p inferno-graph` → compile errors.

- [ ] **Step 4: Implement `error.rs`, `tolerance.rs`, `ir.rs`**

`error.rs` and the interface types exactly as specified above. `tolerance.rs`:

```rust
//! The single home for numeric comparison constants (spec §Scalar
//! interpreter). Every test layer — quant round-trips here, M2 kernel
//! properties, M3 compiled-vs-reference differentials — imports these.

use inferno_formats::DType;

/// pack→dequant max error, relative to the block's max |value|.
pub fn roundtrip_rel_tol(dtype: &DType) -> f32 {
    match dtype {
        DType::F32 => 0.0,
        DType::F16 => 1e-3,
        DType::BF16 => 8e-3,
        DType::Q8_0 => 8e-3,
        DType::Q4_K => 1.1e-1, // simple min/max reference quantizer, not ggml's optimizer
        DType::Unsupported(_) => 0.0,
    }
}

/// Absolute logit tolerance when comparing two implementations of a model
/// whose widest weight dtype is `dtype` (spec: ~1e-2 on quantized paths).
pub fn logits_abs_tol(dtype: &DType) -> f32 {
    match dtype {
        DType::Q8_0 | DType::Q4_K => 1e-2,
        _ => 1e-4,
    }
}

/// Teacher-forced differential: a position where our top-2 logit gap is
/// below this counts as a genuine tie, not a mismatch. Tuned against the
/// gap distributions the nightly diff reports (see AGENTS.md).
pub const LOGIT_TIE_EPSILON: f32 = 0.05;
```

`ir.rs` — types from **Interfaces**, plus the dump:

```rust
use std::fmt::Write as _;

use inferno_formats::{ModelDesc, RopeStyle};

// … type definitions from the Interfaces block …

impl Graph {
    /// Stable text dump for snapshot tests. Weight refs render as
    /// @name[shape:dtype] so builder regressions show up in review.
    pub fn dump(&self, desc: &ModelDesc) -> String {
        let dim = |d: &Dim| match d {
            Dim::Const(c) => c.to_string(),
            Dim::Seq => "seq".into(),
        };
        let shape = |s: &Shape| {
            format!("[{}]", s.0.iter().map(dim).collect::<Vec<_>>().join(","))
        };
        let wref = |t: &TensorRef| {
            let td = &desc.tensors[t.0];
            format!(
                "@{}[{}:{:?}]",
                td.name,
                td.shape.iter().map(u64::to_string).collect::<Vec<_>>().join("x"),
                td.dtype
            )
        };
        let mut out = format!(
            "graph (layers={}, kv_heads={}, head_dim={})\n  %0 = tokens : [seq]\n",
            self.n_layers, self.n_kv_heads, self.head_dim
        );
        for (i, n) in self.nodes.iter().enumerate() {
            let id = i + 1;
            let ins = |sep: &str| {
                n.inputs.iter().map(|v| format!("%{v}")).collect::<Vec<_>>().join(sep)
            };
            let body = match &n.op {
                Op::Embed { weight } => format!("embed({}, {})", ins(", "), wref(weight)),
                Op::MatMul { weight, bias } => match bias {
                    Some(b) => format!("matmul({}, {}, bias={})", ins(", "), wref(weight), wref(b)),
                    None => format!("matmul({}, {})", ins(", "), wref(weight)),
                },
                Op::RmsNorm { weight, eps, head_dim } => match head_dim {
                    Some(hd) => format!(
                        "rmsnorm_per_head({}, {}, eps={eps}, head_dim={hd})",
                        ins(", "),
                        wref(weight)
                    ),
                    None => format!("rmsnorm({}, {}, eps={eps})", ins(", "), wref(weight)),
                },
                Op::Rope { theta, style, n_heads, head_dim } => format!(
                    "rope({}, theta={theta}, style={style:?}, heads={n_heads}, head_dim={head_dim})",
                    ins(", ")
                ),
                Op::Attention { layer, n_heads, n_kv_heads, head_dim } => format!(
                    "attention({}, layer={layer}, heads={n_heads}, kv_heads={n_kv_heads}, head_dim={head_dim})",
                    ins(", ")
                ),
                Op::SwiGlu => format!("swiglu({})", ins(", ")),
                Op::Add => format!("add({})", ins(", ")),
            };
            let _ = writeln!(out, "  %{id} = {body} : {}  ; {}", shape(&n.out_shape), n.label);
        }
        let _ = writeln!(out, "  output %{}", self.output);
        out
    }
}
```

(Adjust the `dump_is_stable_and_readable` assertion to include the trailing `; embed` label part if the exact format differs — the assertion and implementation must agree on one exact format; the snapshot tests in Task 7 are the real guard.)

- [ ] **Step 5: Run tests**

Run: `cargo nextest run -p inferno-graph`
Expected: quant round-trip properties + IR dump test PASS.

- [ ] **Step 6: Lint and commit**

```bash
mise run lint
git add Cargo.toml Cargo.lock crates/inferno-graph
git commit -m "feat(graph): IR types, tolerance constants, quant round-trip properties"
```

---

### Task 7: Llama-family graph builder

**Files:**
- Create: `crates/inferno-graph/src/build.rs`
- Create: `crates/inferno-graph/tests/snapshot_ir.rs`
- Modify: `crates/inferno-graph/src/lib.rs` (`pub mod build;` + `pub use build::build_graph;`)

**Interfaces:**
- Consumes: `ModelDesc` (canonical names from Task 3, `RopeStyle` from Task 4), IR types (Task 6).
- Produces (used by Tasks 9, 13): `pub fn build_graph(desc: &ModelDesc) -> Result<Graph>`
  - Structure driven by tensor presence: bias tensors (Qwen2), `attn.{q,k}_norm` (Qwen3), missing `lm_head.weight` → tied to `token_embed.weight`.
  - Supported architectures: `Llama | Qwen2 | Qwen3 | Mistral`; anything else → `UnsupportedArch`.
  - Validates hyperparams (all nonzero; `hidden % n_heads == 0`; `n_heads % n_kv_heads == 0`; `head_dim` even; hidden/ffn/vocab each ≤ `1 << 20` as hostile-input allocation guards) and every referenced tensor's shape.

- [ ] **Step 1: Write failing tests** (`tests/snapshot_ir.rs`)

```rust
use std::io::Cursor;
use std::path::Path;

use inferno_formats::{fixtures, load_desc};
use inferno_graph::build_graph;

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferno-formats/tests/fixtures")
}

#[test]
fn gguf_fixture_graph_snapshot() {
    let desc = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    let g = build_graph(&desc).unwrap();
    insta::assert_snapshot!("tiny_gguf_ir", g.dump(&desc));
}

#[test]
fn mlx_fixture_graph_snapshot() {
    let desc = load_desc(&fixture_dir().join("mlx")).unwrap();
    let g = build_graph(&desc).unwrap();
    insta::assert_snapshot!("tiny_mlx_ir", g.dump(&desc));
}

#[test]
fn tied_embeddings_reuse_token_embed() {
    let desc = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    let g = build_graph(&desc).unwrap();
    let embed_idx = desc.tensors.iter().position(|t| t.name == "token_embed.weight").unwrap();
    // Final matmul's weight must be the embedding table (no lm_head in fixture).
    let last = g.nodes.last().unwrap();
    match &last.op {
        inferno_graph::Op::MatMul { weight, .. } => assert_eq!(weight.0, embed_idx),
        other => panic!("expected final MatMul, got {other:?}"),
    }
}

#[test]
fn missing_tensor_is_typed_error() {
    let mut desc = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    desc.tensors.retain(|t| t.name != "layers.0.ffn.gate_proj.weight");
    assert!(matches!(
        build_graph(&desc),
        Err(inferno_graph::GraphError::MissingTensor(name)) if name.contains("gate_proj")
    ));
}

#[test]
fn unknown_arch_is_typed_error() {
    let mut desc = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    desc.architecture = inferno_formats::Architecture::Unknown("mamba".into());
    assert!(matches!(
        build_graph(&desc),
        Err(inferno_graph::GraphError::UnsupportedArch(_))
    ));
}

#[test]
fn hostile_hyperparams_are_typed_errors() {
    let base = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    for mutate in [
        (|d: &mut inferno_formats::ModelDesc| d.hyperparams.n_heads = 0)
            as fn(&mut inferno_formats::ModelDesc),
        |d| d.hyperparams.n_heads = 3,               // hidden 64 % 3 != 0
        |d| d.hyperparams.n_kv_heads = 5,            // heads % kv != 0
        |d| d.hyperparams.hidden_size = 1 << 30,     // allocation guard
        |d| d.hyperparams.vocab_size = u64::MAX,
    ] {
        let mut d = base.clone();
        mutate(&mut d);
        assert!(
            matches!(build_graph(&d), Err(inferno_graph::GraphError::BadHyperParams(_))),
            "hyperparam mutation not rejected"
        );
    }
}

#[test]
fn qwen2_biases_and_qwen3_qk_norm_are_wired() {
    // Synthesize: take the fixture desc, relabel arch, add bias/qk-norm
    // tensor descs (data never read at build time).
    let mut desc = load_desc(&fixture_dir().join("tiny.gguf")).unwrap();
    desc.architecture = inferno_formats::Architecture::Qwen3;
    let template = desc.tensors[0].clone();
    for i in 0..2 {
        for (name, shape) in [
            (format!("layers.{i}.attn.q_proj.bias"), vec![64u64]),
            (format!("layers.{i}.attn.k_proj.bias"), vec![32]),
            (format!("layers.{i}.attn.v_proj.bias"), vec![32]),
            (format!("layers.{i}.attn.q_norm.weight"), vec![32]),
            (format!("layers.{i}.attn.k_norm.weight"), vec![32]),
        ] {
            let mut t = template.clone();
            t.name = name;
            t.shape = shape;
            t.dtype = inferno_formats::DType::F32;
            desc.tensors.push(t);
        }
    }
    let g = build_graph(&desc).unwrap();
    let dump = g.dump(&desc);
    assert!(dump.contains("bias=@layers.0.attn.q_proj.bias"));
    assert!(dump.contains("rmsnorm_per_head"));
}
```

- [ ] **Step 2: Run to verify failure** — `cargo nextest run -p inferno-graph snapshot_ir` → compile error (`build_graph` missing).

- [ ] **Step 3: Implement `build.rs`**

```rust
//! ModelDesc → Graph. One data-driven builder covers the Llama family;
//! presence/absence of canonical tensors drives structure (biases, q/k
//! norms, tied embeddings). Everything reachable from a model file is a
//! typed error, never a panic.

use std::collections::HashMap;

use inferno_formats::{Architecture, ModelDesc};

use crate::ir::{Dim, Graph, Node, Op, Shape, TensorRef, ValueId};
use crate::{GraphError, Result};

/// Allocation guard for hostile hyperparams (spec §Error handling): caps the
/// largest single dimension a model file can request the interpreter allocate.
const MAX_DIM: u64 = 1 << 20;

struct Builder<'d> {
    desc: &'d ModelDesc,
    by_name: HashMap<&'d str, usize>,
    nodes: Vec<Node>,
}

impl<'d> Builder<'d> {
    fn get(&self, name: &str) -> Option<TensorRef> {
        self.by_name.get(name).copied().map(TensorRef)
    }

    fn require(&self, name: &str, expected: &[u64]) -> Result<TensorRef> {
        let r = self.get(name).ok_or_else(|| GraphError::MissingTensor(name.into()))?;
        let got = &self.desc.tensors[r.0].shape;
        if got != expected {
            return Err(GraphError::ShapeMismatch {
                name: name.into(),
                expected: expected.to_vec(),
                got: got.clone(),
            });
        }
        Ok(r)
    }

    /// Optional tensor: absent → None, present with wrong shape → error.
    fn optional(&self, name: &str, expected: &[u64]) -> Result<Option<TensorRef>> {
        match self.get(name) {
            None => Ok(None),
            Some(_) => self.require(name, expected).map(Some),
        }
    }

    fn push(&mut self, op: Op, inputs: Vec<ValueId>, out_shape: Shape, label: String) -> ValueId {
        self.nodes.push(Node { op, inputs, out_shape, label });
        self.nodes.len() // node i outputs value i+1; value 0 is the tokens input
    }
}

pub fn build_graph(desc: &ModelDesc) -> Result<Graph> {
    if let Architecture::Unknown(id) = &desc.architecture {
        return Err(GraphError::UnsupportedArch(id.clone()));
    }
    let hp = &desc.hyperparams;
    let bad = |msg: String| Err(GraphError::BadHyperParams(msg));
    if hp.n_heads == 0 || hp.n_kv_heads == 0 || hp.hidden_size == 0 || hp.n_layers == 0 {
        return bad("zero-valued hyperparameter".into());
    }
    if !hp.hidden_size.is_multiple_of(hp.n_heads) {
        return bad(format!("hidden {} not divisible by heads {}", hp.hidden_size, hp.n_heads));
    }
    if !hp.n_heads.is_multiple_of(hp.n_kv_heads) {
        return bad(format!("heads {} not divisible by kv heads {}", hp.n_heads, hp.n_kv_heads));
    }
    let head_dim = hp.hidden_size / hp.n_heads;
    if !head_dim.is_multiple_of(2) {
        return bad(format!("head_dim {head_dim} must be even for rope"));
    }
    for (what, v) in [
        ("hidden_size", hp.hidden_size),
        ("ffn_hidden_size", hp.ffn_hidden_size),
        ("vocab_size", hp.vocab_size),
        ("n_layers", hp.n_layers),
    ] {
        if v == 0 || v > MAX_DIM {
            return bad(format!("{what} = {v} outside 1..={MAX_DIM}"));
        }
    }

    let (h, f, v) = (hp.hidden_size, hp.ffn_hidden_size, hp.vocab_size);
    let kv_dim = head_dim * hp.n_kv_heads;
    let mut b = Builder {
        desc,
        by_name: desc.tensors.iter().enumerate().map(|(i, t)| (t.name.as_str(), i)).collect(),
        nodes: Vec::new(),
    };

    let seq_h = Shape(vec![Dim::Seq, Dim::Const(h)]);
    let embed_w = b.require("token_embed.weight", &[v, h])?;
    let mut x = b.push(Op::Embed { weight: embed_w }, vec![0], seq_h.clone(), "embed".into());

    for i in 0..hp.n_layers {
        let l = |s: &str| format!("layers.{i}.{s}");
        let li = i as usize;

        let norm_w = b.require(&l("attn_norm.weight"), &[h])?;
        let hn = b.push(
            Op::RmsNorm { weight: norm_w, eps: hp.norm_eps, head_dim: None },
            vec![x],
            seq_h.clone(),
            l("attn_norm"),
        );
        let mut proj = |b: &mut Builder, name: &str, rows: u64| -> Result<ValueId> {
            let w = b.require(&l(&format!("attn.{name}.weight")), &[rows, h])?;
            let bias = b.optional(&l(&format!("attn.{name}.bias")), &[rows])?;
            Ok(b.push(
                Op::MatMul { weight: w, bias },
                vec![hn],
                Shape(vec![Dim::Seq, Dim::Const(rows)]),
                l(&format!("attn.{name}")),
            ))
        };
        let mut q = proj(&mut b, "q_proj", h)?;
        let mut k = proj(&mut b, "k_proj", kv_dim)?;
        let vv = proj(&mut b, "v_proj", kv_dim)?;

        // Qwen3 per-head q/k rmsnorm, before rope.
        if let Some(qn) = b.optional(&l("attn.q_norm.weight"), &[head_dim])? {
            q = b.push(
                Op::RmsNorm { weight: qn, eps: hp.norm_eps, head_dim: Some(head_dim) },
                vec![q],
                Shape(vec![Dim::Seq, Dim::Const(h)]),
                l("attn.q_norm"),
            );
        }
        if let Some(kn) = b.optional(&l("attn.k_norm.weight"), &[head_dim])? {
            k = b.push(
                Op::RmsNorm { weight: kn, eps: hp.norm_eps, head_dim: Some(head_dim) },
                vec![k],
                Shape(vec![Dim::Seq, Dim::Const(kv_dim)]),
                l("attn.k_norm"),
            );
        }

        let rope = |b: &mut Builder, x: ValueId, heads: u64, width: u64, label: String| {
            b.push(
                Op::Rope { theta: hp.rope_theta, style: hp.rope_style, n_heads: heads, head_dim },
                vec![x],
                Shape(vec![Dim::Seq, Dim::Const(width)]),
                label,
            )
        };
        let q = rope(&mut b, q, hp.n_heads, h, l("rope_q"));
        let k = rope(&mut b, k, hp.n_kv_heads, kv_dim, l("rope_k"));

        let att = b.push(
            Op::Attention {
                layer: li,
                n_heads: hp.n_heads,
                n_kv_heads: hp.n_kv_heads,
                head_dim,
            },
            vec![q, k, vv],
            seq_h.clone(),
            l("attention"),
        );
        let ow = b.require(&l("attn.o_proj.weight"), &[h, h])?;
        let o = b.push(Op::MatMul { weight: ow, bias: None }, vec![att], seq_h.clone(), l("attn.o_proj"));
        x = b.push(Op::Add, vec![x, o], seq_h.clone(), l("residual_attn"));

        let fnw = b.require(&l("ffn_norm.weight"), &[h])?;
        let hf = b.push(
            Op::RmsNorm { weight: fnw, eps: hp.norm_eps, head_dim: None },
            vec![x],
            seq_h.clone(),
            l("ffn_norm"),
        );
        let gw = b.require(&l("ffn.gate_proj.weight"), &[f, h])?;
        let uw = b.require(&l("ffn.up_proj.weight"), &[f, h])?;
        let dw = b.require(&l("ffn.down_proj.weight"), &[h, f])?;
        let seq_f = Shape(vec![Dim::Seq, Dim::Const(f)]);
        let g = b.push(Op::MatMul { weight: gw, bias: None }, vec![hf], seq_f.clone(), l("ffn.gate"));
        let u = b.push(Op::MatMul { weight: uw, bias: None }, vec![hf], seq_f.clone(), l("ffn.up"));
        let s = b.push(Op::SwiGlu, vec![g, u], seq_f, l("swiglu"));
        let d = b.push(Op::MatMul { weight: dw, bias: None }, vec![s], seq_h.clone(), l("ffn.down"));
        x = b.push(Op::Add, vec![x, d], seq_h.clone(), l("residual_ffn"));
    }

    let onw = b.require("output_norm.weight", &[h])?;
    let x = b.push(
        Op::RmsNorm { weight: onw, eps: hp.norm_eps, head_dim: None },
        vec![x],
        seq_h,
        "output_norm".into(),
    );
    // Tied embeddings: no lm_head.weight → project with the embedding table.
    let lm = match b.optional("lm_head.weight", &[v, h])? {
        Some(w) => w,
        None => embed_w,
    };
    let out = b.push(
        Op::MatMul { weight: lm, bias: None },
        vec![x],
        Shape(vec![Dim::Seq, Dim::Const(v)]),
        "lm_head".into(),
    );

    Ok(Graph {
        nodes: b.nodes,
        output: out,
        n_layers: hp.n_layers,
        n_kv_heads: hp.n_kv_heads,
        head_dim,
    })
}
```

- [ ] **Step 4: Run tests, review snapshots**

Run: `cargo nextest run -p inferno-graph`
Expected: non-snapshot tests PASS; two new IR snapshots created → `cargo insta review`: check node order (norm→q/k/v→rope→attention→o_proj→residual→ffn), the GGUF dump shows `style=Interleaved` and quant dtypes, the MLX dump shows `style=HalfSplit` and F32/F16/BF16, both end in a `lm_head` matmul against `@token_embed.weight`. Accept.

- [ ] **Step 5: Lint and commit**

```bash
mise run lint
git add crates/inferno-graph
git commit -m "feat(graph): data-driven Llama-family graph builder with IR snapshots"
```

---

### Task 8: Scalar ops

**Files:**
- Create: `crates/inferno-graph/src/ops.rs`
- Modify: `crates/inferno-graph/src/lib.rs` (`pub mod ops;` + `pub use ops::Tensor;`)

**Interfaces:**
- Produces (used by Task 9):

```rust
pub struct Tensor { pub shape: Vec<usize>, pub data: Vec<f32> }   // row-major
impl Tensor { pub fn rows(&self) -> usize; pub fn cols(&self) -> usize }  // 2-D helpers

pub fn embed(ids: &[u32], table: &[f32], vocab: usize, hidden: usize) -> Result<Tensor>;
pub fn matmul(x: &Tensor, w: &[f32], n_out: usize, k: usize, bias: Option<&[f32]>) -> Tensor;
pub fn rmsnorm(x: &Tensor, w: &[f32], eps: f32, head_dim: Option<usize>) -> Tensor;
pub fn rope(x: &Tensor, n_heads: usize, head_dim: usize, theta: f32, style: RopeStyle, pos_off: usize) -> Tensor;
pub fn swiglu(gate: &Tensor, up: &Tensor) -> Tensor;
pub fn add(a: &Tensor, b: &Tensor) -> Tensor;
/// kcache/vcache: `total` rows of `n_kv_heads*head_dim` (this batch's rows
/// already appended). Causal GQA attention, softmax in f32.
pub fn attention(
    q: &Tensor, kcache: &[f32], vcache: &[f32], total: usize,
    n_heads: usize, n_kv_heads: usize, head_dim: usize, pos_off: usize,
) -> Tensor;
```

- All scalar loops, no cleverness — this is the oracle. Weight layout: `w` is row-major `[n_out, k]`, matching file order; `matmul` computes `x·wᵀ` (`out[s][n] = Σ_j x[s][j] * w[n][j]`).

- [ ] **Step 1: Write failing unit tests** (`#[cfg(test)]` in `ops.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use inferno_formats::RopeStyle;

    fn t(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor { shape: shape.to_vec(), data: data.to_vec() }
    }

    #[test]
    fn embed_looks_up_rows_and_bounds_checks() {
        let table = [0.0, 1.0, 2.0, 3.0]; // vocab 2, hidden 2
        let e = embed(&[1, 0], &table, 2, 2).unwrap();
        assert_eq!(e.data, vec![2.0, 3.0, 0.0, 1.0]);
        assert!(matches!(
            embed(&[2], &table, 2, 2),
            Err(crate::GraphError::TokenOutOfRange { id: 2, vocab: 2 })
        ));
    }

    #[test]
    fn matmul_hand_computed() {
        // x = [[1,2]], w rows: [3,4], [5,6]  → [1*3+2*4, 1*5+2*6] = [11, 17]
        let out = matmul(&t(&[1, 2], &[1.0, 2.0]), &[3.0, 4.0, 5.0, 6.0], 2, 2, None);
        assert_eq!(out.data, vec![11.0, 17.0]);
        let out = matmul(&t(&[1, 2], &[1.0, 2.0]), &[3.0, 4.0, 5.0, 6.0], 2, 2, Some(&[10.0, 20.0]));
        assert_eq!(out.data, vec![21.0, 37.0]);
    }

    #[test]
    fn rmsnorm_hand_computed() {
        // x=[3,4]: rms = sqrt((9+16)/2 + 0) = 3.5355339; w=[2,2] → [1.6970563, 2.2627417]
        let out = rmsnorm(&t(&[1, 2], &[3.0, 4.0]), &[2.0, 2.0], 0.0, None);
        assert!((out.data[0] - 1.697_056_3).abs() < 1e-5);
        assert!((out.data[1] - 2.262_741_7).abs() < 1e-5);
    }

    #[test]
    fn rmsnorm_per_head_normalizes_each_head() {
        // 2 heads of dim 2; second head huge — must not affect first head's norm.
        let x = t(&[1, 4], &[3.0, 4.0, 300.0, 400.0]);
        let out = rmsnorm(&x, &[1.0, 1.0], 0.0, Some(2));
        assert!((out.data[0] - 3.0 / 3.535_533_9).abs() < 1e-5);
        assert!((out.data[2] - 300.0 / 353.553_39).abs() < 1e-3);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let x = t(&[1, 4], &[1.0, 2.0, 3.0, 4.0]);
        for style in [RopeStyle::Interleaved, RopeStyle::HalfSplit] {
            let out = rope(&x, 1, 4, 10000.0, style, 0);
            assert_eq!(out.data, x.data, "{style:?}");
        }
    }

    #[test]
    fn rope_hand_computed_position_one() {
        // head_dim 2, one pair, freq = theta^0 = 1 → angle = pos = 1 rad.
        // [1, 0] at pos 1 → [cos1, sin1] in both styles (pair (0,1)).
        let x = t(&[1, 2], &[1.0, 0.0]);
        for style in [RopeStyle::Interleaved, RopeStyle::HalfSplit] {
            let out = rope(&x, 1, 2, 10000.0, style, 1);
            assert!((out.data[0] - 0.540_302_3).abs() < 1e-6, "{style:?}");
            assert!((out.data[1] - 0.841_470_96).abs() < 1e-6, "{style:?}");
        }
    }

    #[test]
    fn rope_styles_pair_differently_at_dim4() {
        // head_dim 4: Interleaved pairs (0,1),(2,3); HalfSplit pairs (0,2),(1,3).
        let x = t(&[1, 4], &[1.0, 0.0, 1.0, 0.0]);
        let inter = rope(&x, 1, 4, 10000.0, RopeStyle::Interleaved, 1);
        let half = rope(&x, 1, 4, 10000.0, RopeStyle::HalfSplit, 1);
        assert_ne!(inter.data, half.data);
    }

    #[test]
    fn swiglu_hand_computed() {
        // silu(1) = 1/(1+e^-1) = 0.731058578; × up 2 = 1.462117
        let out = swiglu(&t(&[1, 1], &[1.0]), &t(&[1, 1], &[2.0]));
        assert!((out.data[0] - 1.462_117_2).abs() < 1e-6);
    }

    #[test]
    fn attention_single_head_hand_computed() {
        // 1 head, head_dim 1, scale 1. Cache: k=[1, 2], v=[10, 20], q for the
        // 2nd position (pos_off 1... here: q is the second row, total=2).
        // q=1: scores [1, 2] → softmax [e1,e2]/Σ → out = (10 e1 + 20 e2)/(e1+e2)
        let q = t(&[1, 1], &[1.0]);
        let e1 = 1f32.exp();
        let e2 = 2f32.exp();
        let expect = (10.0 * e1 + 20.0 * e2) / (e1 + e2);
        let out = attention(&q, &[1.0, 2.0], &[10.0, 20.0], 2, 1, 1, 1, 1);
        assert!((out.data[0] - expect).abs() < 1e-4);
    }

    #[test]
    fn attention_is_causal() {
        // 2 query rows over a 2-row cache with pos_off 0: row 0 may only see
        // key 0; making key 1 enormous must not change row 0's output.
        let q = t(&[2, 1], &[1.0, 1.0]);
        let a = attention(&q, &[1.0, 1.0], &[10.0, 999.0], 2, 1, 1, 1, 0);
        assert!((a.data[0] - 10.0).abs() < 1e-5); // only v[0] visible
    }

    #[test]
    fn attention_gqa_maps_query_heads_to_shared_kv_head() {
        // 2 query heads share 1 kv head (head_dim 1): both heads read the
        // same cache but with their own q values.
        let q = t(&[1, 2], &[1.0, 3.0]);
        let out = attention(&q, &[1.0], &[7.0], 1, 2, 1, 1, 0);
        assert_eq!(out.data, vec![7.0, 7.0]); // single key → output is v
    }
}
```

- [ ] **Step 2: Run to verify failure** — `cargo nextest run -p inferno-graph ops` → compile error.

- [ ] **Step 3: Implement `ops.rs`**

```rust
//! Obviously-correct scalar implementations of every graph op. No SIMD, no
//! rayon, no blocking — this is the oracle the compiled paths are compared
//! against (spec §Scalar interpreter). f32 accumulation throughout.

use inferno_formats::RopeStyle;

use crate::{GraphError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn rows(&self) -> usize {
        self.shape[0]
    }
    pub fn cols(&self) -> usize {
        self.shape[1]
    }
}

pub fn embed(ids: &[u32], table: &[f32], vocab: usize, hidden: usize) -> Result<Tensor> {
    let mut data = Vec::with_capacity(ids.len() * hidden);
    for &id in ids {
        let i = id as usize;
        if i >= vocab {
            return Err(GraphError::TokenOutOfRange { id, vocab });
        }
        data.extend_from_slice(&table[i * hidden..(i + 1) * hidden]);
    }
    Ok(Tensor { shape: vec![ids.len(), hidden], data })
}

/// x [seq, k] · wᵀ, w row-major [n_out, k] (file order) → [seq, n_out].
pub fn matmul(x: &Tensor, w: &[f32], n_out: usize, k: usize, bias: Option<&[f32]>) -> Tensor {
    let seq = x.rows();
    let mut data = vec![0f32; seq * n_out];
    for s in 0..seq {
        let xr = &x.data[s * k..(s + 1) * k];
        for n in 0..n_out {
            let wr = &w[n * k..(n + 1) * k];
            let mut acc = 0f32;
            for j in 0..k {
                acc += xr[j] * wr[j];
            }
            data[s * n_out + n] = acc + bias.map_or(0.0, |b| b[n]);
        }
    }
    Tensor { shape: vec![seq, n_out], data }
}

/// head_dim = None: normalize each row. Some(hd): normalize each hd-slice of
/// each row independently, cycling the weight (Qwen3 per-head q/k norm).
pub fn rmsnorm(x: &Tensor, w: &[f32], eps: f32, head_dim: Option<usize>) -> Tensor {
    let cols = x.cols();
    let unit = head_dim.unwrap_or(cols);
    let mut data = Vec::with_capacity(x.data.len());
    for chunk in x.data.chunks_exact(unit) {
        let ms = chunk.iter().map(|v| v * v).sum::<f32>() / unit as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (j, v) in chunk.iter().enumerate() {
            data.push(v * inv * w[j]);
        }
    }
    Tensor { shape: x.shape.clone(), data }
}

pub fn rope(
    x: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    style: RopeStyle,
    pos_off: usize,
) -> Tensor {
    let mut out = x.clone();
    let half = head_dim / 2;
    for s in 0..x.rows() {
        let pos = (pos_off + s) as f32;
        for h in 0..n_heads {
            let base = s * x.cols() + h * head_dim;
            for i in 0..half {
                let freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
                let angle = pos * freq;
                let (sin, cos) = angle.sin_cos();
                let (a, b) = match style {
                    RopeStyle::Interleaved => (base + 2 * i, base + 2 * i + 1),
                    RopeStyle::HalfSplit => (base + i, base + i + half),
                };
                let (x0, x1) = (out.data[a], out.data[b]);
                out.data[a] = x0 * cos - x1 * sin;
                out.data[b] = x0 * sin + x1 * cos;
            }
        }
    }
    out
}

pub fn swiglu(gate: &Tensor, up: &Tensor) -> Tensor {
    let data = gate
        .data
        .iter()
        .zip(&up.data)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    Tensor { shape: gate.shape.clone(), data }
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Tensor { shape: a.shape.clone(), data }
}

/// Causal GQA attention. q [seq, n_heads*head_dim]; kcache/vcache hold
/// `total` rows of n_kv_heads*head_dim (this batch already appended).
/// Query row s has absolute position pos_off + s and attends to keys
/// 0..=pos_off+s. Softmax with max-subtraction, all f32.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &Tensor,
    kcache: &[f32],
    vcache: &[f32],
    total: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos_off: usize,
) -> Tensor {
    let seq = q.rows();
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let group = n_heads / n_kv_heads;
    let mut data = vec![0f32; seq * n_heads * head_dim];
    let mut scores = vec![0f32; total];
    for s in 0..seq {
        let visible = pos_off + s + 1; // causal horizon
        for h in 0..n_heads {
            let g = h / group;
            let qv = &q.data[s * n_heads * head_dim + h * head_dim..][..head_dim];
            for (t, sc) in scores[..visible].iter_mut().enumerate() {
                let kv = &kcache[t * kv_dim + g * head_dim..][..head_dim];
                *sc = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let max = scores[..visible].iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
            let mut denom = 0f32;
            for sc in &mut scores[..visible] {
                *sc = (*sc - max).exp();
                denom += *sc;
            }
            let out = &mut data[s * n_heads * head_dim + h * head_dim..][..head_dim];
            for (t, &w) in scores[..visible].iter().enumerate() {
                let vv = &vcache[t * kv_dim + g * head_dim..][..head_dim];
                let w = w / denom;
                for (o, v) in out.iter_mut().zip(vv) {
                    *o += w * v;
                }
            }
        }
    }
    Tensor { shape: vec![seq, n_heads * head_dim], data }
}
```

- [ ] **Step 4: Run tests** — `cargo nextest run -p inferno-graph ops` → all PASS.

- [ ] **Step 5: Lint and commit**

```bash
mise run lint
git add crates/inferno-graph
git commit -m "feat(graph): scalar reference ops with hand-computed unit tests"
```

---

### Task 9: Interpreter — weight cache, KV cache, executor, format differential

**Files:**
- Create: `crates/inferno-graph/src/interp.rs`
- Create: `crates/inferno-graph/tests/differential.rs`
- Modify: `crates/inferno-graph/src/lib.rs` (`pub mod interp;` + `pub use interp::{Interpreter, KvCache};`)

**Interfaces:**
- Consumes: `build_graph` (7), ops (8), `read_tensor_bytes` + `quant::dequant` (1–2).
- Produces (used by Tasks 13, 15):

```rust
pub struct KvCache { /* per-layer k/v, each seq-major rows of n_kv_heads*head_dim */ }
impl KvCache {
    /// Allocates for max_seq_len up front. Errors (BadHyperParams) if the
    /// hyperparam-derived total exceeds MAX_KV_BYTES (8 GiB) — hostile-input guard.
    pub fn new(graph: &Graph, max_seq_len: usize) -> Result<KvCache>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn max_seq_len(&self) -> usize;
}
pub struct Interpreter { /* weights: HashMap<usize, Vec<f32>> — lazy dequant cache */ }
impl Interpreter {
    pub fn new() -> Interpreter;   // (+ Default)
    /// Executes the graph over `tokens`, appending to `kv`. Returns logits
    /// for EVERY position: [tokens.len(), vocab].
    pub fn run(
        &mut self, desc: &ModelDesc, graph: &Graph, tokens: &[u32], kv: &mut KvCache,
    ) -> Result<Tensor>;
}
```

- [ ] **Step 1: Write failing tests** (`tests/differential.rs`)

```rust
use std::path::Path;

use inferno_formats::load_desc;
use inferno_graph::{Interpreter, KvCache, build_graph};

fn fixture(p: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferno-formats/tests/fixtures").join(p)
}

fn argmax(row: &[f32]) -> usize {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

#[test]
fn prefill_then_decode_matches_single_prefill() {
    let desc = load_desc(&fixture("tiny.gguf")).unwrap();
    let graph = build_graph(&desc).unwrap();
    let mut interp = Interpreter::new();
    let toks = [3u32, 5, 250, 42];

    let mut kv_a = KvCache::new(&graph, 16).unwrap();
    let all = interp.run(&desc, &graph, &toks, &mut kv_a).unwrap();

    let mut kv_b = KvCache::new(&graph, 16).unwrap();
    let _ = interp.run(&desc, &graph, &toks[..3], &mut kv_b).unwrap();
    let last = interp.run(&desc, &graph, &toks[3..], &mut kv_b).unwrap();

    // Same final-position logits whether the last token came via prefill or decode.
    let vocab = desc.hyperparams.vocab_size as usize;
    let a = &all.data[3 * vocab..4 * vocab];
    let b = &last.data[..vocab];
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
    }
}

#[test]
fn gguf_and_mlx_fixtures_agree() {
    // THE two-formats boundary test: same effective weights via different
    // formats, names, rope styles, and Q/K permutation must produce the
    // same logits (within float tolerance) and the same argmax chain.
    let dg = load_desc(&fixture("tiny.gguf")).unwrap();
    let dm = load_desc(&fixture("mlx")).unwrap();
    let gg = build_graph(&dg).unwrap();
    let gm = build_graph(&dm).unwrap();
    let toks = [1u32, 200, 116, 104, 101];
    let mut ig = Interpreter::new();
    let mut im = Interpreter::new();
    let mut kg = KvCache::new(&gg, 16).unwrap();
    let mut km = KvCache::new(&gm, 16).unwrap();
    let lg = ig.run(&dg, &gg, &toks, &mut kg).unwrap();
    let lm = im.run(&dm, &gm, &toks, &mut km).unwrap();
    assert_eq!(lg.shape, lm.shape);
    let widest = inferno_formats::DType::Q4_K; // widest dtype in the fixture
    let tol = inferno_graph::tolerance::logits_abs_tol(&widest);
    for (i, (a, b)) in lg.data.iter().zip(&lm.data).enumerate() {
        assert!((a - b).abs() <= tol, "logit {i}: {a} vs {b}");
    }
    let vocab = dg.hyperparams.vocab_size as usize;
    for p in 0..toks.len() {
        assert_eq!(
            argmax(&lg.data[p * vocab..(p + 1) * vocab]),
            argmax(&lm.data[p * vocab..(p + 1) * vocab]),
            "argmax diverged at position {p}"
        );
    }
}

#[test]
fn run_is_deterministic() {
    let desc = load_desc(&fixture("tiny.gguf")).unwrap();
    let graph = build_graph(&desc).unwrap();
    let toks = [7u32, 8, 9];
    let mut i1 = Interpreter::new();
    let mut i2 = Interpreter::new();
    let mut k1 = KvCache::new(&graph, 8).unwrap();
    let mut k2 = KvCache::new(&graph, 8).unwrap();
    let a = i1.run(&desc, &graph, &toks, &mut k1).unwrap();
    let b = i2.run(&desc, &graph, &toks, &mut k2).unwrap();
    assert_eq!(a.data, b.data); // bitwise: scalar f32, fixed order
}

#[test]
fn seq_overflow_and_hostile_kv_are_errors() {
    let desc = load_desc(&fixture("tiny.gguf")).unwrap();
    let graph = build_graph(&desc).unwrap();
    let mut kv = KvCache::new(&graph, 2).unwrap();
    let mut interp = Interpreter::new();
    assert!(matches!(
        interp.run(&desc, &graph, &[1, 2, 3], &mut kv),
        Err(inferno_graph::GraphError::SeqTooLong { got: 3, max: 2 })
    ));
    // Hostile hyperparams cannot make KvCache::new allocate unboundedly.
    let mut big = graph.clone();
    big.n_layers = u64::MAX / 4;
    assert!(KvCache::new(&big, 1 << 20).is_err());
}
```

- [ ] **Step 2: Run to verify failure** — `cargo nextest run -p inferno-graph differential` → compile error.

- [ ] **Step 3: Implement `interp.rs`**

```rust
//! Graph-walking scalar interpreter. Weights dequantize to f32 lazily on
//! first use and stay cached (fine for the ≤~1B-param models the oracle
//! targets — ~4 bytes/param). KV cache is allocated once up front.

use std::collections::HashMap;

use inferno_formats::{ModelDesc, quant, read_tensor_bytes};

use crate::ir::{Graph, Op};
use crate::ops::{self, Tensor};
use crate::{GraphError, Result};

/// Hostile-input guard: max total KV allocation (spec §Error handling).
const MAX_KV_BYTES: u64 = 8 << 30;

pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    kv_dim: usize,
    max_seq_len: usize,
    len: usize,
}

impl KvCache {
    pub fn new(graph: &Graph, max_seq_len: usize) -> Result<KvCache> {
        let kv_dim = (graph.n_kv_heads * graph.head_dim) as usize;
        let per_layer = (kv_dim as u64)
            .checked_mul(max_seq_len as u64)
            .and_then(|n| n.checked_mul(8)) // k + v, 4 bytes each
            .ok_or_else(|| GraphError::BadHyperParams("kv size overflow".into()))?;
        let total = per_layer
            .checked_mul(graph.n_layers)
            .ok_or_else(|| GraphError::BadHyperParams("kv size overflow".into()))?;
        if total > MAX_KV_BYTES {
            return Err(GraphError::BadHyperParams(format!(
                "kv cache would need {total} bytes (limit {MAX_KV_BYTES})"
            )));
        }
        let layers = graph.n_layers as usize;
        let mk = || {
            let mut v = Vec::new();
            v.reserve_exact(kv_dim * max_seq_len);
            v
        };
        Ok(KvCache {
            k: (0..layers).map(|_| mk()).collect(),
            v: (0..layers).map(|_| mk()).collect(),
            kv_dim,
            max_seq_len,
            len: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[derive(Default)]
pub struct Interpreter {
    weights: HashMap<usize, Vec<f32>>,
}

impl Interpreter {
    pub fn new() -> Interpreter {
        Interpreter::default()
    }

    fn weight(&mut self, desc: &ModelDesc, idx: usize) -> Result<&[f32]> {
        if !self.weights.contains_key(&idx) {
            let t = &desc.tensors[idx];
            let n: u64 = t.shape.iter().product();
            let bytes = read_tensor_bytes(desc, t)?;
            let vals = quant::dequant(&t.dtype, &bytes, n as usize)?;
            self.weights.insert(idx, vals);
        }
        Ok(self.weights.get(&idx).unwrap().as_slice())
    }

    pub fn run(
        &mut self,
        desc: &ModelDesc,
        graph: &Graph,
        tokens: &[u32],
        kv: &mut KvCache,
    ) -> Result<Tensor> {
        if kv.len + tokens.len() > kv.max_seq_len {
            return Err(GraphError::SeqTooLong {
                got: kv.len + tokens.len(),
                max: kv.max_seq_len,
            });
        }
        let pos_off = kv.len;
        let hp = &desc.hyperparams;
        // env[v]: value v's tensor. env[0] unused (tokens are read directly).
        let mut env: Vec<Option<Tensor>> = vec![None; graph.nodes.len() + 1];
        for (i, node) in graph.nodes.iter().enumerate() {
            let id = i + 1;
            let arg = |v: usize, env: &[Option<Tensor>]| -> Tensor {
                env[v].clone().expect("graph is topologically ordered")
            };
            let out = match &node.op {
                Op::Embed { weight } => {
                    let table = self.weight(desc, weight.0)?;
                    ops::embed(
                        tokens,
                        table,
                        hp.vocab_size as usize,
                        hp.hidden_size as usize,
                    )?
                }
                Op::MatMul { weight, bias } => {
                    let x = arg(node.inputs[0], &env);
                    let wt = &desc.tensors[weight.0];
                    let (n_out, k) = (wt.shape[0] as usize, wt.shape[1] as usize);
                    let b = match bias {
                        Some(br) => Some(self.weight(desc, br.0)?.to_vec()),
                        None => None,
                    };
                    let w = self.weight(desc, weight.0)?;
                    ops::matmul(&x, w, n_out, k, b.as_deref())
                }
                Op::RmsNorm { weight, eps, head_dim } => {
                    let x = arg(node.inputs[0], &env);
                    let w = self.weight(desc, weight.0)?;
                    ops::rmsnorm(&x, w, *eps, head_dim.map(|d| d as usize))
                }
                Op::Rope { theta, style, n_heads, head_dim } => {
                    let x = arg(node.inputs[0], &env);
                    ops::rope(&x, *n_heads as usize, *head_dim as usize, *theta, *style, pos_off)
                }
                Op::Attention { layer, n_heads, n_kv_heads, head_dim } => {
                    let q = arg(node.inputs[0], &env);
                    let k = arg(node.inputs[1], &env);
                    let v = arg(node.inputs[2], &env);
                    kv.k[*layer].extend_from_slice(&k.data);
                    kv.v[*layer].extend_from_slice(&v.data);
                    let total = kv.k[*layer].len() / kv.kv_dim;
                    ops::attention(
                        &q,
                        &kv.k[*layer],
                        &kv.v[*layer],
                        total,
                        *n_heads as usize,
                        *n_kv_heads as usize,
                        *head_dim as usize,
                        pos_off,
                    )
                }
                Op::SwiGlu => {
                    ops::swiglu(&arg(node.inputs[0], &env), &arg(node.inputs[1], &env))
                }
                Op::Add => ops::add(&arg(node.inputs[0], &env), &arg(node.inputs[1], &env)),
            };
            env[id] = Some(out);
        }
        kv.len += tokens.len();
        Ok(env[graph.output].take().expect("output value produced"))
    }
}
```

Performance note (documented, not optimized): `arg` clones input tensors — O(seq·hidden) copies against O(seq·hidden²) matmuls; irrelevant for an oracle and kept for borrow simplicity. Do not "fix" this with lifetimes; the compiled path is where speed comes from.

- [ ] **Step 4: Run tests** — `cargo nextest run -p inferno-graph` → all PASS (including the GGUF-vs-MLX differential, which proves canonical naming, rope styles, Q/K permutation, tied embeddings, and all five dtype paths agree end-to-end).

- [ ] **Step 5: Full suite, lint, commit**

```bash
mise run test
mise run lint
git add crates/inferno-graph
git commit -m "feat(graph): scalar interpreter with lazy dequant, KV cache, format differential"
```

---

### Task 10: `inferno-runtime` crate — tokenizer trait, byte decoding, HF wrapper

**Files:**
- Create: `crates/inferno-runtime/Cargo.toml`, `src/lib.rs`, `src/error.rs`
- Create: `crates/inferno-runtime/src/tokenizer/mod.rs`, `src/tokenizer/bytes.rs`, `src/tokenizer/hf.rs`
- Modify: `Cargo.toml` (workspace member + deps)

**Interfaces:**
- Consumes: `TokenizerSpec`, `TokenizerKind`, `SpecialTokens` (Task 4); `tokenizers` crate; fixture `tokenizer.json` (Task 5).
- Produces (used by Tasks 11–15):

```rust
// error.rs
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("model has no usable tokenizer metadata")]
    NoTokenizer,
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("prompt ({got} tokens) exceeds max sequence length ({max})")]
    PromptTooLong { got: usize, max: usize },
    #[error(transparent)]
    Graph(#[from] inferno_graph::GraphError),
    #[error(transparent)]
    Format(#[from] inferno_formats::FormatError),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}
pub type Result<T> = std::result::Result<T, RuntimeError>;

// tokenizer/mod.rs
pub trait Tokenizer: Send {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>>;
    fn decode_token(&self, id: u32) -> Vec<u8>;   // raw bytes; may split UTF-8
    fn bos(&self) -> Option<u32>;
    fn eos(&self) -> Option<u32>;
    fn default_add_bos(&self) -> bool;
}
pub fn tokenizer_for(spec: &TokenizerSpec) -> Result<Box<dyn Tokenizer>>;

// tokenizer/bytes.rs — shared by native BPE, SPM, and the HF wrapper
pub(crate) fn byte_to_unicode() -> [char; 256];               // GPT-2 table
pub(crate) fn unicode_to_byte() -> HashMap<char, u8>;
pub(crate) fn bpe_token_to_bytes(token: &str) -> Option<Vec<u8>>;  // None if any char not in table
pub(crate) fn spm_token_to_bytes(token: &str) -> Vec<u8>;     // ▁→space, <0xNN>→byte
```

- Workspace `Cargo.toml` additions:

```toml
# members: + "crates/inferno-runtime"
# [workspace.dependencies]:
inferno-runtime = { path = "crates/inferno-runtime" }
inferno-graph = { path = "crates/inferno-graph" }   # (added in Task 6)
tokenizers = { version = "0.22", default-features = false, features = ["fancy-regex"] }
fancy-regex = "0.16"
```

(If the `tokenizers` feature name differs at the pinned version, `cargo` will reject it — check `cargo add tokenizers --dry-run --no-default-features` output and use the crate's non-onig regex feature; the requirement is: **no C onig dependency, no HTTP/hub features**.)

- [ ] **Step 1: Scaffold crate**

`crates/inferno-runtime/Cargo.toml`:

```toml
[package]
name = "inferno-runtime"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
inferno-formats.workspace = true
inferno-graph.workspace = true
thiserror.workspace = true
tokenizers.workspace = true
fancy-regex.workspace = true

[dev-dependencies]
proptest.workspace = true
tempfile.workspace = true

[lints]
workspace = true
```

`src/lib.rs`:

```rust
//! Tokenizer, sampling, and the generation loop. In M1 the model executes
//! on the inferno-graph scalar interpreter; M3 swaps in compiled entry
//! points without moving this code.

mod error;
pub mod tokenizer;

pub use error::{Result, RuntimeError};
pub use tokenizer::{Tokenizer, tokenizer_for};
```

- [ ] **Step 2: Write failing tests**

In `tokenizer/bytes.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_roundtrips_all_bytes() {
        let enc = byte_to_unicode();
        let dec = unicode_to_byte();
        for b in 0u16..=255 {
            assert_eq!(dec[&enc[b as usize]], b as u8);
        }
        assert_eq!(enc[b'a' as usize], 'a'); // printables map to themselves
        assert_eq!(enc[b' ' as usize], '\u{120}'); // space → Ġ
    }

    #[test]
    fn bpe_token_to_bytes_decodes_gpt2_form() {
        assert_eq!(bpe_token_to_bytes("Ġhello").unwrap(), b" hello");
        assert_eq!(bpe_token_to_bytes("the").unwrap(), b"the");
        // "<|bos|>" is all printable ASCII, so it decodes to its own bytes —
        // special tokens are filtered by token TYPE before decoding (Tasks
        // 11/13), never detected here.
        assert_eq!(bpe_token_to_bytes("<|bos|>").unwrap(), b"<|bos|>");
    }

    #[test]
    fn spm_token_to_bytes_handles_space_and_byte_fallback() {
        assert_eq!(spm_token_to_bytes("\u{2581}the"), b" the");
        assert_eq!(spm_token_to_bytes("<0x0A>"), b"\n");
        assert_eq!(spm_token_to_bytes("x"), b"x");
    }
}
```

In `tokenizer/hf.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use inferno_formats::TokenizerSpec;
    use std::path::Path;

    fn fixture_tokenizer() -> Box<dyn crate::Tokenizer> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../inferno-formats/tests/fixtures/mlx/tokenizer.json");
        crate::tokenizer_for(&TokenizerSpec::HfJson { path }).unwrap()
    }

    #[test]
    fn encodes_with_merges() {
        let t = fixture_tokenizer();
        // "the" merges via "t h"→"th", "th e"→"the" → single token 259.
        assert_eq!(t.encode("the", false).unwrap(), vec![259]);
        // "cat" has no merges → three byte tokens.
        assert_eq!(t.encode("cat", false).unwrap(), vec![u32::from(b'c'), u32::from(b'a'), u32::from(b't')]);
    }

    #[test]
    fn decode_token_returns_bytes() {
        let t = fixture_tokenizer();
        assert_eq!(t.decode_token(259), b"the");
        assert_eq!(t.decode_token(u32::from(b' ')), b" ");
    }
}
```

Wait — byte-token ids: in `tiny_vocab()` the byte tokens are ids 0..=255 in **byte order**, so `'c'` = 99. The assertions above are correct as written.

- [ ] **Step 3: Run to verify failure** — `cargo nextest run -p inferno-runtime` → compile errors.

- [ ] **Step 4: Implement**

`tokenizer/bytes.rs`:

```rust
//! GPT-2 byte↔unicode mapping and token-string → raw-bytes decoding, shared
//! by the native BPE/SPM tokenizers and the HF wrapper.

use std::collections::HashMap;
use std::sync::OnceLock;

pub(crate) fn byte_to_unicode() -> &'static [char; 256] {
    static TABLE: OnceLock<[char; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let printable =
            |b: u8| (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE;
        let mut table = ['\0'; 256];
        let mut n = 0u32;
        for b in 0u16..=255 {
            let b8 = b as u8;
            table[b as usize] = if printable(b8) {
                char::from_u32(u32::from(b)).unwrap()
            } else {
                n += 1;
                char::from_u32(255 + n).unwrap()
            };
        }
        table
    })
}

pub(crate) fn unicode_to_byte() -> &'static HashMap<char, u8> {
    static MAP: OnceLock<HashMap<char, u8>> = OnceLock::new();
    MAP.get_or_init(|| {
        byte_to_unicode().iter().enumerate().map(|(b, &c)| (c, b as u8)).collect()
    })
}

/// GPT-2-form BPE token string → raw bytes. None if any char is outside the
/// byte-unicode alphabet (e.g. a control/added token that slipped through —
/// callers filter specials by token type before decoding).
pub(crate) fn bpe_token_to_bytes(token: &str) -> Option<Vec<u8>> {
    let map = unicode_to_byte();
    token.chars().map(|c| map.get(&c).copied()).collect()
}

/// SPM token string → raw bytes: ▁ (U+2581) → space, <0xNN> byte-fallback
/// tokens → their byte, everything else passes through as UTF-8.
pub(crate) fn spm_token_to_bytes(token: &str) -> Vec<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        if let Ok(b) = u8::from_str_radix(&token[3..5], 16) {
            return vec![b];
        }
    }
    token.replace('\u{2581}', " ").into_bytes()
}
```

(Sanity check on the table: non-printables get codepoints 256, 257, … in byte order, so the first non-printable (0x00) → 256 and space (0x20, the 33rd non-printable) → 288 = U+0120 = Ġ, which is exactly what the test asserts.)

`tokenizer/mod.rs`:

```rust
pub(crate) mod bytes;
mod hf;
// bpe and spm modules join in Tasks 11–12.

use inferno_formats::TokenizerSpec;

use crate::{Result, RuntimeError};

pub trait Tokenizer: Send {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>>;
    fn decode_token(&self, id: u32) -> Vec<u8>;
    fn bos(&self) -> Option<u32>;
    fn eos(&self) -> Option<u32>;
    fn default_add_bos(&self) -> bool;
}

pub fn tokenizer_for(spec: &TokenizerSpec) -> Result<Box<dyn Tokenizer>> {
    match spec {
        TokenizerSpec::HfJson { path } => Ok(Box::new(hf::HfTokenizer::load(path)?)),
        TokenizerSpec::Embedded { .. } => {
            // Native implementations land in Tasks 11–12.
            Err(RuntimeError::Tokenizer("embedded tokenizers not yet wired".into()))
        }
    }
}
```

`tokenizer/hf.rs`:

```rust
//! tokenizer.json via the `tokenizers` crate (MLX models). Also the
//! property-test reference for the native BPE implementation.

use std::path::Path;

use crate::tokenizer::bytes::bpe_token_to_bytes;
use crate::{Result, RuntimeError};

pub(crate) struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    bos: Option<u32>,
    eos: Option<u32>,
}

impl HfTokenizer {
    pub(crate) fn load(path: &Path) -> Result<HfTokenizer> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        // Conventional special-token names; absent → None (MLX configs vary,
        // and generation just won't auto-stop — CLI max-tokens still bounds it).
        let find = |names: &[&str]| {
            names.iter().find_map(|n| inner.token_to_id(n))
        };
        Ok(HfTokenizer {
            bos: find(&["<|bos|>", "<s>", "<|begin_of_text|>"]),
            eos: find(&["<|eos|>", "</s>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"]),
            inner,
        })
    }
}

impl crate::Tokenizer for HfTokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        let mut ids = Vec::new();
        if add_bos && let Some(b) = self.bos {
            ids.push(b);
        }
        ids.extend_from_slice(enc.get_ids());
        Ok(ids)
    }

    fn decode_token(&self, id: u32) -> Vec<u8> {
        match self.inner.id_to_token(id) {
            // Byte-level BPE token → exact bytes via the shared table.
            Some(tok) => bpe_token_to_bytes(&tok)
                .unwrap_or_else(|| crate::tokenizer::bytes::spm_token_to_bytes(&tok)),
            None => Vec::new(),
        }
    }

    fn bos(&self) -> Option<u32> {
        self.bos
    }
    fn eos(&self) -> Option<u32> {
        self.eos
    }
    fn default_add_bos(&self) -> bool {
        false // HF post-processors handle BOS when configured; fixture/Qwen don't add one
    }
}
```

- [ ] **Step 5: Run tests** — `cargo nextest run -p inferno-runtime` → PASS.

- [ ] **Step 6: Audit new deps, lint, commit**

```bash
mise run audit    # tokenizers + fancy-regex join the supply chain (spec §Error handling)
mise run lint
git add Cargo.toml Cargo.lock crates/inferno-runtime
git commit -m "feat(runtime): tokenizer trait, byte decoding, HF tokenizer.json wrapper"
```

---

### Task 11: Native byte-level BPE tokenizer

**Files:**
- Create: `crates/inferno-runtime/src/tokenizer/bpe.rs`
- Modify: `crates/inferno-runtime/src/tokenizer/mod.rs` (wire `Embedded { kind: Bpe }` into `tokenizer_for`)

**Interfaces:**
- Consumes: `TokenizerSpec::Embedded` fields (Task 4), `bytes.rs` (Task 10), `fancy_regex`.
- Produces: `pub(crate) struct BpeTokenizer` implementing `Tokenizer`; `tokenizer_for` now returns it for `Embedded { kind: Bpe, .. }`.
- GGUF token-type ids used: 1 normal, 3 control, 4 user-defined, 6 byte. Control/user-defined tokens are split out of the text by literal longest-match **before** pre-tokenization and never merge with neighbors.
- Pre-tokenizer patterns by GGUF `pre` id (lookahead needs `fancy_regex`):

| `pre` | pattern |
|---|---|
| `default` (also None, `gpt-2`) | `'s\|'t\|'re\|'ve\|'m\|'ll\|'d\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)\|\s+` |
| `llama-bpe`, `llama3` | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}{1,3}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` |
| `qwen2` | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` |
| anything else | `RuntimeError::Tokenizer("unsupported pre-tokenizer …")` — loud, not wrong |

- [ ] **Step 1: Write failing tests** (`#[cfg(test)]` in `bpe.rs`)

```rust
#[cfg(test)]
mod tests {
    use inferno_formats::{SpecialTokens, TokenizerKind, TokenizerSpec};
    use inferno_formats::fixtures;

    fn native() -> Box<dyn crate::Tokenizer> {
        let (tokens, merges) = fixtures::tiny_vocab();
        let mut token_types = vec![1i32; 256];
        token_types.extend([3, 3, 1, 1]);
        crate::tokenizer_for(&TokenizerSpec::Embedded {
            kind: TokenizerKind::Bpe,
            tokens,
            scores: vec![],
            token_types,
            merges,
            pre: Some("default".into()),
            special: SpecialTokens { bos: Some(256), eos: Some(257) },
            add_bos: false,
        })
        .unwrap()
    }

    fn hf() -> Box<dyn crate::Tokenizer> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../inferno-formats/tests/fixtures/mlx/tokenizer.json");
        crate::tokenizer_for(&TokenizerSpec::HfJson { path }).unwrap()
    }

    #[test]
    fn merges_apply_in_rank_order() {
        let t = native();
        assert_eq!(t.encode("the", false).unwrap(), vec![259]);
        assert_eq!(t.encode("th", false).unwrap(), vec![258]);
        // " the": pre-tokenized as one piece "Ġthe"; no merge with Ġ exists
        // → [Ġ, the] after t+h, th+e merges.
        assert_eq!(t.encode(" the", false).unwrap(), vec![u32::from(b' '), 259]);
    }

    #[test]
    fn special_tokens_split_literally() {
        let t = native();
        let ids = t.encode("<|bos|>the", false).unwrap();
        assert_eq!(ids, vec![256, 259]);
    }

    #[test]
    fn add_bos_prepends() {
        let t = native();
        assert_eq!(t.encode("the", true).unwrap(), vec![256, 259]);
    }

    #[test]
    fn decode_roundtrips_bytes() {
        let t = native();
        for text in ["the cat", "héllo\nworld", "  spaces  "] {
            let ids = t.encode(text, false).unwrap();
            let bytes: Vec<u8> = ids.iter().flat_map(|&i| t.decode_token(i)).collect();
            assert_eq!(bytes, text.as_bytes(), "{text:?}");
        }
    }

    #[test]
    fn matches_hf_reference_on_fixture_vocab() {
        let (n, h) = (native(), hf());
        for text in ["the", "th the then", "a\nb", "unrelated words", "  the  "] {
            assert_eq!(n.encode(text, false).unwrap(), h.encode(text, false).unwrap(), "{text:?}");
        }
    }

    #[test]
    fn unsupported_pre_id_is_loud_error() {
        let (tokens, merges) = fixtures::tiny_vocab();
        let r = crate::tokenizer_for(&TokenizerSpec::Embedded {
            kind: TokenizerKind::Bpe,
            tokens,
            scores: vec![],
            token_types: vec![],
            merges,
            pre: Some("some-future-model".into()),
            special: SpecialTokens::default(),
            add_bos: false,
        });
        assert!(matches!(r, Err(crate::RuntimeError::Tokenizer(_))));
    }
}
```

Plus a property test file `crates/inferno-runtime/tests/tokenizer_equivalence.rs`:

```rust
//! Native BPE must agree with the HF `tokenizers` crate on the same vocab
//! (spec §Testing). ASCII-focused generator plus unicode spot checks — the
//! fixture vocab covers all 256 bytes, so any disagreement is a merge/pre-
//! tokenizer bug, not a coverage artifact.

use std::path::Path;

use inferno_formats::{SpecialTokens, TokenizerKind, TokenizerSpec};
use inferno_formats::fixtures;
use proptest::prelude::*;

fn native() -> Box<dyn inferno_runtime::Tokenizer> {
    let (tokens, merges) = fixtures::tiny_vocab();
    let mut token_types = vec![1i32; 256];
    token_types.extend([3, 3, 1, 1]);
    inferno_runtime::tokenizer_for(&TokenizerSpec::Embedded {
        kind: TokenizerKind::Bpe,
        tokens,
        scores: vec![],
        token_types,
        merges,
        pre: Some("default".into()),
        special: SpecialTokens { bos: Some(256), eos: Some(257) },
        add_bos: false,
    })
    .unwrap()
}

fn hf() -> Box<dyn inferno_runtime::Tokenizer> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../inferno-formats/tests/fixtures/mlx/tokenizer.json");
    inferno_runtime::tokenizer_for(&TokenizerSpec::HfJson { path }).unwrap()
}

proptest! {
    #[test]
    fn native_bpe_matches_hf(text in "[ -~\\n\\t]{0,64}") {
        let (n, h) = (native(), hf());
        prop_assert_eq!(n.encode(&text, false).unwrap(), h.encode(&text, false).unwrap());
    }

    #[test]
    fn native_bpe_roundtrips_arbitrary_unicode(text in "\\PC{0,32}") {
        let n = native();
        let ids = n.encode(&text, false).unwrap();
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| n.decode_token(i)).collect();
        prop_assert_eq!(bytes, text.as_bytes());
    }
}
```

- [ ] **Step 2: Run to verify failure** — `cargo nextest run -p inferno-runtime bpe` → compile error.

- [ ] **Step 3: Implement `bpe.rs`**

```rust
//! Native byte-level BPE over GGUF-embedded vocab (merge-rank driven, GPT-2
//! byte↔unicode alphabet, fancy-regex pre-tokenization).

use std::collections::HashMap;

use inferno_formats::SpecialTokens;

use crate::tokenizer::bytes::{bpe_token_to_bytes, byte_to_unicode};
use crate::{Result, RuntimeError};

const GGUF_TOKEN_CONTROL: i32 = 3;
const GGUF_TOKEN_USER_DEFINED: i32 = 4;

fn pre_pattern(pre: Option<&str>) -> Result<&'static str> {
    match pre.unwrap_or("default") {
        "default" | "gpt-2" => Ok(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        ),
        "llama-bpe" | "llama3" => Ok(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ),
        "qwen2" => Ok(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ),
        other => Err(RuntimeError::Tokenizer(format!(
            "unsupported pre-tokenizer id {other:?} — add its pattern to bpe.rs"
        ))),
    }
}

pub(crate) struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    tokens: Vec<String>,
    ranks: HashMap<(String, String), usize>,
    /// Control/user-defined tokens, longest first, matched literally.
    specials: Vec<(String, u32)>,
    pre: fancy_regex::Regex,
    special: SpecialTokens,
    add_bos: bool,
}

impl BpeTokenizer {
    pub(crate) fn new(
        tokens: Vec<String>,
        token_types: &[i32],
        merges: &[String],
        pre: Option<&str>,
        special: SpecialTokens,
        add_bos: bool,
    ) -> Result<BpeTokenizer> {
        let pre = fancy_regex::Regex::new(pre_pattern(pre)?)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        let vocab: HashMap<String, u32> =
            tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let ranks = merges
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                m.split_once(' ').map(|(a, b)| ((a.to_string(), b.to_string()), i))
            })
            .collect();
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                matches!(
                    token_types.get(*i),
                    Some(&GGUF_TOKEN_CONTROL) | Some(&GGUF_TOKEN_USER_DEFINED)
                )
            })
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));
        Ok(BpeTokenizer { vocab, tokens, ranks, specials, pre, special, add_bos })
    }

    /// One pre-tokenized piece → token ids via rank-ordered pair merging.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) -> Result<()> {
        let table = byte_to_unicode();
        let mut syms: Vec<String> =
            piece.bytes().map(|b| table[b as usize].to_string()).collect();
        loop {
            let best = syms
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    self.ranks.get(&(w[0].clone(), w[1].clone())).map(|&r| (r, i))
                })
                .min();
            match best {
                Some((_, i)) => {
                    let merged = format!("{}{}", syms[i], syms[i + 1]);
                    syms.splice(i..=i + 1, [merged]);
                }
                None => break,
            }
        }
        for s in &syms {
            match self.vocab.get(s) {
                Some(&id) => out.push(id),
                None => {
                    // Unmergeable multi-char symbol without a vocab entry can
                    // only happen with an inconsistent vocab/merges pair.
                    return Err(RuntimeError::Tokenizer(format!(
                        "symbol {s:?} not in vocab (inconsistent merges)"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Split text on literal special-token occurrences (longest match wins).
    fn split_specials<'a>(&'a self, text: &'a str) -> Vec<(bool, &'a str, u32)> {
        let mut parts = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            let mut earliest: Option<(usize, &(String, u32))> = None;
            for sp in &self.specials {
                if let Some(pos) = rest.find(&sp.0)
                    && earliest.is_none_or(|(e, cur)| pos < e || (pos == e && sp.0.len() > cur.0.len()))
                {
                    earliest = Some((pos, sp));
                }
            }
            match earliest {
                Some((pos, (tok, id))) => {
                    if pos > 0 {
                        parts.push((false, &rest[..pos], 0));
                    }
                    parts.push((true, tok.as_str(), *id));
                    rest = &rest[pos + tok.len()..];
                }
                None => {
                    parts.push((false, rest, 0));
                    break 'outer;
                }
            }
        }
        parts
    }
}

impl crate::Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if add_bos && let Some(b) = self.special.bos {
            ids.push(b);
        }
        for (is_special, chunk, id) in self.split_specials(text) {
            if is_special {
                ids.push(id);
                continue;
            }
            let mut at = 0;
            while at < chunk.len() {
                let m = self
                    .pre
                    .find_from_pos(chunk, at)
                    .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
                let Some(m) = m else { break };
                self.encode_piece(&chunk[m.start()..m.end()], &mut ids)?;
                at = m.end();
            }
        }
        Ok(ids)
    }

    fn decode_token(&self, id: u32) -> Vec<u8> {
        let Some(tok) = self.tokens.get(id as usize) else { return Vec::new() };
        if self.specials.iter().any(|(_, sid)| *sid == id) {
            return tok.as_bytes().to_vec(); // specials pass through literally
        }
        bpe_token_to_bytes(tok).unwrap_or_else(|| tok.as_bytes().to_vec())
    }

    fn bos(&self) -> Option<u32> {
        self.special.bos
    }
    fn eos(&self) -> Option<u32> {
        self.special.eos
    }
    fn default_add_bos(&self) -> bool {
        self.add_bos
    }
}
```

Wire into `tokenizer/mod.rs` (replace the `Embedded` arm):

```rust
        TokenizerSpec::Embedded {
            kind: TokenizerKind::Bpe,
            tokens,
            token_types,
            merges,
            pre,
            special,
            add_bos,
            ..
        } => Ok(Box::new(bpe::BpeTokenizer::new(
            tokens.clone(),
            token_types,
            merges,
            pre.as_deref(),
            special.clone(),
            *add_bos,
        )?)),
        TokenizerSpec::Embedded { kind: TokenizerKind::Spm, .. } => {
            Err(RuntimeError::Tokenizer("spm lands in Task 12".into()))
        }
```

(add `mod bpe;` and the `TokenizerKind` import.)

- [ ] **Step 4: Run tests** — `cargo nextest run -p inferno-runtime` → all PASS, including the HF-equivalence property. If the property finds a divergence, minimize it (proptest shrinks automatically), fix the merge loop or pattern — do **not** loosen the test.

- [ ] **Step 5: Lint and commit**

```bash
mise run lint
git add crates/inferno-runtime
git commit -m "feat(runtime): native byte-level BPE with HF-equivalence property tests"
```

---

### Task 12: Native SPM tokenizer

**Files:**
- Create: `crates/inferno-runtime/src/tokenizer/spm.rs`
- Modify: `crates/inferno-runtime/src/tokenizer/mod.rs` (wire `Embedded { kind: Spm }`)

**Interfaces:**
- Consumes: `TokenizerSpec::Embedded` with `scores` (Task 4), `spm_token_to_bytes` (Task 10).
- Produces: `pub(crate) struct SpmTokenizer` implementing `Tokenizer`. Algorithm (llama.cpp's SPM): escape spaces to `▁`, prepend `▁`, start from UTF-8 characters, repeatedly merge the adjacent pair whose concatenation exists in the vocab with the **highest score**; unknown characters fall back to `<0xNN>` byte tokens (GGUF type 6).

- [ ] **Step 1: Write failing tests** (`#[cfg(test)]` in `spm.rs`)

```rust
#[cfg(test)]
mod tests {
    use inferno_formats::{SpecialTokens, TokenizerKind, TokenizerSpec};

    /// Hand-built SPM vocab: byte-fallback tokens 0..=255 as <0xNN> (type 6),
    /// then ▁(256), h(257), e(258), he(259, score 1.0), ▁he(260, score 2.0),
    /// <s>(261, control), </s>(262, control).
    fn spm() -> Box<dyn crate::Tokenizer> {
        let mut tokens: Vec<String> = (0u16..256).map(|b| format!("<0x{b:02X}>")).collect();
        let mut token_types = vec![6i32; 256];
        let mut scores = vec![0f32; 256];
        for (t, ty, sc) in [
            ("\u{2581}", 1, 0.0),
            ("h", 1, 0.0),
            ("e", 1, 0.0),
            ("he", 1, 1.0),
            ("\u{2581}he", 1, 2.0),
            ("<s>", 3, 0.0),
            ("</s>", 3, 0.0),
        ] {
            tokens.push(t.into());
            token_types.push(ty);
            scores.push(sc);
        }
        crate::tokenizer_for(&TokenizerSpec::Embedded {
            kind: TokenizerKind::Spm,
            tokens,
            scores,
            token_types,
            merges: vec![],
            pre: None,
            special: SpecialTokens { bos: Some(261), eos: Some(262) },
            add_bos: true,
        })
        .unwrap()
    }

    #[test]
    fn merges_by_score_with_space_prefix() {
        // "he" → "▁he" after prefixing → single token 260 (score 2.0 beats
        // merging h+e first).
        assert_eq!(spm().encode("he", false).unwrap(), vec![260]);
    }

    #[test]
    fn unknown_chars_use_byte_fallback() {
        // "Z" is not in the vocab → ▁ then <0x5A>.
        assert_eq!(spm().encode("Z", false).unwrap(), vec![256, 0x5A]);
    }

    #[test]
    fn add_bos_default_is_true_for_spm() {
        let t = spm();
        assert!(t.default_add_bos());
        assert_eq!(t.encode("he", true).unwrap(), vec![261, 260]);
    }

    #[test]
    fn decode_restores_text() {
        let t = spm();
        let ids = t.encode("he he", false).unwrap();
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| t.decode_token(i)).collect();
        // SPM's leading ▁ decodes to a leading space; strip for comparison.
        assert_eq!(String::from_utf8(bytes).unwrap().trim_start(), "he he");
    }
}
```

- [ ] **Step 2: Run to verify failure** — `cargo nextest run -p inferno-runtime spm` → compile error.

- [ ] **Step 3: Implement `spm.rs`**

```rust
//! Native SentencePiece-style tokenizer (score-driven greedy bigram merging,
//! llama.cpp's SPM semantics) for GGUF-embedded Llama-2/Mistral vocabs.

use std::collections::HashMap;

use inferno_formats::SpecialTokens;

use crate::tokenizer::bytes::spm_token_to_bytes;
use crate::{Result, RuntimeError};

const GGUF_TOKEN_CONTROL: i32 = 3;

pub(crate) struct SpmTokenizer {
    tokens: Vec<String>,
    vocab: HashMap<String, u32>,
    scores: Vec<f32>,
    token_types: Vec<i32>,
    special: SpecialTokens,
    add_bos: bool,
}

impl SpmTokenizer {
    pub(crate) fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<i32>,
        special: SpecialTokens,
        add_bos: bool,
    ) -> Result<SpmTokenizer> {
        if scores.len() != tokens.len() {
            return Err(RuntimeError::Tokenizer(format!(
                "spm vocab has {} tokens but {} scores",
                tokens.len(),
                scores.len()
            )));
        }
        let vocab = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        Ok(SpmTokenizer { vocab, scores, token_types, special, add_bos, tokens })
    }

    fn is_special(&self, id: u32) -> bool {
        self.token_types.get(id as usize) == Some(&GGUF_TOKEN_CONTROL)
    }
}

impl crate::Tokenizer for SpmTokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if add_bos && let Some(b) = self.special.bos {
            ids.push(b);
        }
        if text.is_empty() {
            return Ok(ids);
        }
        // SPM whitespace escaping + add_space_prefix (llama.cpp default).
        let escaped = format!("\u{2581}{}", text.replace(' ', "\u{2581}"));
        // Symbols start as single characters.
        let mut syms: Vec<String> = escaped.chars().map(String::from).collect();
        // Greedy: always merge the adjacent pair with the highest score.
        loop {
            let best = syms
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    let cat = format!("{}{}", w[0], w[1]);
                    self.vocab.get(&cat).map(|&id| (self.scores[id as usize], i))
                })
                .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
            match best {
                Some((_, i)) => {
                    let merged = format!("{}{}", syms[i], syms[i + 1]);
                    syms.splice(i..=i + 1, [merged]);
                }
                None => break,
            }
        }
        for s in &syms {
            match self.vocab.get(s) {
                Some(&id) => ids.push(id),
                None => {
                    // Byte fallback: emit <0xNN> per UTF-8 byte.
                    for b in s.bytes() {
                        match self.vocab.get(&format!("<0x{b:02X}>")) {
                            Some(&id) => ids.push(id),
                            None => {
                                return Err(RuntimeError::Tokenizer(format!(
                                    "no vocab entry or byte fallback for {s:?}"
                                )));
                            }
                        }
                    }
                }
            }
        }
        Ok(ids)
    }

    fn decode_token(&self, id: u32) -> Vec<u8> {
        let Some(tok) = self.tokens.get(id as usize) else { return Vec::new() };
        if self.is_special(id) {
            return tok.as_bytes().to_vec();
        }
        spm_token_to_bytes(tok)
    }

    fn bos(&self) -> Option<u32> {
        self.special.bos
    }
    fn eos(&self) -> Option<u32> {
        self.special.eos
    }
    fn default_add_bos(&self) -> bool {
        self.add_bos
    }
}
```

Wire the `Spm` arm in `tokenizer/mod.rs`:

```rust
        TokenizerSpec::Embedded {
            kind: TokenizerKind::Spm, tokens, scores, token_types, special, add_bos, ..
        } => Ok(Box::new(spm::SpmTokenizer::new(
            tokens.clone(),
            scores.clone(),
            token_types.clone(),
            special.clone(),
            *add_bos,
        )?)),
```

- [ ] **Step 4: Run tests** — `cargo nextest run -p inferno-runtime` → PASS.

- [ ] **Step 5: Lint and commit**

```bash
mise run lint
git add crates/inferno-runtime
git commit -m "feat(runtime): native SPM tokenizer with score-driven merging"
```

---

### Task 13: Sampler + Generator (end-to-end tokens)

**Files:**
- Create: `crates/inferno-runtime/src/sampler.rs`, `src/generate.rs`
- Create: `crates/inferno-runtime/tests/end_to_end.rs`
- Modify: `crates/inferno-runtime/src/lib.rs` (export `Sampler`, `Greedy`, `Generator`, `GenStats`)

**Interfaces:**
- Consumes: `build_graph`, `Interpreter`, `KvCache` (7, 9), `tokenizer_for` (10–12).
- Produces (used by Tasks 14–15):

```rust
// sampler.rs
pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> u32;
}
pub struct Greedy;   // argmax, lowest-index tie-break — fully deterministic

// generate.rs
pub struct GenStats { pub prompt_tokens: usize, pub generated: usize,
                      pub prefill_secs: f64, pub decode_secs: f64 }
pub struct Generator { /* desc, graph, interp, tokenizer, max_seq_len */ }
impl Generator {
    /// load → build graph → tokenizer_for. max_seq_len capped by
    /// hp.context_length when the model declares one (>0).
    pub fn load(model: &Path, max_seq_len: usize) -> Result<Generator>;
    pub fn encode(&self, text: &str) -> Result<Vec<u32>>;   // default add_bos
    /// Fresh KV per call. Streams UTF-8-safe byte chunks to on_bytes.
    /// Stops at EOS or max_tokens. Returns generated token ids + stats.
    pub fn generate(
        &mut self, prompt: &str, max_tokens: usize,
        sampler: &mut dyn Sampler, on_bytes: &mut dyn FnMut(&[u8]),
    ) -> Result<(Vec<u32>, GenStats)>;
    /// Full-sequence logits (teacher forcing) — used by diff (Task 15).
    pub fn full_logits(&mut self, tokens: &[u32]) -> Result<inferno_graph::Tensor>;
    pub fn vocab_size(&self) -> usize;
}
// generate.rs also (pub(crate)): struct Utf8Buffer { fn push(&mut self, bytes: &[u8]) -> Vec<u8> }
```

- [ ] **Step 1: Write failing tests**

`sampler.rs` `#[cfg(test)]`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_argmax_lowest_index_tie_break() {
        let mut g = Greedy;
        assert_eq!(g.sample(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(g.sample(&[0.5, 0.9, 0.9]), 1); // tie → lowest index
        assert_eq!(g.sample(&[f32::NEG_INFINITY, -1.0]), 1);
    }
}
```

`generate.rs` `#[cfg(test)]` (Utf8Buffer only):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_buffer_holds_split_codepoints() {
        let mut b = Utf8Buffer::default();
        let euro = "€".as_bytes(); // 3 bytes
        assert_eq!(b.push(&euro[..1]), b"");
        assert_eq!(b.push(&euro[1..2]), b"");
        assert_eq!(b.push(&euro[2..]), "€".as_bytes());
        assert_eq!(b.push(b"ab"), b"ab");
    }

    #[test]
    fn utf8_buffer_replaces_invalid_bytes() {
        let mut b = Utf8Buffer::default();
        // 0xFF can never start a UTF-8 sequence → replacement char.
        assert_eq!(b.push(&[0xFF, b'a']), "\u{FFFD}a".as_bytes());
    }
}
```

`tests/end_to_end.rs`:

```rust
use std::path::Path;

use inferno_runtime::{Generator, Greedy};

fn fixture(p: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferno-formats/tests/fixtures").join(p)
}

fn generate_ids(model: &Path) -> Vec<u32> {
    let mut g = Generator::load(model, 64).unwrap();
    let mut sink = Vec::new();
    let (ids, stats) = g
        .generate("the", 8, &mut Greedy, &mut |b| sink.extend_from_slice(b))
        .unwrap();
    assert_eq!(stats.generated, ids.len());
    assert!(stats.prompt_tokens >= 1);
    ids
}

#[test]
fn gguf_fixture_generates_deterministic_tokens() {
    let a = generate_ids(&fixture("tiny.gguf"));
    let b = generate_ids(&fixture("tiny.gguf"));
    assert_eq!(a, b);
    assert!(!a.is_empty());
}

#[test]
fn gguf_and_mlx_generate_identical_tokens() {
    // Spec acceptance: the two formats of the same model produce the same
    // greedy tokens (same effective weights; see Task 9's logit differential).
    assert_eq!(generate_ids(&fixture("tiny.gguf")), generate_ids(&fixture("mlx")));
}

#[test]
fn max_tokens_bounds_generation() {
    let mut g = Generator::load(&fixture("tiny.gguf"), 64).unwrap();
    let (ids, _) = g.generate("the", 3, &mut Greedy, &mut |_| {}).unwrap();
    assert!(ids.len() <= 3);
}

#[test]
fn prompt_longer_than_max_seq_len_is_typed_error() {
    let mut g = Generator::load(&fixture("tiny.gguf"), 2).unwrap();
    let err = g.generate("the cat sat on the mat", 4, &mut Greedy, &mut |_| {});
    assert!(matches!(err, Err(inferno_runtime::RuntimeError::PromptTooLong { .. })));
}
```

- [ ] **Step 2: Run to verify failure** — `cargo nextest run -p inferno-runtime` → compile errors.

- [ ] **Step 3: Implement**

`sampler.rs`:

```rust
//! Sampling. M1 ships greedy only; the trait is the M4 extension point
//! (temperature/top-k/top-p slot in without touching the generation loop).

pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> u32;
}

pub struct Greedy;

impl Sampler for Greedy {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, v) in logits.iter().enumerate() {
            if *v > logits[best] {
                best = i; // strict > keeps the lowest index on ties
            }
        }
        best as u32
    }
}
```

`generate.rs`:

```rust
//! The generation loop: tokenize → prefill → [sample → decode]* → stream.

use std::path::Path;
use std::time::Instant;

use inferno_formats::{DType, ModelDesc, load_desc};
use inferno_graph::{Graph, Interpreter, KvCache, Tensor, build_graph};

use crate::sampler::Sampler;
use crate::tokenizer::{Tokenizer, tokenizer_for};
use crate::{Result, RuntimeError};

/// Buffers streamed token bytes, emitting only complete UTF-8 sequences.
/// Invalid bytes (impossible from a well-formed vocab, cheap to guard)
/// become U+FFFD.
#[derive(Default)]
pub(crate) struct Utf8Buffer {
    pending: Vec<u8>,
}

impl Utf8Buffer {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<u8> {
        self.pending.extend_from_slice(bytes);
        let mut out = Vec::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(_) => {
                    out.append(&mut self.pending);
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.extend_from_slice(&self.pending[..valid]);
                    match e.error_len() {
                        None => {
                            // Incomplete tail — keep it pending.
                            self.pending.drain(..valid);
                            return out;
                        }
                        Some(bad) => {
                            out.extend_from_slice("\u{FFFD}".as_bytes());
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }
}

pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
}

pub struct Generator {
    desc: ModelDesc,
    graph: Graph,
    interp: Interpreter,
    tokenizer: Box<dyn Tokenizer>,
    max_seq_len: usize,
}

impl Generator {
    pub fn load(model: &Path, max_seq_len: usize) -> Result<Generator> {
        let desc = load_desc(model)?;
        let graph = build_graph(&desc)?;
        let spec = desc.tokenizer.as_ref().ok_or(RuntimeError::NoTokenizer)?;
        let tokenizer = tokenizer_for(spec)?;
        let ctx = desc.hyperparams.context_length as usize;
        let max_seq_len = if ctx > 0 { max_seq_len.min(ctx) } else { max_seq_len };
        Ok(Generator { desc, graph, interp: Interpreter::new(), tokenizer, max_seq_len })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, self.tokenizer.default_add_bos())
    }

    pub fn vocab_size(&self) -> usize {
        self.desc.hyperparams.vocab_size as usize
    }

    /// Single full-sequence pass returning logits at every position
    /// (teacher forcing / diff harness).
    pub fn full_logits(&mut self, tokens: &[u32]) -> Result<Tensor> {
        let mut kv = KvCache::new(&self.graph, self.max_seq_len)?;
        if tokens.len() > self.max_seq_len {
            return Err(RuntimeError::PromptTooLong { got: tokens.len(), max: self.max_seq_len });
        }
        Ok(self.interp.run(&self.desc, &self.graph, tokens, &mut kv)?)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampler: &mut dyn Sampler,
        on_bytes: &mut dyn FnMut(&[u8]),
    ) -> Result<(Vec<u32>, GenStats)> {
        let prompt_ids = self.encode(prompt)?;
        if prompt_ids.is_empty() || prompt_ids.len() >= self.max_seq_len {
            return Err(RuntimeError::PromptTooLong {
                got: prompt_ids.len(),
                max: self.max_seq_len,
            });
        }
        let mut kv = KvCache::new(&self.graph, self.max_seq_len)?;
        let vocab = self.vocab_size();
        let eos = self.tokenizer.eos();
        let mut buf = Utf8Buffer::default();
        let mut out_ids = Vec::new();

        let t0 = Instant::now();
        let logits = self.interp.run(&self.desc, &self.graph, &prompt_ids, &mut kv)?;
        let prefill_secs = t0.elapsed().as_secs_f64();
        let mut last = logits.data[(prompt_ids.len() - 1) * vocab..].to_vec();

        let t1 = Instant::now();
        for _ in 0..max_tokens {
            let next = sampler.sample(&last);
            if Some(next) == eos {
                break;
            }
            out_ids.push(next);
            let chunk = buf.push(&self.tokenizer.decode_token(next));
            if !chunk.is_empty() {
                on_bytes(&chunk);
            }
            if kv.len() + 1 > self.max_seq_len {
                break; // context full
            }
            let step = self.interp.run(&self.desc, &self.graph, &[next], &mut kv)?;
            last = step.data;
        }
        let stats = GenStats {
            prompt_tokens: prompt_ids.len(),
            generated: out_ids.len(),
            prefill_secs,
            decode_secs: t1.elapsed().as_secs_f64(),
        };
        Ok((out_ids, stats))
    }
}
```

`lib.rs` additions:

```rust
mod generate;
mod sampler;

pub use generate::{GenStats, Generator};
pub use sampler::{Greedy, Sampler};
```

- [ ] **Step 4: Run tests** — `cargo nextest run -p inferno-runtime` → all PASS. The GGUF-vs-MLX identical-tokens test is the spec's executable acceptance for the two-formats boundary.

- [ ] **Step 5: Full suite, lint, commit**

```bash
mise run test
mise run lint
git add crates/inferno-runtime
git commit -m "feat(runtime): greedy sampler and streaming Generator — first tokens out"
```

---

### Task 14: CLI `inferno run`

**Files:**
- Create: `cli/src/run.rs`
- Create: `cli/tests/run.rs`
- Modify: `cli/src/main.rs` (subcommand), `cli/Cargo.toml` (`inferno-runtime` dep)

**Interfaces:**
- Consumes: `Generator`, `Greedy`, `GenStats` (Task 13).
- Produces: `inferno run <model> --prompt <text> [--max-tokens N] [--max-seq-len N]` — tokens stream to stdout, `tok/s` summary to stderr, exit 0/1.

- [ ] **Step 1: Write failing test** (`cli/tests/run.rs`)

```rust
use assert_cmd::Command;
use predicates::prelude::*;

fn fixture(p: &str) -> String {
    format!("{}/../crates/inferno-formats/tests/fixtures/{p}", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn run_streams_tokens_from_gguf_fixture() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args(["run", &fixture("tiny.gguf"), "--prompt", "the", "--max-tokens", "4"])
        .assert()
        .success()
        .stderr(predicate::str::contains("decode:"));
}

#[test]
fn run_works_on_mlx_dir() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args(["run", &fixture("mlx"), "--prompt", "the", "--max-tokens", "2"])
        .assert()
        .success();
}

#[test]
fn run_reports_model_errors_cleanly() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args(["run", "/nonexistent.gguf", "--prompt", "x"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"));
}
```

- [ ] **Step 2: Run to verify failure** — `cargo nextest run -p inferno run` → fails (no subcommand).

- [ ] **Step 3: Implement**

`cli/Cargo.toml`: add `inferno-runtime.workspace = true` to `[dependencies]`.

`cli/src/run.rs`:

```rust
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use inferno_runtime::{Generator, Greedy};

pub fn run(model: &Path, prompt: &str, max_tokens: usize, max_seq_len: usize) -> ExitCode {
    let mut generator = match Generator::load(model, max_seq_len) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut stdout = std::io::stdout().lock();
    let result = generator.generate(prompt, max_tokens, &mut Greedy, &mut |bytes| {
        let _ = stdout.write_all(bytes);
        let _ = stdout.flush();
    });
    match result {
        Ok((_, stats)) => {
            let _ = writeln!(stdout);
            eprintln!(
                "prefill: {} tok in {:.1}s ({:.2} tok/s) | decode: {} tok in {:.1}s ({:.2} tok/s)",
                stats.prompt_tokens,
                stats.prefill_secs,
                stats.prompt_tokens as f64 / stats.prefill_secs.max(1e-9),
                stats.generated,
                stats.decode_secs,
                stats.generated as f64 / stats.decode_secs.max(1e-9),
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
```

`cli/src/main.rs` — add the variant and match arm:

```rust
    /// Generate text from a prompt (M1: reference interpreter — slow by
    /// design; the compiler arrives in M3).
    Run {
        /// Path to a .gguf file, an MLX directory, or a .safetensors file.
        model: PathBuf,
        /// Prompt text (raw completion; no chat template).
        #[arg(long, short)]
        prompt: String,
        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// KV-cache capacity (clamped to the model's context length).
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,
    },
```

```rust
        Command::Run { model, prompt, max_tokens, max_seq_len } => {
            run::run(&model, &prompt, max_tokens, max_seq_len)
        }
```

(plus `mod run;` at the top.)

- [ ] **Step 4: Run tests** — `cargo nextest run -p inferno` → PASS.

- [ ] **Step 5: Lint and commit**

```bash
mise run lint
git add cli
git commit -m "feat(cli): inferno run — interpreter-backed streaming generation"
```

---

### Task 15: Teacher-forced differential (library + hidden CLI)

**Files:**
- Create: `crates/inferno-runtime/src/diff.rs`
- Create: `cli/src/diff.rs`
- Modify: `crates/inferno-runtime/src/lib.rs`, `cli/src/main.rs`, `cli/Cargo.toml` (`serde`/`serde_json` deps)

**Interfaces:**
- Consumes: `Generator::full_logits/encode/vocab_size` (13), `tolerance::LOGIT_TIE_EPSILON` (6).
- Produces:

```rust
// inferno-runtime diff.rs
pub struct Mismatch { pub position: usize, pub expected: u32, pub got: u32,
                      pub gap: f32, pub top: Vec<(u32, f32)> }   // our top-5 (id, logit)
pub struct DiffOutcome { pub checked: usize, pub matched: usize, pub ties: usize,
                         pub min_gap: f32, pub mismatches: Vec<Mismatch> }
impl DiffOutcome { pub fn passed(&self) -> bool }  // mismatches.is_empty()
/// Teacher-forced: one full_logits pass over prompt++forced; at each forced
/// position compare our argmax vs the forced token; our top-2 gap <
/// LOGIT_TIE_EPSILON ⇒ tie, not mismatch (spec §Nightly tier).
pub fn teacher_forced(gen: &mut Generator, prompt_tokens: &[u32], forced: &[u32])
    -> Result<DiffOutcome>;
```

- CLI: `inferno diff --model <path> --prompt-file <txt> --tokens-file <json>` (hidden from help). `tokens-file` format: `{"prompt_tokens": [..], "generated_tokens": [..]}` (produced by the nightly script from llama.cpp). Exit 0 = pass; prints per-position detail on mismatch; **fails loudly at position 0 if our prompt tokenization ≠ llama.cpp's**.

- [ ] **Step 1: Write failing tests** (`crates/inferno-runtime/tests/teacher_forced.rs`)

```rust
use std::path::Path;

use inferno_runtime::{Generator, Greedy, teacher_forced};

fn generator() -> Generator {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../inferno-formats/tests/fixtures/tiny.gguf");
    Generator::load(&p, 64).unwrap()
}

#[test]
fn own_greedy_output_agrees_perfectly() {
    // Feed our own greedy generation back as the forced sequence: every
    // position must match (same weights, same math).
    let mut g = generator();
    let prompt = g.encode("the").unwrap();
    let (ids, _) = g.generate("the", 6, &mut Greedy, &mut |_| {}).unwrap();
    let out = teacher_forced(&mut g, &prompt, &ids).unwrap();
    assert!(out.passed(), "mismatches: {:?}", out.mismatches);
    assert_eq!(out.checked, ids.len());
    assert_eq!(out.matched + out.ties, out.checked);
}

#[test]
fn wrong_forced_token_is_reported_with_position_and_top5() {
    let mut g = generator();
    let prompt = g.encode("the").unwrap();
    let (mut ids, _) = g.generate("the", 6, &mut Greedy, &mut |_| {}).unwrap();
    // Corrupt position 2 with a token that is definitely not the argmax
    // AND whose gap exceeds the tie epsilon (pick the argmin instead).
    let logits = g.full_logits(&[prompt.clone(), ids.clone()].concat()).unwrap();
    let vocab = g.vocab_size();
    let row = &logits.data[(prompt.len() + 1) * vocab..(prompt.len() + 2) * vocab];
    let worst = row
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;
    ids[2] = worst;
    let out = teacher_forced(&mut g, &prompt, &ids).unwrap();
    // Positions after the corruption legitimately diverge (different
    // history); the FIRST mismatch must be exactly position 2.
    assert!(!out.passed());
    let first = &out.mismatches[0];
    assert_eq!(first.position, 2);
    assert_eq!(first.expected, worst);
    assert_eq!(first.top.len(), 5);
}
```

- [ ] **Step 2: Run to verify failure** — compile error (`teacher_forced` missing).

- [ ] **Step 3: Implement**

`crates/inferno-runtime/src/diff.rs`:

```rust
//! Teacher-forced differential against an external reference (llama.cpp).
//! One full-sequence pass; per-position argmax comparison with a tie
//! tolerance computed from OUR top-2 logit gap — no reference-logit
//! extraction needed (spec §Nightly tier).

use inferno_graph::tolerance::LOGIT_TIE_EPSILON;

use crate::generate::Generator;
use crate::Result;

#[derive(Debug)]
pub struct Mismatch {
    pub position: usize,
    pub expected: u32,
    pub got: u32,
    pub gap: f32,
    pub top: Vec<(u32, f32)>,
}

#[derive(Debug)]
pub struct DiffOutcome {
    pub checked: usize,
    pub matched: usize,
    pub ties: usize,
    pub min_gap: f32,
    pub mismatches: Vec<Mismatch>,
}

impl DiffOutcome {
    pub fn passed(&self) -> bool {
        self.mismatches.is_empty()
    }
}

fn top_n(row: &[f32], n: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_by(|a, b| row[*b as usize].total_cmp(&row[*a as usize]));
    idx.into_iter().take(n).map(|i| (i, row[i as usize])).collect()
}

pub fn teacher_forced(
    gen: &mut Generator,
    prompt_tokens: &[u32],
    forced: &[u32],
) -> Result<DiffOutcome> {
    if prompt_tokens.is_empty() {
        // Position 0 would have no predicting row (logits[p] predict p+1).
        return Err(crate::RuntimeError::Tokenizer(
            "teacher forcing needs a non-empty prompt".into(),
        ));
    }
    let full: Vec<u32> = prompt_tokens.iter().chain(forced).copied().collect();
    let logits = gen.full_logits(&full)?;
    let vocab = gen.vocab_size();
    let mut out = DiffOutcome {
        checked: forced.len(),
        matched: 0,
        ties: 0,
        min_gap: f32::INFINITY,
        mismatches: Vec::new(),
    };
    for (i, &expected) in forced.iter().enumerate() {
        // logits row at position p predict token p+1; forced[i] sits at
        // absolute position prompt.len()+i, predicted by row prompt.len()+i-1.
        let row_idx = prompt_tokens.len() + i - 1;
        let row = &logits.data[row_idx * vocab..(row_idx + 1) * vocab];
        let top = top_n(row, 5);
        let got = top[0].0;
        let gap = top[0].1 - top[1].1;
        out.min_gap = out.min_gap.min(gap);
        if got == expected {
            out.matched += 1;
        } else if gap < LOGIT_TIE_EPSILON {
            out.ties += 1;
        } else {
            out.mismatches.push(Mismatch { position: i, expected, got, gap, top });
        }
    }
    Ok(out)
}
```

`lib.rs`: `mod diff;` + `pub use diff::{DiffOutcome, Mismatch, teacher_forced};`

`cli/src/diff.rs`:

```rust
use std::path::Path;
use std::process::ExitCode;

use inferno_runtime::{Generator, teacher_forced};
use serde::Deserialize;

#[derive(Deserialize)]
struct TokensFile {
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
}

pub fn diff(model: &Path, prompt_file: &Path, tokens_file: &Path) -> ExitCode {
    let inner = || -> Result<bool, Box<dyn std::error::Error>> {
        let prompt = std::fs::read_to_string(prompt_file)?;
        let tf: TokensFile = serde_json::from_str(&std::fs::read_to_string(tokens_file)?)?;
        let mut generator = Generator::load(model, 4096)?;

        // Gate 0: our tokenization must match the reference's exactly, or
        // every later position is comparing different sequences.
        let ours = generator.encode(&prompt)?;
        if ours != tf.prompt_tokens {
            let first =
                ours.iter().zip(&tf.prompt_tokens).position(|(a, b)| a != b).unwrap_or(
                    ours.len().min(tf.prompt_tokens.len()),
                );
            eprintln!(
                "TOKENIZATION MISMATCH at prompt position {first}:\n  ours:  {ours:?}\n  llama: {:?}",
                tf.prompt_tokens
            );
            return Ok(false);
        }
        println!("prompt tokenization matches ({} tokens)", ours.len());

        let out = teacher_forced(&mut generator, &tf.prompt_tokens, &tf.generated_tokens)?;
        println!(
            "teacher-forced: {} checked, {} matched, {} ties, min top-2 gap {:.4}",
            out.checked, out.matched, out.ties, out.min_gap
        );
        for m in &out.mismatches {
            eprintln!(
                "MISMATCH at generated position {}: expected {}, got {} (gap {:.4})\n  our top-5: {:?}",
                m.position, m.expected, m.got, m.gap, m.top
            );
        }
        Ok(out.passed())
    };
    match inner() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
```

`cli/Cargo.toml`: add `serde.workspace = true` and `serde_json.workspace = true`.

`cli/src/main.rs` — hidden subcommand:

```rust
    /// Teacher-forced differential vs an external reference (nightly harness).
    #[command(hide = true)]
    Diff {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt_file: PathBuf,
        #[arg(long)]
        tokens_file: PathBuf,
    },
```

```rust
        Command::Diff { model, prompt_file, tokens_file } => {
            diff::diff(&model, &prompt_file, &tokens_file)
        }
```

- [ ] **Step 4: Run tests** — `cargo nextest run -p inferno-runtime teacher_forced && cargo nextest run -p inferno` → PASS. Also smoke the CLI end-to-end against ourselves:

```bash
printf 'the' > /tmp/prompt.txt
cargo run -p inferno -- run crates/inferno-formats/tests/fixtures/tiny.gguf -p "the" --max-tokens 4
# Build tokens.json from our own output via a tiny jq-free check is not needed —
# the library test already covers self-agreement; this smoke just proves the
# subcommand parses files and exits 0/1 correctly:
echo '{"prompt_tokens": [116, 104, 101], "generated_tokens": []}' > /tmp/tokens.json
cargo run -p inferno -- diff --model crates/inferno-formats/tests/fixtures/tiny.gguf \
  --prompt-file /tmp/prompt.txt --tokens-file /tmp/tokens.json
```

Expected: "the" encodes as `[259]` (one merged token), NOT `[116,104,101]` — so this exits 1 with a TOKENIZATION MISMATCH report. Then fix the file to `{"prompt_tokens": [259], "generated_tokens": []}` and expect exit 0 with "0 checked".

- [ ] **Step 5: Lint and commit**

```bash
mise run lint
git add crates/inferno-runtime cli
git commit -m "feat(runtime,cli): teacher-forced differential with tie tolerance"
```

---

### Task 16: Nightly differential harness + docs

**Files:**
- Create: `scripts/nightly-differential.sh`
- Modify: `mise.toml` (`[tasks.differential]`), `.github/workflows/nightly.yml` (job), `ARCHITECTURE.md`, `AGENTS.md`

**Interfaces:**
- Consumes: `inferno diff` (15), `inferno run` (14), devenv-pinned llama.cpp (`llama-server`), `jq` + `curl` (present on ubuntu runners and dev machines; the script checks and fails with an install hint).
- Produces: `mise run differential` — downloads Qwen2.5-0.5B-Instruct (GGUF Q8_0 + MLX bf16) into `~/.cache/inferno-tests/`, extracts llama.cpp's greedy tokens via `llama-server`'s `/tokenize` and `/completion` (with `return_tokens`), runs the teacher-forced diff, then an MLX smoke run.

- [ ] **Step 1: Write `scripts/nightly-differential.sh`**

```bash
#!/usr/bin/env bash
# Nightly teacher-forced differential vs llama.cpp (spec §Nightly tier).
# Requires: llama-server on PATH (devenv shell), curl, jq, cargo.
set -euo pipefail

for tool in llama-server curl jq cargo; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool (run inside 'devenv shell')" >&2; exit 2; }
done

CACHE="${INFERNO_TEST_MODEL_DIR:-$HOME/.cache/inferno-tests}"
mkdir -p "$CACHE/qwen2.5-0.5b-mlx"
HF="https://huggingface.co"

GGUF="$CACHE/qwen2.5-0.5b-instruct-q8_0.gguf"
[ -f "$GGUF" ] || curl -fL --retry 3 -o "$GGUF" \
  "$HF/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"

MLX="$CACHE/qwen2.5-0.5b-mlx"
for f in config.json model.safetensors tokenizer.json; do
  [ -f "$MLX/$f" ] || curl -fL --retry 3 -o "$MLX/$f" \
    "$HF/mlx-community/Qwen2.5-0.5B-Instruct-bf16/resolve/main/$f"
done

PROMPT="The capital of France is"
N_TOKENS=64
PORT=18080
printf '%s' "$PROMPT" > "$CACHE/prompt.txt"

# Single-threaded, no warmup randomness: greedy decoding is deterministic
# for a pinned llama.cpp build + fixed thread count.
llama-server -m "$GGUF" -t 1 --port "$PORT" --host 127.0.0.1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 1
done

PROMPT_TOKENS=$(curl -sf "http://127.0.0.1:$PORT/tokenize" \
  -d "$(jq -n --arg c "$PROMPT" '{content: $c}')" | jq -c '.tokens')
GENERATED=$(curl -sf "http://127.0.0.1:$PORT/completion" \
  -d "$(jq -n --arg p "$PROMPT" --argjson n "$N_TOKENS" \
        '{prompt: $p, n_predict: $n, temperature: 0, top_k: 1, samplers: ["top_k"],
          return_tokens: true, cache_prompt: false}')" | jq -c '.tokens')
kill "$SERVER_PID"; trap - EXIT

jq -n --argjson p "$PROMPT_TOKENS" --argjson g "$GENERATED" \
  '{prompt_tokens: $p, generated_tokens: $g}' > "$CACHE/tokens.json"
echo "llama.cpp: $(jq length <<<"$PROMPT_TOKENS") prompt + $(jq length <<<"$GENERATED") generated tokens"

echo "=== teacher-forced differential (GGUF) ==="
cargo run --release -p inferno -- diff \
  --model "$GGUF" --prompt-file "$CACHE/prompt.txt" --tokens-file "$CACHE/tokens.json"

echo "=== MLX smoke run ==="
cargo run --release -p inferno -- run "$MLX" --prompt "$PROMPT" --max-tokens 16

echo "differential: PASS"
```

Make it executable: `chmod +x scripts/nightly-differential.sh`.

- [ ] **Step 2: mise task** (`mise.toml`, after `[tasks.fuzz]`)

```toml
[tasks.differential]
description = "Nightly teacher-forced differential vs llama.cpp (downloads a real model; run inside devenv shell)"
run = "bash scripts/nightly-differential.sh"
```

- [ ] **Step 3: nightly workflow job** (`.github/workflows/nightly.yml`, after `onboarding`; mirrors onboarding's devenv setup because llama.cpp comes from nix)

```yaml
  # Teacher-forced greedy differential vs the devenv-pinned llama.cpp on a
  # real quantized model (M1 spec §Nightly tier).
  differential:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: DeterminateSystems/nix-installer-action@v16
      - uses: DeterminateSystems/magic-nix-cache-action@v9
      - run: nix profile install nixpkgs#devenv
      - uses: jdx/mise-action@v2
      - uses: Swatinem/rust-cache@v2
      - uses: actions/cache@v4
        with:
          path: ~/.cache/inferno-tests
          key: inferno-test-models-qwen25-05b-v1
      - run: devenv shell -- mise run differential
```

- [ ] **Step 4: Verify locally** (this is the one step that needs network + ~30 min of interpreter time; run it once)

```bash
devenv shell -- mise run differential
```

Expected: prompt tokenization matches; `64 checked, N matched, M ties, 0 mismatches`; MLX smoke prints 16 tokens of plausible text. If ties are frequent or a mismatch's reported gap sits just above `LOGIT_TIE_EPSILON` (0.05), record the observed `min_gap` distribution in the PR description — the epsilon lives in `inferno-graph/src/tolerance.rs` and is expected to be tuned against real gap data (AGENTS.md note below). If the interpreter is too slow for 64 tokens (>2 h), reduce `N_TOKENS` in the script to 32 — the check's power is per-position (spec §Risks).

- [ ] **Step 5: Docs**

`ARCHITECTURE.md`: add `inferno-graph` and `inferno-runtime` to the crate map with one boundary line each — graph: "IR + builder + scalar oracle; tolerances live here and nowhere else"; runtime: "tokenizer/sampling/generation; drives the interpreter in M1, compiled entry points from M3". Follow the existing document's format.

`AGENTS.md` — append to "Non-obvious constraints":

```markdown
- **Rope style is coupled to weight layout:** GGUF llama-arch files carry
  *row-permuted* Q/K weights (Interleaved rope); MLX/HF files are unpermuted
  (HalfSplit). `HyperParams::rope_style` records which; the fixture
  differential (`inferno-graph/tests/differential.rs`) guards the coupling.
  Never "simplify" one side without the other.
- **Embedded and JSON tokenizer fixtures must stay equivalent:**
  `fixtures::tiny_vocab()` feeds both the GGUF metadata and
  `mlx/tokenizer.json`; the BPE equivalence property tests depend on it.
- **`LOGIT_TIE_EPSILON`** (`inferno-graph/src/tolerance.rs`) is tuned against
  the gap distributions printed by `mise run differential` — adjust it with
  observed data, never to make a red nightly green without understanding the
  divergence.
```

`README.md`: if it lists example commands, add `inferno run` next to `inspect` (task names stay the single source of truth — only mention the subcommand, don't re-spell workflows).

- [ ] **Step 6: Full verification and commit**

```bash
mise run test
mise run lint
git add scripts mise.toml .github/workflows/nightly.yml ARCHITECTURE.md AGENTS.md README.md
git commit -m "ci(nightly): teacher-forced llama.cpp differential; M1 docs"
```

Note: editing `mise.toml` invalidates the CI tool cache (documented warning at the top of that file) — the next CI run rebuilds cargo tools (~10 min), then self-heals.

---

## Milestone acceptance (from the spec)

1. `inferno run <qwen2.5-0.5b-instruct-q8_0.gguf> -p "…"` streams coherent text (slowly) — verified manually in Task 16 Step 4.
2. The MLX build of the same model runs identically (same pipeline) — Task 16's smoke run.
3. Nightly teacher-forced differential passes at N=64 — Task 16.
4. Blocking tier green within budget; fuzz targets re-run after parser changes — Tasks 3–5.




