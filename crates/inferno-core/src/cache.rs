//! Content-addressed cache key + on-disk artifact cache directory.
//!
//! `cache_key` hashes the model bytes, the target description, the requested
//! `max_seq_len`, this crate's version, and the codegen host-ABI version into
//! a single stable hex digest; `cache_dir` maps that key to
//! `$XDG_CACHE_HOME/inferno/<key>` (falling back to `$HOME/.cache` or
//! `./.cache`).

use std::path::{Path, PathBuf};

use inferno_target::TargetDesc;
use sha2::{Digest, Sha256};

use crate::Result;

/// Hex SHA-256 digest of `bytes`.
pub fn content_hash(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// Read the bytes that identify `model`'s content for hashing.
///
/// A single-file model (GGUF, `.safetensors`) is read directly. A
/// directory-based model (MLX: `config.json` + `model.safetensors` +
/// `tokenizer.json`) has no single canonical byte stream, so its content is
/// the concatenation of every regular file's `(relative path, length, bytes)`
/// beneath it, walked in a stable (lexicographically path-sorted) order —
/// deterministic across runs/platforms and sensitive to any file in the
/// directory changing, being added, or being removed.
pub(crate) fn read_model_bytes(model: &Path) -> Result<Vec<u8>> {
    if std::fs::metadata(model)?.is_dir() {
        let mut files = Vec::new();
        collect_files(model, &mut files)?;
        files.sort();
        let mut buf = Vec::new();
        for path in files {
            let rel = path.strip_prefix(model).unwrap_or(&path);
            buf.extend_from_slice(rel.to_string_lossy().as_bytes());
            buf.push(0);
            let bytes = std::fs::read(&path)?;
            buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(&bytes);
        }
        Ok(buf)
    } else {
        Ok(std::fs::read(model)?)
    }
}

/// Recursively collect every regular file beneath `dir` into `out`.
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}

/// Content-addressed key for a compiled artifact: a stable hex digest of the
/// model bytes, the target description, `max_seq_len`, this crate's version,
/// and the codegen host-ABI version. Deterministic for identical inputs;
/// changes whenever any input changes.
pub fn cache_key(
    model_path: &Path,
    target: &TargetDesc,
    max_seq_len: usize,
    opts: &inferno_codegen::CompileOptions,
) -> Result<String> {
    let model_bytes = read_model_bytes(model_path)?;
    let mut h = Sha256::new();
    h.update(content_hash(&model_bytes).as_bytes());
    // TargetDesc is Debug+deterministic; hash its debug form (isa, features, caches).
    h.update(format!("{target:?}").as_bytes());
    h.update((max_seq_len as u64).to_le_bytes());
    h.update(env!("CARGO_PKG_VERSION").as_bytes());
    h.update(inferno_codegen::HOST_ABI_VERSION.as_bytes());
    // Profiling and tile size change the emitted artifact.
    h.update([opts.profile as u8]);
    h.update((opts.prefill_tile as u64).to_le_bytes());
    Ok(format!("{:x}", h.finalize()))
}

/// On-disk directory for the artifact cached under `key`:
/// `$XDG_CACHE_HOME/inferno/<key>`, falling back to `$HOME/.cache` or
/// `./.cache` when neither environment variable is set.
pub fn cache_dir(key: &str) -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("inferno").join(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferno_target::TargetDesc;
    use std::path::Path;

    #[test]
    fn key_is_stable_and_input_sensitive() {
        let t = TargetDesc::detect().unwrap();
        let m = Path::new("../inferno-formats/tests/fixtures/tiny.gguf");
        let k1 = cache_key(m, &t, 64, &inferno_codegen::CompileOptions::default()).unwrap();
        let k2 = cache_key(m, &t, 64, &inferno_codegen::CompileOptions::default()).unwrap();
        assert_eq!(k1, k2); // deterministic
        let k3 = cache_key(m, &t, 128, &inferno_codegen::CompileOptions::default()).unwrap();
        assert_ne!(k1, k3); // max_seq_len is part of the key

        let k_prof = cache_key(
            m,
            &t,
            64,
            &inferno_codegen::CompileOptions {
                profile: true,
                prefill_tile: 64,
            },
        )
        .unwrap();
        assert_ne!(k1, k_prof); // profiling is part of the key

        let k_tile = cache_key(
            m,
            &t,
            64,
            &inferno_codegen::CompileOptions {
                profile: false,
                prefill_tile: 32,
            },
        )
        .unwrap();
        assert_ne!(k1, k_tile); // prefill_tile is part of the key
    }

    #[test]
    fn content_hash_changes_with_bytes() {
        assert_ne!(content_hash(b"a"), content_hash(b"b"));
    }
}
