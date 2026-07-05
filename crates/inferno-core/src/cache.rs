//! Content-addressed cache key + on-disk artifact cache directory.
//!
//! `cache_key` hashes the model bytes, the target description, the requested
//! `max_seq_len`, and this crate's version into a single stable hex digest;
//! `cache_dir` maps that key to `$XDG_CACHE_HOME/inferno/<key>` (falling back
//! to `$HOME/.cache` or `./.cache`).

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

/// Content-addressed key for a compiled artifact: a stable hex digest of the
/// model bytes, the target description, `max_seq_len`, and this crate's
/// version. Deterministic for identical inputs; changes whenever any input
/// changes.
pub fn cache_key(model_path: &Path, target: &TargetDesc, max_seq_len: usize) -> Result<String> {
    let model_bytes = std::fs::read(model_path)?;
    let mut h = Sha256::new();
    h.update(content_hash(&model_bytes).as_bytes());
    // TargetDesc is Debug+deterministic; hash its debug form (isa, features, caches).
    h.update(format!("{target:?}").as_bytes());
    h.update((max_seq_len as u64).to_le_bytes());
    h.update(env!("CARGO_PKG_VERSION").as_bytes());
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
        let k1 = cache_key(m, &t, 64).unwrap();
        let k2 = cache_key(m, &t, 64).unwrap();
        assert_eq!(k1, k2); // deterministic
        let k3 = cache_key(m, &t, 128).unwrap();
        assert_ne!(k1, k3); // max_seq_len is part of the key
    }

    #[test]
    fn content_hash_changes_with_bytes() {
        assert_ne!(content_hash(b"a"), content_hash(b"b"));
    }
}
