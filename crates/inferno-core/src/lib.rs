//! `inferno-core`: the embeddable engine. Task 13 adds the content-addressed
//! cache key + cache directory; later tasks add the dlopen loader, backend,
//! and CLI. See docs/superpowers/specs/2026-07-05-m3-compiler-design.md.

pub mod cache;
pub mod error;

pub use cache::{cache_dir, cache_key, content_hash};
pub use error::{CoreError, Result};
