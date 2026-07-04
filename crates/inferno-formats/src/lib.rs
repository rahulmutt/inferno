//! Model-file parsing: GGUF and MLX/safetensors → format-agnostic [`ModelDesc`].
//!
//! Parsers treat every input byte as untrusted (see docs/threat-model.md):
//! bounded reads, checked arithmetic, and no `unsafe` anywhere in this crate.
#![forbid(unsafe_code)]
