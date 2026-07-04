//! Model-file parsing: GGUF and MLX/safetensors → format-agnostic [`ModelDesc`].
//!
//! Parsers treat every input byte as untrusted (see docs/threat-model.md):
//! bounded reads, checked arithmetic, and no `unsafe` anywhere in this crate.
#![forbid(unsafe_code)]

mod desc;
mod error;
pub mod fixtures;
pub mod gguf;
pub mod limits;
#[allow(dead_code)] // consumed by load_desc in Task 8
pub(crate) mod mlx;
mod read;
pub mod safetensors;

pub use desc::{Architecture, DType, HyperParams, ModelDesc, TensorDesc};
pub use error::{FormatError, Result};
