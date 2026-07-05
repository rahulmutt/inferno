//! Model-file parsing: GGUF and MLX/safetensors → format-agnostic [`ModelDesc`].
//!
//! Parsers treat every input byte as untrusted (see docs/threat-model.md):
//! bounded reads, checked arithmetic, and no `unsafe` anywhere in this crate.
#![forbid(unsafe_code)]

mod data;
mod desc;
mod error;
pub mod fixtures;
pub mod gguf;
pub mod limits;
pub(crate) mod mlx;
pub(crate) mod names;
pub mod quant;
mod read;
pub mod safetensors;

pub use data::read_tensor_bytes;
pub use desc::{Architecture, DType, HyperParams, ModelDesc, TensorDesc};
pub use error::{FormatError, Result};

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
