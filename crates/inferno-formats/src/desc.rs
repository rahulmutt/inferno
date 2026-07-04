use std::path::PathBuf;

use serde::Serialize;

/// Tensor element type. Quant formats are first-class dtypes (spec §Graph IR).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[allow(non_camel_case_types)]
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
        if !n_elems.is_multiple_of(block_elems) {
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
