//! GGUF header parsing (versions 2 and 3): magic, metadata KVs, tensor infos.
//! Tensor *data* is never read here.

pub(crate) mod value;

use std::collections::BTreeMap;
use std::io::Read;

use value::GgufValue;

use crate::read::*;
use crate::{Architecture, DType, FormatError, HyperParams, ModelDesc, Result, TensorDesc, limits};

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
    r.read_exact(&mut magic)
        .map_err(|_| FormatError::BadMagic { expected: "GGUF" })?;
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
                detail: format!(
                    "{}: offset {} not {}-aligned",
                    t.name, t.data_offset, alignment
                ),
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

#[cfg(test)]
mod tests {
    use super::parse;
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
        // Deviation from the plan text: the plan's `bytes.len() / 3` cut lands
        // in the zero-filled data section, which the parser never reads, so
        // parsing succeeded. Instead, cut inside the tensor-info block by
        // construction:
        // Find where the header ends, then cut inside the tensor-info block:
        // the aligned data-section offset is at most 31 bytes past the end of
        // the tensor infos, so cutting 40 bytes earlier is always mid-info.
        let hdr_end = parse(&mut Cursor::new(&bytes))
            .unwrap()
            .data_section_offsets[0] as usize;
        let cut = &bytes[..hdr_end - 40];
        assert!(parse(&mut Cursor::new(cut)).is_err());
    }
}
