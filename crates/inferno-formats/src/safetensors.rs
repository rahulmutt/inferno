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
    r.read_exact(&mut json)
        .map_err(|_| FormatError::Malformed {
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
        let entry: StEntry = serde_json::from_value(v).map_err(FormatError::Json)?;
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
        if let Some(expect) = dtype.byte_len(n_elems)
            && expect != span
        {
            return Err(FormatError::Malformed {
                context: "safetensors data_offsets",
                detail: format!("{name}: span {span} != dtype size {expect}"),
            });
        }
        let name = crate::names::canonical_hf(&name).unwrap_or(name);
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
        let e = tensors
            .iter()
            .find(|t| t.name == "token_embed.weight")
            .unwrap();
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
        let mut b = (crate::limits::MAX_ST_HEADER_BYTES + 1)
            .to_le_bytes()
            .to_vec();
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

    #[test]
    fn rejects_shape_overflow() {
        // shape[0] * shape[1] overflows u64: must hit the checked_mul
        // try_fold path in the element-count computation, not panic.
        let json =
            r#"{"w": {"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,16]}}"#;
        assert!(matches!(
            parse(&mut Cursor::new(wrap(json)), 0),
            Err(crate::FormatError::Malformed { .. })
        ));
    }
}
