//! Raw tensor-byte access. Offsets and lengths come from an untrusted header,
//! so every value is validated against the real file length before reading.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::{FormatError, ModelDesc, Result, TensorDesc};

pub fn read_tensor_bytes(desc: &ModelDesc, tensor: &TensorDesc) -> Result<Vec<u8>> {
    let malformed = |detail: String| FormatError::Malformed {
        context: "tensor data",
        detail,
    };
    let idx = tensor.file_index as usize;
    let (path, base) = match (
        desc.weight_files.get(idx),
        desc.data_section_offsets.get(idx),
    ) {
        (Some(p), Some(b)) => (p, *b),
        _ => {
            return Err(malformed(format!(
                "{}: file index {idx} out of range",
                tensor.name
            )));
        }
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
