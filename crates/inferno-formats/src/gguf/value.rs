//! GGUF metadata value tree (types 0–12 of the GGUF spec).

use std::io::Read;

use crate::read::*;
use crate::{FormatError, Result, limits};

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn parse<R: Read>(r: &mut R, type_id: u32, depth: u32) -> Result<Self> {
        Ok(match type_id {
            0 => Self::U8(read_u8(r)?),
            1 => Self::I8(read_i8(r)?),
            2 => Self::U16(read_u16(r)?),
            3 => Self::I16(read_i16(r)?),
            4 => Self::U32(read_u32(r)?),
            5 => Self::I32(read_i32(r)?),
            6 => Self::F32(read_f32(r)?),
            7 => Self::Bool(read_bool(r)?),
            8 => Self::String(read_string(r)?),
            9 => {
                if depth >= limits::MAX_ARRAY_DEPTH {
                    return Err(FormatError::LimitExceeded {
                        what: "array nesting depth",
                        got: u64::from(depth) + 1,
                        limit: u64::from(limits::MAX_ARRAY_DEPTH),
                    });
                }
                let elem_type = read_u32(r)?;
                let count = read_u64(r)?;
                if count > limits::MAX_ARRAY_ELEMS {
                    return Err(FormatError::LimitExceeded {
                        what: "array element count",
                        got: count,
                        limit: limits::MAX_ARRAY_ELEMS,
                    });
                }
                // No preallocation from the untrusted count: elements are
                // parsed one at a time, so a lying count hits EOF cheaply.
                let mut items = Vec::new();
                for _ in 0..count {
                    items.push(Self::parse(r, elem_type, depth + 1)?);
                }
                Self::Array(items)
            }
            10 => Self::U64(read_u64(r)?),
            11 => Self::I64(read_i64(r)?),
            12 => Self::F64(read_f64(r)?),
            other => {
                return Err(FormatError::Malformed {
                    context: "metadata value type",
                    detail: format!("unknown type id {other}"),
                });
            }
        })
    }

    /// Widen any integer value to u64 (metadata writers vary integer widths).
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v.into()),
            Self::U16(v) => Some(v.into()),
            Self::U32(v) => Some(v.into()),
            Self::U64(v) => Some(v),
            Self::I8(v) => u64::try_from(v).ok(),
            Self::I16(v) => u64::try_from(v).ok(),
            Self::I32(v) => u64::try_from(v).ok(),
            Self::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn array_len(&self) -> Option<u64> {
        match self {
            Self::Array(v) => Some(v.len() as u64),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse_bytes(type_id: u32, bytes: &[u8]) -> crate::Result<GgufValue> {
        GgufValue::parse(&mut Cursor::new(bytes), type_id, 0)
    }

    #[test]
    fn scalar_values() {
        assert_eq!(
            parse_bytes(4, &7u32.to_le_bytes()).unwrap().as_u64(),
            Some(7)
        );
        assert_eq!(
            parse_bytes(10, &9u64.to_le_bytes()).unwrap().as_u64(),
            Some(9)
        );
        assert_eq!(
            parse_bytes(6, &1.5f32.to_le_bytes()).unwrap().as_f32(),
            Some(1.5)
        );
        assert_eq!(parse_bytes(7, &[1]).unwrap(), GgufValue::Bool(true));
    }

    #[test]
    fn string_value() {
        // GGUF string: u64 LE length + UTF-8 bytes.
        let mut b = 5u64.to_le_bytes().to_vec();
        b.extend_from_slice(b"hello");
        assert_eq!(parse_bytes(8, &b).unwrap().as_str(), Some("hello"));
    }

    #[test]
    fn string_over_limit_rejected() {
        let b = (crate::limits::MAX_STRING_BYTES + 1).to_le_bytes();
        assert!(matches!(
            parse_bytes(8, &b),
            Err(crate::FormatError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn array_of_u32() {
        // array: u32 elem type + u64 count + payload
        let mut b = 4u32.to_le_bytes().to_vec(); // elem type = u32
        b.extend_from_slice(&3u64.to_le_bytes()); // count = 3
        for v in [1u32, 2, 3] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let v = parse_bytes(9, &b).unwrap();
        assert_eq!(v.array_len(), Some(3));
    }

    #[test]
    fn array_count_over_limit_rejected() {
        let mut b = 4u32.to_le_bytes().to_vec();
        b.extend_from_slice(&(crate::limits::MAX_ARRAY_ELEMS + 1).to_le_bytes());
        assert!(matches!(
            parse_bytes(9, &b),
            Err(crate::FormatError::LimitExceeded { .. })
        ));
    }

    #[test]
    fn deep_array_nesting_rejected() {
        // arrays-of-arrays beyond MAX_ARRAY_DEPTH must be rejected, not recurse.
        // Build: array(elem=array(elem=array(...))) with depth 5, count 1 each.
        let mut b: Vec<u8> = Vec::new();
        for _ in 0..5 {
            b.extend_from_slice(&9u32.to_le_bytes()); // elem type = array
            b.extend_from_slice(&1u64.to_le_bytes()); // count = 1
        }
        assert!(parse_bytes(9, &b).is_err());
    }

    #[test]
    fn truncated_input_is_error_not_panic() {
        assert!(parse_bytes(4, &[0x01]).is_err()); // u32 needs 4 bytes
    }

    #[test]
    fn unknown_type_id_rejected() {
        assert!(parse_bytes(99, &[]).is_err());
    }
}
