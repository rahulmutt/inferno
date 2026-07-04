//! Bounded little-endian primitive readers over `io::Read`.
//! Truncation maps to `Malformed`, never a panic; strings are length-limited.

use std::io::Read;
use std::mem::size_of;

use crate::{FormatError, Result, limits};

fn fill<R: Read>(r: &mut R, buf: &mut [u8], context: &'static str) -> Result<()> {
    r.read_exact(buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            FormatError::Malformed {
                context,
                detail: "unexpected end of input".into(),
            }
        } else {
            FormatError::Io(e)
        }
    })
}

macro_rules! reader {
    ($name:ident, $ty:ty) => {
        pub(crate) fn $name<R: Read>(r: &mut R) -> Result<$ty> {
            let mut buf = [0u8; size_of::<$ty>()];
            fill(r, &mut buf, stringify!($ty))?;
            Ok(<$ty>::from_le_bytes(buf))
        }
    };
}

reader!(read_u8, u8);
reader!(read_u16, u16);
reader!(read_u32, u32);
reader!(read_u64, u64);
reader!(read_i8, i8);
reader!(read_i16, i16);
reader!(read_i32, i32);
reader!(read_i64, i64);
reader!(read_f32, f32);
reader!(read_f64, f64);

pub(crate) fn read_bool<R: Read>(r: &mut R) -> Result<bool> {
    match read_u8(r)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(FormatError::Malformed {
            context: "bool",
            detail: format!("expected 0 or 1, got {other}"),
        }),
    }
}

/// GGUF string: u64 LE byte length + UTF-8 payload.
pub(crate) fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)?;
    if len > limits::MAX_STRING_BYTES {
        return Err(FormatError::LimitExceeded {
            what: "string length",
            got: len,
            limit: limits::MAX_STRING_BYTES,
        });
    }
    let mut buf = vec![0u8; len as usize];
    fill(r, &mut buf, "string payload")?;
    String::from_utf8(buf).map_err(|_| FormatError::Malformed {
        context: "string payload",
        detail: "invalid utf-8".into(),
    })
}
