use thiserror::Error;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("{what}: got {got} bytes, expected {expected}")]
    SizeMismatch {
        what: &'static str,
        got: usize,
        expected: usize,
    },
    #[error("k={k} is not a positive multiple of the {block}-element block")]
    BadK { k: usize, block: usize },
    #[error("rows must be non-zero")]
    ZeroRows,
    #[error("row range {row_start}..{row_end} invalid for {rows} rows")]
    BadRowRange {
        row_start: usize,
        row_end: usize,
        rows: usize,
    },
    #[error("size overflow computing a buffer length")]
    Overflow,
    #[error("kernel ISA `{isa}` is not available on this CPU")]
    IsaUnavailable { isa: &'static str },
}

pub type Result<T> = std::result::Result<T, KernelError>;
