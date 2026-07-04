//! Parser limits (spec §Security: allocation limits on untrusted input).
//! Sized generously above real models (vocab ~300k, merges ~500k) but far
//! below anything that could exhaust memory from a header alone.

pub const MAX_TENSORS: u64 = 65_536;
pub const MAX_KV_PAIRS: u64 = 65_536;
pub const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_ARRAY_ELEMS: u64 = 10_000_000;
pub const MAX_DIMS: u32 = 8;
pub const MAX_ARRAY_DEPTH: u32 = 4;
/// safetensors spec caps the JSON header at 100 MB.
pub const MAX_ST_HEADER_BYTES: u64 = 100 * 1024 * 1024;
