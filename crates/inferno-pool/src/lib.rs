//! Persistent fork-join thread pool + the `inferno_par_gemv` dispatcher
//! that M4b.1 generated code calls by symbol. Kernels stay single-threaded
//! (spec boundary rule: parallelism is the caller's job — this crate IS
//! that caller): the dispatcher splits a GEMV's row range into 8-row-aligned
//! shards, so each output row is computed entirely by one thread with the
//! kernel's fixed combine order and **thread count never changes output
//! bits**.

pub mod pool;
pub mod shard;

pub use pool::{GemvFn, Pool};
pub use shard::{SHARD_ALIGN, shard_table};
