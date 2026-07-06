//! Persistent fork-join thread pool + the `inferno_par_gemv` dispatcher
//! that M4b.1 generated code calls by symbol. Kernels stay single-threaded
//! (spec boundary rule: parallelism is the caller's job — this crate IS
//! that caller): the dispatcher splits a GEMV's row range into 8-row-aligned
//! shards, so each output row is computed entirely by one thread with the
//! kernel's fixed combine order and **thread count never changes output
//! bits**.

pub mod error;
pub mod pool;
pub mod shard;

pub use error::PoolError;
pub use pool::{GemvFn, Pool};
pub use shard::{SHARD_ALIGN, shard_table};

use std::sync::OnceLock;

static GLOBAL: OnceLock<Pool> = OnceLock::new();

/// Initialize the process-global pool with `threads.max(1)` lanes. Idempotent
/// for the same count; a different count is [`PoolError::AlreadyInitialized`].
pub fn init_global(threads: usize) -> Result<(), PoolError> {
    let t = threads.max(1);
    let pool = GLOBAL.get_or_init(|| Pool::new(t));
    if pool.capacity() != t {
        return Err(PoolError::AlreadyInitialized {
            current: pool.capacity(),
            requested: t,
        });
    }
    Ok(())
}

/// Cap the global pool's per-dispatch parallelism (clamped to
/// `1..=capacity`). Returns `false` (and does nothing) if [`init_global`]
/// has not run.
pub fn set_global_active_threads(n: usize) -> bool {
    match GLOBAL.get() {
        Some(p) => {
            p.set_active_threads(n);
            true
        }
        None => false,
    }
}

/// The host symbol M4b.1 generated code calls for every GEMV (resolved at
/// `dlopen` time via `inferno-core`'s symbol retention, like the kernels).
/// Splits `0..rows` across the global pool; with no initialized pool it
/// degrades to one serial kernel call, so a host that never initializes the
/// pool still works — just single-threaded.
///
/// # Safety
/// Same contract as [`Pool::par_gemv`]; additionally `kernel` must be a
/// valid non-null function pointer with the M2 GEMV ABI. Generated code
/// guarantees all of this by construction (M3 trust model).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_par_gemv(
    kernel: GemvFn,
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    rows: usize,
) {
    if rows == 0 {
        return;
    }
    match GLOBAL.get() {
        // SAFETY: forwarding the caller's contract unchanged.
        Some(pool) => unsafe { pool.par_gemv(kernel, y, xq, w, k, rows) },
        // SAFETY: forwarding the caller's contract for the full range.
        None => unsafe { kernel(y, xq, w, k, 0, rows) },
    }
}
