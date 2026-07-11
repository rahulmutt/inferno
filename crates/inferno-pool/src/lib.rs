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
pub use pool::{AttnFn, AttnJob, GemmFn, GemvFn, Pool};
pub use shard::{SHARD_ALIGN, shard_table, shard_table_aligned};

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

static GLOBAL: OnceLock<Pool> = OnceLock::new();

/// Single-dispatcher guard for the global pool. `Pool::par_gemv`'s protocol
/// requires calls not to overlap, but two `CompiledBackend`s on two user
/// threads can both reach `inferno_par_gemv` through the safe API
/// concurrently. Claimed with `compare_exchange` before the pool is used;
/// the thread that loses the race never touches the pool at all — see
/// `inferno_par_gemv` below.
static DISPATCH_CLAIMED: AtomicBool = AtomicBool::new(false);

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

/// Cap the global pool's decode-phase (`inferno_par_gemv`) parallelism
/// (clamped to `1..=capacity`). Prefill (`inferno_par_gemm`) is unaffected.
/// Returns `false` (and does nothing) if [`init_global`] has not run.
pub fn set_global_decode_threads(n: usize) -> bool {
    match GLOBAL.get() {
        Some(p) => {
            p.set_decode_threads(n);
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
/// `Pool::par_gemv` is single-dispatcher, but two safe-API callers (e.g. two
/// `CompiledBackend`s on two user threads, each via `Backend::forward`) can
/// reach this symbol concurrently against the same process-global pool. A
/// static dispatch guard (`compare_exchange`) makes that safe: whichever
/// call claims the guard drives the pool; the other, on losing the race,
/// falls back to the same direct serial kernel call used when the pool
/// isn't initialized at all, over the full row range — correctness and
/// bit-identity hold by construction, since that path is identical to the
/// uninitialized-pool fallback. Concurrent dispatches never corrupt the
/// pool's job/epoch/remaining protocol; the loser just runs serial. The
/// accepted cost is one uncontended CAS per GEMV.
///
/// A panic inside the dispatcher or kernel aborts the process at this
/// `extern "C"` boundary — there is no unwind across FFI.
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
        Some(pool) => {
            if DISPATCH_CLAIMED
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: forwarding the caller's contract unchanged.
                unsafe { pool.par_gemv(kernel, y, xq, w, k, rows) };
                DISPATCH_CLAIMED.store(false, Ordering::Release);
            } else {
                // Lost the race for the pool's single dispatcher slot:
                // run serially over the full range instead of overlapping
                // another thread's in-flight pool dispatch.
                // SAFETY: forwarding the caller's contract for the full range.
                unsafe { kernel(y, xq, w, k, 0, rows) };
            }
        }
        // SAFETY: forwarding the caller's contract for the full range.
        None => unsafe { kernel(y, xq, w, k, 0, rows) },
    }
}

/// Host dispatcher for batched prefill GEMM (M4b.2). Same single-dispatcher
/// guard + serial fallback as [`inferno_par_gemv`]; shares `DISPATCH_CLAIMED`
/// deliberately — within one forward pass GEMV and GEMM are issued serially
/// and never overlap, so one guard suffices. On the CAS-loss (or
/// uninitialized-pool) path this runs one serial kernel call over the full
/// row range for all `m` tokens, bit-identical to the pooled path since each
/// row is computed by a single kernel invocation either way.
///
/// A panic inside the dispatcher or kernel aborts the process at this
/// `extern "C"` boundary — there is no unwind across FFI.
///
/// # Safety
/// Same contract as [`Pool::par_gemm`]; additionally `kernel` must be a
/// valid non-null function pointer with the GEMM ABI. Generated code
/// guarantees all of this by construction (M3 trust model).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_par_gemm(
    kernel: GemmFn,
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    m: usize,
    rows: usize,
) {
    if rows == 0 || m == 0 {
        return;
    }
    match GLOBAL.get() {
        Some(pool) => {
            if DISPATCH_CLAIMED
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: forwarding the caller's contract unchanged.
                unsafe { pool.par_gemm(kernel, y, xq, w, k, m, rows) };
                DISPATCH_CLAIMED.store(false, Ordering::Release);
            } else {
                // Lost the race for the pool's single dispatcher slot: run
                // serially over the full range (all m tokens) instead of
                // overlapping another thread's in-flight pool dispatch.
                // SAFETY: forwarding the caller's contract for the full range.
                unsafe { kernel(y, xq, w, k, m, rows, 0, rows) };
            }
        }
        // SAFETY: forwarding the caller's contract for the full range.
        None => unsafe { kernel(y, xq, w, k, m, rows, 0, rows) },
    }
}
