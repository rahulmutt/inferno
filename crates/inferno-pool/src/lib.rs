//! Persistent fork-join thread pool + the five `inferno_par_{gemv,gemm,attention,attention_heads,token_loop}`
//! dispatchers that M4b.1+ generated code calls by symbol. Kernels stay single-threaded
//! (spec boundary rule: parallelism is the caller's job — this crate IS
//! that caller): the dispatcher splits a GEMV's row range into 8-row-aligned
//! shards, so each output row is computed entirely by one thread with the
//! kernel's fixed combine order and **thread count never changes output
//! bits**.

pub mod error;
pub mod pool;
pub mod probe;
pub mod prof;
pub mod shard;

pub use error::PoolError;
pub use pool::{
    AttnBlockFn, AttnFn, AttnHeadsJob, AttnHspanFn, AttnJob, GemmFn, GemvFn, Pool, TokenBodyFn,
};
pub use probe::{bandwidth_curve, knee_at_fraction};
pub use prof::PoolProfSnapshot;
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

/// Enable/disable the M4b.12 dispatch-split recording. No-op without the
/// `pool-profile` feature — shipping builds carry none of the instrument.
pub fn set_pool_profiling(on: bool) {
    #[cfg(feature = "pool-profile")]
    prof::set_enabled(on);
    #[cfg(not(feature = "pool-profile"))]
    let _ = on;
}

/// Zero the global pool's dispatch-split accounting (between profile
/// phases). No-op without the feature or before `init_global`.
pub fn pool_prof_reset() {
    #[cfg(feature = "pool-profile")]
    if let Some(p) = GLOBAL.get() {
        p.prof_reset();
    }
}

/// Snapshot the global pool's dispatch-split accounting. `None` without
/// the feature or before `init_global`.
pub fn pool_prof_snapshot() -> Option<PoolProfSnapshot> {
    #[cfg(feature = "pool-profile")]
    {
        GLOBAL.get().map(|p| p.prof_snapshot())
    }
    #[cfg(not(feature = "pool-profile"))]
    None
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

/// Host dispatcher for tiled prefill attention (M4b.8). Same
/// single-dispatcher guard + serial fallback as [`inferno_par_gemv`];
/// shares `DISPATCH_CLAIMED` deliberately — within one forward pass GEMV,
/// GEMM and attention dispatches are issued serially and never overlap,
/// so one guard suffices. `m <= 1` (decode-shaped calls, T=1 prefill
/// tiles) takes a direct serial path with no CAS and no job publish
/// (decode does not call this dispatcher — since M4b.11 its codegen calls
/// inferno_par_attention_heads; the m <= 1 arm here covers T=1 prefill
/// tiles). On the CAS-loss (or
/// uninitialized-pool) path this runs the serial full-range token loop,
/// bit-identical to the pooled path since each token's out row is
/// computed by a single kernel invocation either way.
///
/// A panic inside the dispatcher or kernel aborts the process at this
/// `extern "C"` boundary — there is no unwind across FFI.
///
/// # Safety
/// Same contract as [`Pool::par_attention`] over tokens `0..m`;
/// additionally `kernel` must be a valid non-null function pointer with
/// the M4b.3 attention ABI, and the KV cache must already contain every
/// position `< pos0 + m` (the tile's append loop runs before this call).
/// Generated code guarantees all of this by construction (M3 trust model).
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_par_attention(
    kernel: AttnBlockFn,
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    pos0: usize,
    m: usize,
    kv_base: usize,
    v_off: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_stride: usize,
    out_stride: usize,
) {
    if m == 0 {
        return;
    }
    let job = pool::AttnJob {
        kernel,
        out,
        q,
        kv,
        pos0,
        kv_base,
        v_off,
        kv_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        q_stride,
        out_stride,
    };
    if m == 1 {
        // SAFETY: forwarding the caller's contract for the single token.
        unsafe { pool::run_attn_span(&job, 0, 1) };
        return;
    }
    match GLOBAL.get() {
        Some(p) => {
            if DISPATCH_CLAIMED
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: forwarding the caller's contract unchanged.
                unsafe { p.par_attention(&job, m) };
                DISPATCH_CLAIMED.store(false, Ordering::Release);
            } else {
                // Lost the race for the pool's single dispatcher slot: run
                // serially over the full token range instead of overlapping
                // another thread's in-flight pool dispatch.
                // SAFETY: forwarding the caller's contract for the full range.
                unsafe { pool::run_attn_span(&job, 0, m) };
            }
        }
        // SAFETY: forwarding the caller's contract for the full range.
        None => unsafe { pool::run_attn_span(&job, 0, m) },
    }
}

/// Host dispatcher for head-sharded decode attention (M4b.11): ONE query
/// token, the head range `0..n_heads` sharded align-1 across up to
/// `min(active_threads, decode_threads)` lanes (decode work — the
/// `INFERNO_DECODE_THREADS` override applies, like `inferno_par_gemv`).
/// Same single-dispatcher guard + serial fallback as its four siblings;
/// shares `DISPATCH_CLAIMED` deliberately — within one forward pass all
/// pool dispatches are issued serially and never overlap. On the CAS-loss
/// (or uninitialized-pool) path this runs one serial hspan call over the
/// full head range, bit-identical to the pooled path since each head is
/// computed by unchanged per-head math either way.
///
/// A panic inside the dispatcher or kernel aborts the process at this
/// `extern "C"` boundary — there is no unwind across FFI.
///
/// # Safety
/// Same contract as [`Pool::par_attention_heads`]; additionally `kernel`
/// must be a valid non-null function pointer with the M4b.11 head-span
/// attention ABI, and the KV cache must already contain every position
/// `<= pos` (decode codegen appends this token's k/v first). Generated
/// code guarantees all of this by construction (M3 trust model).
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_par_attention_heads(
    kernel: AttnHspanFn,
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    pos: usize,
    kv_base: usize,
    v_off: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    if n_heads == 0 {
        return;
    }
    let job = pool::AttnHeadsJob {
        kernel,
        out,
        q,
        kv,
        pos,
        kv_base,
        v_off,
        kv_dim,
        n_heads,
        n_kv_heads,
        head_dim,
    };
    match GLOBAL.get() {
        Some(p) => {
            if DISPATCH_CLAIMED
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: forwarding the caller's contract unchanged.
                unsafe { p.par_attention_heads(&job) };
                DISPATCH_CLAIMED.store(false, Ordering::Release);
            } else {
                // Lost the race for the pool's single dispatcher slot: run
                // serially over the full head range instead of overlapping
                // another thread's in-flight pool dispatch.
                // SAFETY: forwarding the caller's contract for the full range.
                unsafe { pool::run_attn_heads_span(&job, 0, n_heads) };
            }
        }
        // SAFETY: forwarding the caller's contract for the full range.
        None => unsafe { pool::run_attn_heads_span(&job, 0, n_heads) },
    }
}

/// Host dispatcher for outlined serial-tail token loops (M4b.9). Same
/// single-dispatcher guard + serial fallback as [`inferno_par_gemv`];
/// shares `DISPATCH_CLAIMED` deliberately — within one forward pass all
/// pool dispatches are issued serially and never overlap, so one guard
/// suffices. `m <= 1` (the T=1 prefill tile tail) takes a direct body
/// call with no CAS and no job publish — decode never calls this
/// dispatcher at all (its codegen lowers ops inline, single-token). On
/// the CAS-loss (or uninitialized-pool) path this runs the body once
/// over the full token range, bit-identical to the pooled path since
/// each token's rows are written by a single body invocation either way.
///
/// A panic inside the dispatcher or body aborts the process at this
/// `extern "C"` boundary — there is no unwind across FFI.
///
/// # Safety
/// Same contract as [`Pool::par_token_loop`]; additionally `body` must
/// be a valid non-null function pointer with the M4b.9 token-body ABI
/// and `ctx` the pack that body expects. Generated code guarantees all
/// of this by construction (M3 trust model).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_par_token_loop(body: TokenBodyFn, ctx: *const u8, m: usize) {
    if m == 0 {
        return;
    }
    if m == 1 {
        // SAFETY: forwarding the caller's contract for the single token.
        unsafe { body(ctx, 0, 1) };
        return;
    }
    match GLOBAL.get() {
        Some(p) => {
            if DISPATCH_CLAIMED
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: forwarding the caller's contract unchanged.
                unsafe { p.par_token_loop(body, ctx, m) };
                DISPATCH_CLAIMED.store(false, Ordering::Release);
            } else {
                // Lost the race for the pool's single dispatcher slot: run
                // the body serially over the full token range instead of
                // overlapping another thread's in-flight pool dispatch.
                // SAFETY: forwarding the caller's contract for the full range.
                unsafe { body(ctx, 0, m) };
            }
        }
        // SAFETY: forwarding the caller's contract for the full range.
        None => unsafe { body(ctx, 0, m) },
    }
}
