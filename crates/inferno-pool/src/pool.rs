//! The persistent fork-join pool. One job at a time: the dispatching thread
//! publishes a fully partitioned GEMV, bumps an epoch, runs shard 0 itself,
//! and spins until workers drain the remaining shards. Workers spin briefly
//! (GEMVs arrive every few hundred µs in the decode hot loop) then park, so
//! idle processes go quiet. No queues, no work-stealing: the shard→thread
//! map is a pure function of `(rows, active_threads)`, deterministic
//! run-to-run.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{JoinHandle, Thread};

use crate::shard::shard_table;

/// The M2 GEMV kernel ABI: `(y, xq, w, k, row_start, row_end)`. Must match
/// `inferno-kernels`' `#[unsafe(no_mangle)]` symbols exactly (the rig in
/// `tests/par_rig.rs` coerces the real symbols to this type, so a drift is
/// a compile error).
pub type GemvFn = unsafe extern "C" fn(*mut f32, *const u8, *const u8, usize, usize, usize);

/// The M4b.2 batched-GEMM kernel ABI:
/// `(y, xq, w, k, m, rows, row_start, row_end)`. `xq` is a panel of `m`
/// quantized activation rows; `y` is `m * rows` f32 laid out token-major
/// (`y[t * rows + r]`). Must match `inferno-kernels`' `inferno_gemm_*`
/// symbols exactly (the rig coerces the real symbol to this type).
pub type GemmFn =
    unsafe extern "C" fn(*mut f32, *const u8, *const u8, usize, usize, usize, usize, usize);

/// The kind of kernel a published [`Job`] carries. A `Copy` enum of `Copy`
/// fields so a worker reads it out of the shared job exactly like the old
/// bare `GemvFn` payload — no change to the epoch/remaining SAFETY protocol.
#[derive(Clone, Copy)]
enum JobKind {
    Gemv {
        kernel: GemvFn,
    },
    Gemm {
        kernel: GemmFn,
        m: usize,
        rows: usize,
    },
}

/// Run one shard's slice `[start, end)` of the current job. For `Gemm` the
/// kernel writes `y[t * rows + r]` for every token `t in 0..m`; those writes
/// are disjoint across shards because shards partition the row range and
/// every token reuses the same partition.
///
/// # Safety
/// The dispatcher's caller contract must cover `[start, end)` (all `m`
/// tokens for `Gemm`); `kind`'s pointers/fields are those the dispatcher
/// published for this epoch.
unsafe fn run_shard(
    kind: &JobKind,
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    start: usize,
    end: usize,
) {
    match *kind {
        // SAFETY: forwarding the caller's contract for the disjoint range.
        JobKind::Gemv { kernel } => unsafe { kernel(y, xq, w, k, start, end) },
        // SAFETY: forwarding the caller's contract; all m tokens, disjoint rows.
        JobKind::Gemm { kernel, m, rows } => unsafe { kernel(y, xq, w, k, m, rows, start, end) },
    }
}

/// Spin iterations before a waiter parks (workers) or yields (dispatcher).
/// ≈50µs of `spin_loop` on current x86 — covers the decode hot loop where
/// GEMVs arrive every few hundred µs, without burning CPU in idle hosts.
/// Named so a real embedding host that objects can tune one constant.
const SPIN_ITERS: u32 = 20_000;

/// The epoch word packs `(counter << PACKED_SHARD_BITS) | worker_shard_count+1`.
/// Workers learn THIS dispatch's shard count from the epoch itself, so a
/// worker whose index holds no shard never reads `job` at all — otherwise a
/// slow shardless worker could race the dispatcher's next `job` write after
/// `remaining` (which only counts shard holders) hit zero.
const PACKED_SHARD_BITS: u32 = 16;
const PACKED_SHARD_MASK: usize = (1 << PACKED_SHARD_BITS) - 1;

struct Job {
    kind: Option<JobKind>,
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    shards: Vec<(usize, usize)>,
}

impl Job {
    fn empty() -> Job {
        Job {
            kind: None,
            y: std::ptr::null_mut(),
            xq: std::ptr::null(),
            w: std::ptr::null(),
            k: 0,
            shards: Vec::new(),
        }
    }
}

struct Slot {
    parked: AtomicBool,
    thread: OnceLock<Thread>,
}

struct Shared {
    /// Packed `(counter, shard_count)` — see `PACKED_SHARD_BITS`.
    epoch: AtomicUsize,
    /// Worker shards not yet completed for the current epoch.
    remaining: AtomicUsize,
    shutdown: AtomicBool,
    /// Per-dispatch parallelism cap (≤ capacity); `Pool::set_active_threads`.
    active: AtomicUsize,
    /// Decode-phase parallelism cap (≤ capacity); `Pool::set_decode_threads`.
    /// `par_gemv` shards over `min(active, decode_cap)` so decode stops past
    /// its bandwidth knee while prefill (`par_gemm`) keeps full `active`.
    decode_cap: AtomicUsize,
    job: UnsafeCell<Job>,
    /// One slot per worker (capacity - 1 of them).
    slots: Vec<Slot>,
}

// SAFETY: `job` (raw pointers + Vec) crosses threads under a strict
// protocol: the dispatcher writes it only while no reader exists (previous
// dispatch fully drained: `remaining == 0`, and shardless workers never
// read `job` — they learn the shard count from the packed epoch). Workers
// read it only between observing a new epoch and decrementing `remaining`.
// All signalling goes through SeqCst atomics.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// A fixed-size fork-join pool of `capacity - 1` workers; the dispatching
/// thread executes shard 0. Dropping the pool shuts workers down and joins
/// them (unit tests create and drop pools freely; the process-global pool
/// in `lib.rs` lives forever).
pub struct Pool {
    shared: Arc<Shared>,
    capacity: usize,
    handles: Vec<JoinHandle<()>>,
}

impl Pool {
    /// Spawn a pool with `threads.max(1)` total lanes (calling thread
    /// included). Blocks until every worker has registered, so `unpark` is
    /// possible from the first dispatch.
    pub fn new(threads: usize) -> Pool {
        let capacity = threads.max(1);
        assert!(
            capacity <= PACKED_SHARD_MASK,
            "pool capacity {capacity} exceeds packed-epoch limit"
        );
        let shared = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            active: AtomicUsize::new(capacity),
            decode_cap: AtomicUsize::new(capacity),
            job: UnsafeCell::new(Job::empty()),
            slots: (0..capacity - 1)
                .map(|_| Slot {
                    parked: AtomicBool::new(false),
                    thread: OnceLock::new(),
                })
                .collect(),
        });
        let handles = (0..capacity - 1)
            .map(|i| {
                let sh = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("inferno-pool-{i}"))
                    .spawn(move || worker_loop(&sh, i))
                    .expect("spawn inferno-pool worker")
            })
            .collect();
        for slot in &shared.slots {
            let mut spins = 0u32;
            while slot.thread.get().is_none() {
                if spins < SPIN_ITERS {
                    spins += 1;
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
        }
        Pool {
            shared,
            capacity,
            handles,
        }
    }

    /// Total lanes (calling thread included).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Cap parallelism for subsequent dispatches to `n.clamp(1, capacity)`
    /// lanes without touching pool threads — how `inferno bench` gets its
    /// t=1 diagnostic row from the same process.
    pub fn set_active_threads(&self, n: usize) {
        self.shared
            .active
            .store(n.clamp(1, self.capacity), Ordering::Relaxed);
    }

    pub fn active_threads(&self) -> usize {
        self.shared.active.load(Ordering::Relaxed)
    }

    /// Cap decode-phase (`par_gemv`) parallelism to `n.clamp(1, capacity)`
    /// lanes. Prefill (`par_gemm`) is unaffected. Defaults to `capacity`
    /// (no cap); `inferno-core` lowers it to the bandwidth-knee heuristic.
    pub fn set_decode_threads(&self, n: usize) {
        self.shared
            .decode_cap
            .store(n.clamp(1, self.capacity), Ordering::Relaxed);
    }

    pub fn decode_threads(&self) -> usize {
        self.shared.decode_cap.load(Ordering::Relaxed)
    }

    /// Run `kernel` over `0..rows`, split across up to `active_threads()`
    /// lanes. Returns after every shard completes. Output is bit-identical
    /// for every thread count: each row is computed entirely by one lane.
    ///
    /// # Safety
    /// `kernel`'s documented contract must hold for `(y, xq, w, k)` over
    /// every row in `0..rows` (`y` valid for `rows` f32 writes; `xq`/`w`
    /// valid packed buffers for this `k`/`rows`; 32-byte alignment where
    /// the kernel requires it), and all buffers must stay live and
    /// otherwise-untouched until this call returns. Calls to `par_gemv` must
    /// not overlap; the pool runs one job at a time, and concurrent dispatches
    /// would corrupt the job/epoch/remaining protocol.
    pub unsafe fn par_gemv(
        &self,
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
        // Decode is bandwidth-bound: cap below prefill's full-core count so
        // sharding stops at its bandwidth knee (M4b.5). `par_gemm` is not capped.
        let active = self.active_threads().min(self.decode_threads());
        let shards = shard_table(rows, active);
        if shards.len() == 1 {
            // SAFETY: caller contract covers the full range.
            unsafe { kernel(y, xq, w, k, 0, rows) };
            return;
        }
        let n_worker = shards.len() - 1;
        let (s0, e0) = shards[0];
        // SAFETY (job write): the previous dispatch ended with
        // `remaining == 0`, and shardless workers never read `job`
        // (packed-epoch protocol) — no reader exists here.
        unsafe {
            *self.shared.job.get() = Job {
                kind: Some(JobKind::Gemv { kernel }),
                y,
                xq,
                w,
                k,
                shards,
            };
        }
        self.shared.remaining.store(n_worker, Ordering::SeqCst);
        let counter =
            (self.shared.epoch.load(Ordering::SeqCst) >> PACKED_SHARD_BITS).wrapping_add(1);
        self.shared.epoch.store(
            (counter << PACKED_SHARD_BITS) | (n_worker + 1),
            Ordering::SeqCst,
        );
        // Wake exactly the workers that hold a shard. SeqCst on both the
        // epoch store above and the workers' `parked` handshake makes the
        // classic lost-wakeup interleaving impossible (a worker that read
        // the old epoch published `parked=true` before that read, so we
        // see the flag and bank an unpark token).
        for slot in &self.shared.slots[..n_worker] {
            if slot.parked.load(Ordering::SeqCst) {
                slot.thread
                    .get()
                    .expect("worker registered in Pool::new")
                    .unpark();
            }
        }
        // SAFETY: caller contract; shard 0 is disjoint from worker shards.
        unsafe { run_shard(&JobKind::Gemv { kernel }, y, xq, w, k, s0, e0) };
        let mut spins = 0u32;
        while self.shared.remaining.load(Ordering::Acquire) != 0 {
            if spins < SPIN_ITERS {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Batched GEMM across up to `active_threads()` lanes; splits `0..rows`
    /// into the same shards as [`par_gemv`]. Each output row — all `m`
    /// tokens — is computed by exactly one lane, so thread count never
    /// changes output bits.
    ///
    /// # Safety
    /// As [`par_gemv`], but for the GEMM ABI: `y` valid for `m * rows` f32
    /// writes (token-major `y[t * rows + r]`), `xq` a panel of `m` quantized
    /// activation rows for this `k`, `w` the packed weights for `(rows, k)`.
    /// All buffers stay live and otherwise-untouched until this returns;
    /// calls must not overlap (one job at a time).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn par_gemm(
        &self,
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
        let active = self.active_threads();
        let shards = shard_table(rows, active);
        if shards.len() == 1 {
            // SAFETY: caller contract covers the full range for all m tokens.
            unsafe { kernel(y, xq, w, k, m, rows, 0, rows) };
            return;
        }
        let n_worker = shards.len() - 1;
        let (s0, e0) = shards[0];
        let kind = JobKind::Gemm { kernel, m, rows };
        // SAFETY (job write): the previous dispatch ended with
        // `remaining == 0`, and shardless workers never read `job`
        // (packed-epoch protocol) — no reader exists here.
        unsafe {
            *self.shared.job.get() = Job {
                kind: Some(kind),
                y,
                xq,
                w,
                k,
                shards,
            };
        }
        self.shared.remaining.store(n_worker, Ordering::SeqCst);
        let counter =
            (self.shared.epoch.load(Ordering::SeqCst) >> PACKED_SHARD_BITS).wrapping_add(1);
        self.shared.epoch.store(
            (counter << PACKED_SHARD_BITS) | (n_worker + 1),
            Ordering::SeqCst,
        );
        // Wake exactly the workers that hold a shard (same handshake as
        // par_gemv; see there for the lost-wakeup argument).
        for slot in &self.shared.slots[..n_worker] {
            if slot.parked.load(Ordering::SeqCst) {
                slot.thread
                    .get()
                    .expect("worker registered in Pool::new")
                    .unpark();
            }
        }
        // SAFETY: caller contract; shard 0 is disjoint from worker shards.
        unsafe { run_shard(&kind, y, xq, w, k, s0, e0) };
        let mut spins = 0u32;
        while self.shared.remaining.load(Ordering::Acquire) != 0 {
            if spins < SPIN_ITERS {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        for slot in &self.shared.slots {
            if let Some(t) = slot.thread.get() {
                t.unpark();
            }
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker_loop(shared: &Shared, idx: usize) {
    // Load the current epoch before registering this worker. If we registered
    // first, Pool::new would return, the dispatcher could bump the epoch and
    // dispatch the first job, and when we finally load epoch here, we'd see
    // the bumped value and mistakenly treat the first dispatch as "nothing new",
    // never decrementing remaining and causing the dispatcher to hang forever.
    let mut seen = shared.epoch.load(Ordering::SeqCst);
    shared.slots[idx]
        .thread
        .set(std::thread::current())
        .expect("worker slot set once");
    loop {
        // Wait for a new epoch: bounded spin, then park.
        let mut spins = 0u32;
        let epoch = loop {
            if shared.shutdown.load(Ordering::SeqCst) {
                return;
            }
            let e = shared.epoch.load(Ordering::SeqCst);
            if e != seen {
                break e;
            }
            if spins < SPIN_ITERS {
                spins += 1;
                std::hint::spin_loop();
            } else {
                let slot = &shared.slots[idx];
                slot.parked.store(true, Ordering::SeqCst);
                if shared.epoch.load(Ordering::SeqCst) == seen
                    && !shared.shutdown.load(Ordering::SeqCst)
                {
                    std::thread::park();
                }
                slot.parked.store(false, Ordering::SeqCst);
                spins = 0;
            }
        };
        seen = epoch;
        let n_shards = epoch & PACKED_SHARD_MASK;
        if idx + 1 >= n_shards {
            continue; // no shard this dispatch: never touch `job`.
        }
        // SAFETY: this worker holds shard `idx + 1` of the current epoch;
        // the dispatcher does not touch `job` until `remaining == 0`, and
        // this worker has not yet decremented. `kind` is a `Copy` enum of
        // `Copy` fields, read out exactly like the old bare `GemvFn`.
        let (kind, y, xq, w, k, start, end) = unsafe {
            let job = &*shared.job.get();
            let (start, end) = job.shards[idx + 1];
            (
                job.kind.expect("published job has a kind"),
                job.y,
                job.xq,
                job.w,
                job.k,
                start,
                end,
            )
        };
        // SAFETY: dispatcher's caller contract covers this disjoint range.
        unsafe { run_shard(&kind, y, xq, w, k, start, end) };
        shared.remaining.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test kernel: y[r] = (r * 31 + k) as f32, plus a marker read from xq
    /// so the pointer plumbing is exercised. Panic-free, extern "C", same
    /// shape as the real kernels.
    unsafe extern "C" fn stamp_rows(
        y: *mut f32,
        xq: *const u8,
        _w: *const u8,
        k: usize,
        row_start: usize,
        row_end: usize,
    ) {
        // SAFETY: tests pass buffers sized to `rows`/1 byte respectively.
        let bias = unsafe { *xq } as usize;
        for r in row_start..row_end {
            // SAFETY: r < rows and y has rows elements.
            unsafe { *y.add(r) = (r * 31 + k + bias) as f32 };
        }
    }

    fn dispatch(pool: &Pool, rows: usize, k: usize) -> Vec<f32> {
        let mut y = vec![f32::NAN; rows];
        let xq = [7u8];
        let w = [0u8];
        // SAFETY: buffers sized per stamp_rows' expectations, live for the call.
        unsafe { pool.par_gemv(stamp_rows, y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, rows) };
        y
    }

    fn expected(rows: usize, k: usize) -> Vec<f32> {
        (0..rows).map(|r| (r * 31 + k + 7) as f32).collect()
    }

    #[test]
    fn parallel_matches_serial_expectation() {
        let pool = Pool::new(4);
        for rows in [1, 7, 8, 9, 64, 1000, 1024] {
            assert_eq!(dispatch(&pool, rows, 3), expected(rows, 3), "rows={rows}");
        }
    }

    #[test]
    fn zero_rows_is_a_noop() {
        let pool = Pool::new(4);
        assert!(dispatch(&pool, 0, 3).is_empty());
    }

    #[test]
    fn decode_cap_defaults_to_capacity() {
        let pool = Pool::new(8);
        assert_eq!(pool.decode_threads(), 8);
    }

    #[test]
    fn set_decode_threads_clamps_to_1_capacity() {
        let pool = Pool::new(8);
        pool.set_decode_threads(0);
        assert_eq!(pool.decode_threads(), 1);
        pool.set_decode_threads(999);
        assert_eq!(pool.decode_threads(), 8);
        pool.set_decode_threads(4);
        assert_eq!(pool.decode_threads(), 4);
    }

    #[test]
    fn decode_cap_bounds_shard_count_but_not_result() {
        // Cap below active must still produce the exact serial expectation:
        // capping only regroups rows into fewer shards.
        let pool = Pool::new(8);
        pool.set_decode_threads(2);
        for rows in [1, 7, 8, 9, 64, 1000, 1024] {
            assert_eq!(dispatch(&pool, rows, 3), expected(rows, 3), "rows={rows}");
        }
    }

    #[test]
    fn capacity_one_runs_inline() {
        let pool = Pool::new(1);
        assert_eq!(pool.capacity(), 1);
        assert_eq!(dispatch(&pool, 100, 5), expected(100, 5));
    }

    #[test]
    fn active_threads_clamps_and_still_computes_identically() {
        let pool = Pool::new(8);
        pool.set_active_threads(0);
        assert_eq!(pool.active_threads(), 1);
        pool.set_active_threads(64);
        assert_eq!(pool.active_threads(), 8);
        pool.set_active_threads(3);
        assert_eq!(dispatch(&pool, 1000, 2), expected(1000, 2));
    }

    #[test]
    fn workers_park_and_wake_correctly() {
        let pool = Pool::new(4);
        assert_eq!(dispatch(&pool, 512, 1), expected(512, 1));
        // Well past the spin window: workers are parked now.
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(dispatch(&pool, 512, 1), expected(512, 1));
    }

    /// Stress: thousands of back-to-back dispatches with varying row counts
    /// (hitting the shardless-worker path when rows < 8 * capacity) — the
    /// epoch/remaining protocol must never lose or duplicate a shard.
    #[test]
    fn stress_repeated_dispatches() {
        let pool = Pool::new(4);
        for i in 0..5_000usize {
            let rows = 1 + (i * 37) % 1024;
            let y = dispatch(&pool, rows, i % 13);
            assert_eq!(y, expected(rows, i % 13), "iteration {i}, rows {rows}");
        }
    }

    #[test]
    fn drop_joins_workers() {
        let pool = Pool::new(4);
        assert_eq!(dispatch(&pool, 64, 0), expected(64, 0));
        drop(pool); // must not hang or panic
    }
}
