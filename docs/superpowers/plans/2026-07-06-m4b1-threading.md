# M4b.1 — Multi-Threaded Generated Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Multi-thread the compiled path's GEMVs via a new `inferno-pool` crate (persistent fork-join pool + `inferno_par_gemv` C-ABI dispatcher), with output bit-identical across thread counts, and record a new bench-protocol data point.

**Architecture:** Codegen stops calling GEMV kernels directly and instead calls one host-exported dispatcher, `inferno_par_gemv`, passing the kernel it already selected as a function pointer. The dispatcher splits `0..rows` into static, 8-row-aligned contiguous shards across a std-only spin-then-park thread pool; each shard is a plain call to the unchanged single-threaded kernel, so each output row is computed entirely by one thread and thread count never changes output bits. Everything else (attention, norms, elementwise, prefill batching) stays serial per the spec.

**Tech Stack:** Rust (edition 2024, workspace), std-only threading (no rayon), inkwell/LLVM 18 for the one codegen change, cargo-nextest + insta + proptest for tests.

**Spec:** `docs/superpowers/specs/2026-07-06-m4b1-threading-design.md` — read it before starting.

## Global Constraints

- **No new external dependencies.** The pool is std-only; `thiserror` (already a workspace dep) is the only crate `inferno-pool` may use.
- **Kernels are untouched.** No file under `crates/inferno-kernels/src/` changes in this plan.
- **Bit-identity is a hard contract:** thread count must never change output bits. Tests assert **exact** equality (`to_bits()`), never tolerance.
- **Shard rules (exact values from the spec):** boundaries aligned to 8 rows (`SHARD_ALIGN = 8`, must equal `inferno_kernels::STRIP`); `shards = min(threads, ceil(rows/8))`; `rows == 0` returns immediately; shard map is a pure function of `(rows, threads)`.
- **Spin window:** bounded spin ≈50µs before parking, a named constant (`SPIN_ITERS`).
- **Thread count:** clamped to `1..=logical_cores`; default = physical cores from `inferno-target` topology (never re-probed).
- **Pool init:** process-global, once; re-init with a *different* count is a loud Rust-level error; an **uninitialized** pool means `inferno_par_gemv` falls back to a direct serial kernel call.
- **No CI perf gates** (AGENTS.md). Scaling is measured only by the manual protocol on quiet hardware; perf numbers come only from real runs — never fabricate or estimate one.
- **The nightly `bench-compiled` speedup gate stays pinned at `--threads 1`.**
- Workspace lints deny `unsafe_code`; `inferno-pool` opts out with its own `[lints.rust]` table (the `inferno-kernels` pattern) and denies `unsafe_op_in_unsafe_fn`. Every `unsafe` block needs a `// SAFETY:` comment (semgrep/clippy will flag otherwise).
- Run `mise run lint && mise run test` before every commit; both must be clean.

---

### Task 1: `inferno-pool` crate scaffold + shard math

**Files:**
- Modify: `/workspace/Cargo.toml` (workspace members + workspace.dependencies)
- Create: `/workspace/crates/inferno-pool/Cargo.toml`
- Create: `/workspace/crates/inferno-pool/src/lib.rs`
- Create: `/workspace/crates/inferno-pool/src/shard.rs`

**Interfaces:**
- Consumes: nothing (pure new code).
- Produces: `inferno_pool::shard::{shard_table(rows: usize, threads: usize) -> Vec<(usize, usize)>, SHARD_ALIGN: usize}` re-exported at crate root; the crate `inferno-pool` in the workspace.

- [ ] **Step 1: Register the crate in the workspace**

In `/workspace/Cargo.toml`, add `"crates/inferno-pool"` to `members` (keep the list on one line, matching current style):

```toml
members = ["crates/inferno-formats", "crates/inferno-graph", "crates/inferno-target", "crates/inferno-runtime", "crates/inferno-kernels", "crates/inferno-plan", "crates/inferno-codegen", "crates/inferno-core", "crates/inferno-pool", "cli"]
```

and in `[workspace.dependencies]`, after the `inferno-core` line:

```toml
inferno-pool = { path = "crates/inferno-pool" }
```

- [ ] **Step 2: Create the crate manifest**

`/workspace/crates/inferno-pool/Cargo.toml`:

```toml
[package]
name = "inferno-pool"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
thiserror.workspace = true

[dev-dependencies]
inferno-kernels.workspace = true
inferno-formats.workspace = true

# The workspace lints deny `unsafe_code`; this crate is the third sanctioned
# exception (after inferno-kernels and inferno-core): it ships raw pointers
# across threads inside the fork-join dispatcher and exposes the
# `inferno_par_gemv` C-ABI symbol generated code calls. Like the other two,
# it opts out with its OWN lint table and denies `unsafe_op_in_unsafe_fn`.
[lints.rust]
unsafe_op_in_unsafe_fn = "deny"
```

(The dev-dependencies are used by Task 4's bit-identity rig; declaring them now avoids touching the manifest twice.)

- [ ] **Step 3: Write the failing shard tests**

`/workspace/crates/inferno-pool/src/shard.rs`:

```rust
//! Static shard partitioning: pure math, no threads. The shard map is a
//! deterministic function of `(rows, threads)` — boundaries align to the
//! kernels' 8-row strip so AVX2 strips are never split across threads, and
//! only the final shard may end off-alignment (at `rows` itself).

/// Shard boundary alignment in rows. Must equal `inferno_kernels::STRIP`
/// (asserted by a test in `tests/par_rig.rs`); duplicated here so the pool
/// has no runtime dependency on the kernels crate.
pub const SHARD_ALIGN: usize = 8;

/// Split `0..rows` into at most `threads` contiguous shards whose internal
/// boundaries are multiples of [`SHARD_ALIGN`]. Strips are distributed as
/// evenly as possible (earlier shards get the remainder strips); the final
/// (possibly partial) strip lands in the last shard. `rows == 0` yields no
/// shards; `threads == 0` is treated as 1.
pub fn shard_table(rows: usize, threads: usize) -> Vec<(usize, usize)> {
    if rows == 0 {
        return Vec::new();
    }
    let strips = rows.div_ceil(SHARD_ALIGN);
    let n = threads.max(1).min(strips);
    let base = strips / n;
    let extra = strips % n;
    let mut out = Vec::with_capacity(n);
    let mut strip = 0usize;
    for i in 0..n {
        let take = base + usize::from(i < extra);
        let start = strip * SHARD_ALIGN;
        strip += take;
        out.push((start, (strip * SHARD_ALIGN).min(rows)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rows_yields_no_shards() {
        assert!(shard_table(0, 8).is_empty());
    }

    #[test]
    fn exact_split() {
        assert_eq!(shard_table(16, 2), vec![(0, 8), (8, 16)]);
    }

    #[test]
    fn partial_final_strip_goes_to_last_shard() {
        // 20 rows = 3 strips (8, 8, 4); 4 threads clamp to 3 shards.
        assert_eq!(shard_table(20, 4), vec![(0, 8), (8, 16), (16, 20)]);
    }

    #[test]
    fn fewer_strips_than_threads_collapses() {
        assert_eq!(shard_table(7, 12), vec![(0, 7)]);
    }

    #[test]
    fn threads_zero_behaves_as_one() {
        assert_eq!(shard_table(100, 0), vec![(0, 100)]);
    }

    /// Exhaustive structural properties over a grid: shards tile `0..rows`
    /// contiguously, every internal boundary is 8-aligned, shard count is
    /// `min(threads, ceil(rows/8))`, and the map is deterministic.
    #[test]
    fn structural_properties_hold_on_grid() {
        for rows in (0..2048usize).step_by(7) {
            for threads in 1..=16usize {
                let s = shard_table(rows, threads);
                assert_eq!(s, shard_table(rows, threads), "determinism");
                if rows == 0 {
                    assert!(s.is_empty());
                    continue;
                }
                assert_eq!(s.len(), threads.min(rows.div_ceil(SHARD_ALIGN)));
                assert_eq!(s[0].0, 0);
                assert_eq!(s.last().unwrap().1, rows);
                for w in s.windows(2) {
                    assert_eq!(w[0].1, w[1].0, "contiguous");
                    assert_eq!(w[0].1 % SHARD_ALIGN, 0, "aligned boundary");
                }
                for &(a, b) in &s {
                    assert!(a < b, "non-empty shard");
                }
            }
        }
    }
}
```

`/workspace/crates/inferno-pool/src/lib.rs` (minimal for now; grows in Tasks 2–3):

```rust
//! Persistent fork-join thread pool + the `inferno_par_gemv` dispatcher
//! that M4b.1 generated code calls by symbol. Kernels stay single-threaded
//! (spec boundary rule: parallelism is the caller's job — this crate IS
//! that caller): the dispatcher splits a GEMV's row range into 8-row-aligned
//! shards, so each output row is computed entirely by one thread with the
//! kernel's fixed combine order and **thread count never changes output
//! bits**.

pub mod shard;

pub use shard::{SHARD_ALIGN, shard_table};
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `cargo nextest run -p inferno-pool`
Expected: all 6 tests PASS. (The tests and implementation land together here because `shard_table` is a single pure function — the grid test is the real gate.)

- [ ] **Step 5: Lint and commit**

Run: `mise run lint && mise run test`
Expected: clean, all PASS.

```bash
git add Cargo.toml Cargo.lock crates/inferno-pool
git commit -m "feat(pool): inferno-pool crate — 8-row-aligned static shard math"
```

---

### Task 2: the fork-join `Pool`

**Files:**
- Create: `/workspace/crates/inferno-pool/src/pool.rs`
- Modify: `/workspace/crates/inferno-pool/src/lib.rs`

**Interfaces:**
- Consumes: `shard_table`, `SHARD_ALIGN` (Task 1).
- Produces:
  - `inferno_pool::GemvFn = unsafe extern "C" fn(*mut f32, *const u8, *const u8, usize, usize, usize)` (y, xq, w, k, row_start, row_end — exactly the M2 kernel ABI).
  - `inferno_pool::Pool` with `Pool::new(threads: usize) -> Pool`, `capacity(&self) -> usize`, `set_active_threads(&self, n: usize)`, `active_threads(&self) -> usize`, and `unsafe fn par_gemv(&self, kernel: GemvFn, y: *mut f32, xq: *const u8, w: *const u8, k: usize, rows: usize)`.

- [ ] **Step 1: Write `pool.rs`**

The concurrency protocol, exactly as specified here (the packed epoch is load-bearing — see the comment on `PACKED_SHARD_BITS`):

```rust
//! The persistent fork-join pool. One job at a time: the dispatching thread
//! publishes a fully partitioned GEMV, bumps an epoch, runs shard 0 itself,
//! and spins until workers drain the remaining shards. Workers spin briefly
//! (decode-loop dispatch cadence is ~100µs) then park, so idle processes go
//! quiet. No queues, no work-stealing: the shard→thread map is a pure
//! function of `(rows, active_threads)`, deterministic run-to-run.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{JoinHandle, Thread};

use crate::shard::shard_table;

/// The M2 GEMV kernel ABI: `(y, xq, w, k, row_start, row_end)`. Must match
/// `inferno-kernels`' `#[unsafe(no_mangle)]` symbols exactly (the rig in
/// `tests/par_rig.rs` coerces the real symbols to this type, so a drift is
/// a compile error).
pub type GemvFn =
    unsafe extern "C" fn(*mut f32, *const u8, *const u8, usize, usize, usize);

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
    kernel: Option<GemvFn>,
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    shards: Vec<(usize, usize)>,
}

impl Job {
    fn empty() -> Job {
        Job {
            kernel: None,
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
            while slot.thread.get().is_none() {
                std::hint::spin_loop();
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

    /// Run `kernel` over `0..rows`, split across up to `active_threads()`
    /// lanes. Returns after every shard completes. Output is bit-identical
    /// for every thread count: each row is computed entirely by one lane.
    ///
    /// # Safety
    /// `kernel`'s documented contract must hold for `(y, xq, w, k)` over
    /// every row in `0..rows` (`y` valid for `rows` f32 writes; `xq`/`w`
    /// valid packed buffers for this `k`/`rows`; 32-byte alignment where
    /// the kernel requires it), and all buffers must stay live and
    /// otherwise-untouched until this call returns.
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
        let active = self.active_threads();
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
                kernel: Some(kernel),
                y,
                xq,
                w,
                k,
                shards,
            };
        }
        self.shared.remaining.store(n_worker, Ordering::SeqCst);
        let counter = (self.shared.epoch.load(Ordering::SeqCst) >> PACKED_SHARD_BITS)
            .wrapping_add(1);
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
        unsafe { kernel(y, xq, w, k, s0, e0) };
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
    shared.slots[idx]
        .thread
        .set(std::thread::current())
        .expect("worker slot set once");
    let mut seen = shared.epoch.load(Ordering::SeqCst);
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
        // this worker has not yet decremented.
        let (kernel, y, xq, w, k, start, end) = unsafe {
            let job = &*shared.job.get();
            let (start, end) = job.shards[idx + 1];
            (
                job.kernel.expect("published job has a kernel"),
                job.y,
                job.xq,
                job.w,
                job.k,
                start,
                end,
            )
        };
        // SAFETY: dispatcher's caller contract covers this disjoint range.
        unsafe { kernel(y, xq, w, k, start, end) };
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
        unsafe {
            pool.par_gemv(stamp_rows, y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, rows)
        };
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
```

- [ ] **Step 2: Export from lib.rs**

In `/workspace/crates/inferno-pool/src/lib.rs`, add:

```rust
pub mod pool;

pub use pool::{GemvFn, Pool};
```

- [ ] **Step 3: Run the tests**

Run: `cargo nextest run -p inferno-pool`
Expected: all Task 1 + Task 2 tests PASS. The stress test should finish in well under a minute; if it hangs, the epoch/park protocol is wrong — debug it, do not weaken the test.

- [ ] **Step 4: Lint and commit**

Run: `mise run lint && mise run test`
Expected: clean, all PASS.

```bash
git add crates/inferno-pool
git commit -m "feat(pool): spin-then-park fork-join pool with packed-epoch dispatch"
```

---

### Task 3: global pool + `inferno_par_gemv` C-ABI entry

**Files:**
- Create: `/workspace/crates/inferno-pool/src/error.rs`
- Modify: `/workspace/crates/inferno-pool/src/lib.rs`
- Create: `/workspace/crates/inferno-pool/tests/global.rs`
- Create: `/workspace/crates/inferno-pool/tests/fallback.rs`

**Interfaces:**
- Consumes: `Pool`, `GemvFn` (Task 2).
- Produces (later tasks call all of these):
  - `inferno_pool::init_global(threads: usize) -> Result<(), PoolError>` — idempotent for the same count; `PoolError::AlreadyInitialized { current, requested }` for a different count.
  - `inferno_pool::set_global_active_threads(n: usize) -> bool` — `false` if no global pool exists.
  - `inferno_pool::PoolError` (thiserror, `Debug + PartialEq`).
  - `#[unsafe(no_mangle)] pub unsafe extern "C" fn inferno_par_gemv(kernel: GemvFn, y: *mut f32, xq: *const u8, w: *const u8, k: usize, rows: usize)` — dispatches on the global pool; **serial direct kernel call when the global pool is uninitialized**.

- [ ] **Step 1: Write `error.rs`**

```rust
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PoolError {
    /// The process-global pool is sized once; a mismatched re-init is a
    /// caller bug (spec: error loudly, never silently reconfigure). Use
    /// `set_global_active_threads` to vary per-run parallelism instead.
    #[error(
        "thread pool already initialized with {current} threads (requested {requested}); \
         use set_global_active_threads to change per-dispatch parallelism"
    )]
    AlreadyInitialized { current: usize, requested: usize },
}
```

- [ ] **Step 2: Add the global + extern entry to `lib.rs`**

Append to `/workspace/crates/inferno-pool/src/lib.rs` (keeping the Task 1–2 content):

```rust
pub mod error;

pub use error::PoolError;

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
```

- [ ] **Step 3: Write the two global-state integration tests**

These are separate integration-test **files** on purpose: each `tests/*.rs`
file is its own binary/process, so the `fallback` test really sees an
uninitialized global even under plain `cargo test`.

`/workspace/crates/inferno-pool/tests/fallback.rs`:

```rust
//! `inferno_par_gemv` with NO global pool: must run serially and correctly.
//! Own test binary so nothing else can have initialized the global first.

use inferno_pool::GemvFn;

unsafe extern "C" fn stamp(
    y: *mut f32,
    _xq: *const u8,
    _w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    for r in row_start..row_end {
        // SAFETY: test sizes y to `rows`.
        unsafe { *y.add(r) = (r + k) as f32 };
    }
}

#[test]
fn uninitialized_global_falls_back_to_serial() {
    let mut y = vec![f32::NAN; 100];
    let (xq, w) = ([0u8], [0u8]);
    let kernel: GemvFn = stamp;
    // SAFETY: buffers sized per stamp's expectations.
    unsafe {
        inferno_pool::inferno_par_gemv(kernel, y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), 5, 100)
    };
    assert_eq!(y, (0..100).map(|r| (r + 5) as f32).collect::<Vec<_>>());
    // rows == 0 is a no-op even uninitialized.
    // SAFETY: rows == 0 → no writes.
    unsafe {
        inferno_pool::inferno_par_gemv(kernel, y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), 5, 0)
    };
}
```

`/workspace/crates/inferno-pool/tests/global.rs`:

```rust
//! Global init semantics + the extern entry over an initialized pool.
//! ONE #[test] fn: these steps share the process-global OnceLock, so their
//! order must be fixed regardless of test-runner parallelism.

use inferno_pool::{GemvFn, PoolError};

unsafe extern "C" fn stamp(
    y: *mut f32,
    _xq: *const u8,
    _w: *const u8,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    for r in row_start..row_end {
        // SAFETY: test sizes y to `rows`.
        unsafe { *y.add(r) = (r * 3 + k) as f32 };
    }
}

#[test]
fn init_dispatch_and_mismatch_semantics() {
    assert!(inferno_pool::init_global(4).is_ok());
    assert!(inferno_pool::init_global(4).is_ok(), "same count: idempotent");
    assert_eq!(
        inferno_pool::init_global(2),
        Err(PoolError::AlreadyInitialized { current: 4, requested: 2 })
    );

    let run = || {
        let mut y = vec![f32::NAN; 1000];
        let (xq, w) = ([0u8], [0u8]);
        let kernel: GemvFn = stamp;
        // SAFETY: buffers sized per stamp's expectations.
        unsafe {
            inferno_pool::inferno_par_gemv(
                kernel, y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), 9, 1000,
            )
        };
        y
    };
    let want: Vec<f32> = (0..1000).map(|r| (r * 3 + 9) as f32).collect();
    assert_eq!(run(), want, "threaded");

    assert!(inferno_pool::set_global_active_threads(1));
    assert_eq!(run(), want, "t=1 via active-threads cap");
    assert!(inferno_pool::set_global_active_threads(4));
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo nextest run -p inferno-pool`
Expected: all PASS, including the two new integration binaries.

- [ ] **Step 5: Lint and commit**

Run: `mise run lint && mise run test`
Expected: clean, all PASS.

```bash
git add crates/inferno-pool
git commit -m "feat(pool): global pool init + inferno_par_gemv C-ABI dispatcher"
```

---

### Task 4: bit-identity rig against the real kernels

**Files:**
- Create: `/workspace/crates/inferno-pool/tests/par_rig.rs`

**Interfaces:**
- Consumes: `Pool`, `GemvFn`, `SHARD_ALIGN` (Tasks 1–3); `inferno_kernels` pack/quantize/kernel symbols (dev-dependency declared in Task 1).
- Produces: nothing new — this is the contract gate: **thread count never changes output bits**, per dtype.

- [ ] **Step 1: Write the rig**

`/workspace/crates/inferno-pool/tests/par_rig.rs`:

```rust
//! Thread-count bit-identity: the same GEMV dispatched at t=1 / t=4 / t=12
//! must produce EXACTLY the same bits as one direct single-threaded kernel
//! call, per dtype. Row-partitioned shards never reassociate any f32 op, so
//! this extends the kernels' "ISA variants are bit-identical" contract to
//! thread count. Exact equality (`to_bits`), never tolerance.

use inferno_formats::{DType, quant};
use inferno_kernels::{AlignedBuf, reference_kernels};
use inferno_pool::{GemvFn, Pool};

/// Deterministic pseudo-random f32s in [-1, 1) (same generator as the
/// kernels' rig — xorshift, no deps).
fn pseudo(mut seed: u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Pack weights + quantize the activation for `dtype` via the scalar
/// KernelSet, returning (packed weights, activation bytes).
fn prep(dtype: &DType, rows: usize, k: usize, seed: u64) -> (AlignedBuf, Vec<u8>) {
    let set = reference_kernels(dtype).expect("scalar set always available");
    let wvals = pseudo(seed, rows * k);
    let wbytes = quant::pack(dtype, &wvals).unwrap();
    let w = set.pack(&wbytes, rows, k).unwrap();
    let x = pseudo(seed ^ 0x9e3779b97f4a7c15, k);
    let xq = set.quantize_row(&x).unwrap();
    (w, xq)
}

fn serial(kernel: GemvFn, w: &AlignedBuf, xq: &[u8], rows: usize, k: usize) -> Vec<f32> {
    let mut y = vec![f32::NAN; rows];
    // SAFETY: w/xq built by prep() for exactly (rows, k); y has rows f32s.
    unsafe { kernel(y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, 0, rows) };
    y
}

fn pooled(
    pool: &Pool,
    kernel: GemvFn,
    w: &AlignedBuf,
    xq: &[u8],
    rows: usize,
    k: usize,
) -> Vec<f32> {
    let mut y = vec![f32::NAN; rows];
    // SAFETY: same contract as `serial`; the pool only splits the range.
    unsafe { pool.par_gemv(kernel, y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), k, rows) };
    y
}

fn assert_bit_identical(dtype: &DType, kernel: GemvFn, rows: usize, k: usize) {
    let (w, xq) = prep(dtype, rows, k, 0xfeed_beef);
    let want = serial(kernel, &w, &xq, rows, k);
    for threads in [1usize, 4, 12] {
        let pool = Pool::new(threads);
        let got = pooled(&pool, kernel, &w, &xq, rows, k);
        for (i, (g, s)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                s.to_bits(),
                "{dtype:?} t={threads} row {i}: {g} != {s}"
            );
        }
    }
}

#[test]
fn shard_align_matches_kernel_strip() {
    assert_eq!(inferno_pool::SHARD_ALIGN, inferno_kernels::STRIP);
}

#[test]
fn f32_thread_count_is_bit_invisible() {
    // rows deliberately not a multiple of 8; k unconstrained for f32.
    assert_bit_identical(&DType::F32, inferno_kernels::inferno_gemv_f32_rs8_scalar, 1003, 33);
}

#[test]
fn q8_0_thread_count_is_bit_invisible() {
    // k must be a multiple of 32 (Q8_0 block).
    assert_bit_identical(&DType::Q8_0, inferno_kernels::inferno_gemv_q8_0_rs8_scalar, 1003, 64);
}

#[test]
fn q4_k_thread_count_is_bit_invisible() {
    // k must be a multiple of 256 (Q4_K superblock).
    assert_bit_identical(&DType::Q4_K, inferno_kernels::inferno_gemv_q4_k_rs8_scalar, 1003, 256);
}
```

- [ ] **Step 2: Run the rig**

Run: `cargo nextest run -p inferno-pool`
Expected: all PASS. If `quant::pack` or `quantize_row` reject a dimension, fix the test's `k` to the dtype's block multiple (the values above respect Q8_0=32 and Q4_K=256) — do NOT loosen the bit-equality assertion.

- [ ] **Step 3: Lint and commit**

Run: `mise run lint && mise run test`
Expected: clean, all PASS.

```bash
git add crates/inferno-pool/tests/par_rig.rs
git commit -m "test(pool): thread-count bit-identity rig vs real kernels (f32/q8_0/q4_k)"
```

---

### Task 5: codegen calls `inferno_par_gemv`; host retains it; cache salt

**Files:**
- Modify: `/workspace/crates/inferno-codegen/src/lib.rs`
- Modify: `/workspace/crates/inferno-codegen/src/llvm/mod.rs` (declare_kernels + its test)
- Modify: `/workspace/crates/inferno-codegen/src/llvm/ops.rs` (lower_gemv)
- Modify: `/workspace/crates/inferno-core/Cargo.toml` (add inferno-pool dep)
- Modify: `/workspace/crates/inferno-core/src/artifact.rs` (ensure_kernels_linked)
- Modify: `/workspace/crates/inferno-core/src/cache.rs` (cache-key salt)

**Interfaces:**
- Consumes: `inferno_pool::inferno_par_gemv` (Task 3).
- Produces: generated `model.so` objects whose GEMV steps call `inferno_par_gemv(kernel_ptr, y, xq, w, k, rows)`; `inferno_codegen::HOST_ABI_VERSION: &str` folded into `cache_key` so pre-M4b.1 cached artifacts are never reused by a host that expects the new call shape (and vice versa).

- [ ] **Step 1: Add the ABI version constant to `inferno-codegen`**

In `/workspace/crates/inferno-codegen/src/lib.rs`, after the existing `pub use` lines:

```rust
/// Version of the host-symbol surface generated code links against
/// (kernel symbols + `inferno_par_gemv`). Folded into `inferno-core`'s
/// artifact cache key: bump it whenever the emitted code's host-call shape
/// changes, so stale cached `model.so`s are recompiled instead of silently
/// running with the old call pattern. "2" = M4b.1's `inferno_par_gemv`
/// dispatch (v1 was M3's direct kernel calls).
pub const HOST_ABI_VERSION: &str = "2";
```

- [ ] **Step 2: Declare `inferno_par_gemv` in `declare_kernels`**

In `/workspace/crates/inferno-codegen/src/llvm/mod.rs`, inside `declare_kernels`, after the quantize loop:

```rust
        // void inferno_par_gemv(ptr kernel, ptr y, ptr xq, ptr w, i64 k, i64 rows)
        // — the M4b.1 host dispatcher; the kernel chosen by `gemv_symbol` is
        // passed as a function pointer, so the per-(dtype, isa) selection
        // logic is unchanged.
        let par_gemv_ty = void.fn_type(
            &[
                ptr.into(),
                ptr.into(),
                ptr.into(),
                ptr.into(),
                i64_t.into(),
                i64_t.into(),
            ],
            false,
        );
        self.module
            .add_function("inferno_par_gemv", par_gemv_ty, Some(Linkage::External));
```

Extend the existing declare test (`llvm/mod.rs` ~line 196, the one asserting `ir.contains("inferno_gemv_")`) with one more assertion:

```rust
        assert!(ir.contains("inferno_par_gemv"));
```

- [ ] **Step 3: Swap the call in `lower_gemv`**

In `/workspace/crates/inferno-codegen/src/llvm/ops.rs`, replace the tail of
`lower_gemv` (everything from `let gfn = ...` through the `build_call(...)`
for `"gemv"`) with:

```rust
        let w_ptr = self.byte_ptr(frame.weights, self.const_i64(pw.offset as u64));
        let out_ptr = self.arena_row_ptr(frame, out);
        let gfn = self
            .module
            .get_function(symbol)
            .expect("gemv kernel declared (Task 8)");
        let pfn = self
            .module
            .get_function("inferno_par_gemv")
            .expect("par gemv dispatcher declared");
        let rows_c = self.const_i64(rows as u64);
        self.builder
            .build_call(
                pfn,
                &[
                    gfn.as_global_value().as_pointer_value().into(),
                    out_ptr.into(),
                    xq_ptr.into(),
                    w_ptr.into(),
                    k_c.into(),
                    rows_c.into(),
                ],
                "par_gemv",
            )
            .unwrap();
```

(The `let w_ptr`/`let out_ptr` lines are unchanged — shown for anchoring.
`zero` disappears; the dispatcher owns the range split now.) Update the
doc comment on `lower_gemv`: the sentence «row range `[0, rows)`
(single-threaded), mirroring the decode kernel» becomes «dispatched through
`inferno_par_gemv`, which shards `[0, rows)` across the host thread pool
(serial when the pool is uninitialized)».

- [ ] **Step 4: Retain the symbol in the host + salt the cache key**

`/workspace/crates/inferno-core/Cargo.toml` — add to `[dependencies]`:

```toml
inferno-pool.workspace = true
```

`/workspace/crates/inferno-core/src/artifact.rs` — in `ensure_kernels_linked`, after the last quantize line, add:

```rust
    p(inferno_pool::inferno_par_gemv as *const ());
```

and extend that function's doc comment: it retains the kernel symbols *and
the `inferno_par_gemv` dispatcher* the compiled `model.so` resolves against.

`/workspace/crates/inferno-core/src/cache.rs` — in `cache_key`, after the
`CARGO_PKG_VERSION` update line, add:

```rust
    h.update(inferno_codegen::HOST_ABI_VERSION.as_bytes());
```

and extend the function's doc comment to list "the codegen host-ABI version"
among the hashed inputs (module doc at the top of the file too).

- [ ] **Step 5: Run the full suite**

Run: `mise run lint && mise run test`
Expected: clean, all PASS. Two things this proves end-to-end:
- `crates/inferno-core/tests/backend.rs` dlopens a freshly compiled artifact and generates — the artifact now resolves `inferno_par_gemv` from the host, and since nothing initialized the global pool in that test process, it exercises the **serial fallback** path.
- `cli/tests/run.rs::compiled_and_interp_agree_on_greedy_tokens` still agrees — the compiled path's numerics are unchanged.

If a dlopen "undefined symbol: inferno_par_gemv" error appears, the retention step (or `-rdynamic` propagation) is broken — fix that; do not work around it by weakening the test.

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-codegen crates/inferno-core Cargo.lock
git commit -m "feat(codegen): route GEMVs through inferno_par_gemv; salt artifact cache with host-ABI version"
```

---

### Task 6: `Engine` thread plumbing + threaded pool init

**Files:**
- Modify: `/workspace/crates/inferno-core/src/lib.rs` (Engine)
- Modify: `/workspace/crates/inferno-core/src/error.rs` (CoreError::Pool)
- Modify: `/workspace/crates/inferno-core/tests/backend.rs` (threaded bit-identity test)

**Interfaces:**
- Consumes: `inferno_pool::{init_global, set_global_active_threads, PoolError}` (Task 3).
- Produces (CLI tasks rely on these exact signatures):
  - `Engine::set_threads(&mut self, threads: usize)` — clamps to `1..=logical_cores`.
  - `Engine::threads(&self) -> usize` — defaults to `physical_cores.max(1)` from the engine's already-detected `TargetDesc`.
  - `Engine::compiled_backend(&self)` now initializes the global pool with `self.threads` (and sets it active) before constructing the backend; returns `CoreError::Pool` on a capacity mismatch.

- [ ] **Step 1: Add the error variant**

In `/workspace/crates/inferno-core/src/error.rs`, add to `CoreError`:

```rust
    #[error("pool: {0}")]
    Pool(#[from] inferno_pool::PoolError),
```

- [ ] **Step 2: Extend `Engine`**

In `/workspace/crates/inferno-core/src/lib.rs`:

```rust
pub struct Engine {
    model: PathBuf,
    target: TargetDesc,
    max_seq_len: usize,
    threads: usize,
}
```

In `Engine::load`, initialize the new field from the detected topology:

```rust
        let threads = target.topology.physical_cores.max(1) as usize;
        Ok(Engine {
            model: model.to_path_buf(),
            target,
            max_seq_len,
            threads,
        })
```

Add the accessor pair (after `load`):

```rust
    /// Compiled-path thread count for backends built by this engine.
    /// Defaults to the target's physical cores; clamped to
    /// `1..=logical_cores` (the pool's spec bounds).
    pub fn set_threads(&mut self, threads: usize) {
        let max = self.target.topology.logical_cores.max(1) as usize;
        self.threads = threads.clamp(1, max);
    }

    pub fn threads(&self) -> usize {
        self.threads
    }
```

In `compiled_backend`, before `Artifact::load_or_compile`:

```rust
        // Size the process-global pool once (loud error on a mismatched
        // re-init — spec), then cap active parallelism to this engine's
        // count so bench's t=1 diagnostics can vary it per run.
        inferno_pool::init_global(self.threads)?;
        inferno_pool::set_global_active_threads(self.threads);
```

Add `inferno-pool` is already a dependency (Task 5). Update the doc comment
on `compiled_backend` to mention pool initialization.

- [ ] **Step 3: Add the end-to-end threaded bit-identity test**

In `/workspace/crates/inferno-core/tests/backend.rs` (the file already has
`use_temp_cache()`, `const MODEL`, and a `backend(max_seq_len)` helper —
reuse the first two, but build the engine inline since this test needs
`set_threads`), add:

```rust
/// Thread count must be invisible in the logits, bit for bit: forward the
/// same tokens at active-threads=4 and =1 on the same backend (M4b.1
/// bit-identity contract, end-to-end through the dlopen'd artifact).
#[test]
fn threaded_forward_is_bit_identical_to_serial() {
    use_temp_cache();
    let mut engine = Engine::load(Path::new(MODEL), 64).unwrap();
    engine.set_threads(4);
    let mut backend = engine.compiled_backend().unwrap();
    let tokens = [1u32, 4, 7];

    assert!(inferno_pool::set_global_active_threads(4));
    let threaded = backend.forward(&tokens).unwrap();

    backend.reset();
    assert!(inferno_pool::set_global_active_threads(1));
    let serial = backend.forward(&tokens).unwrap();
    assert!(inferno_pool::set_global_active_threads(4));

    assert_eq!(threaded.len(), serial.len());
    for (i, (a, b)) in threaded.iter().zip(&serial).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "logit {i}");
    }
}
```

`inferno-pool` is already a runtime dependency of `inferno-core` (Task 5),
so integration tests can `use inferno_pool` with no manifest change.

**Caveat:** other tests in this binary may construct engines with the
default (physical-cores) thread count. Under cargo-nextest each test is its
own process, so the global `OnceLock` never collides. This new test must
therefore do its own `Engine::load` + `set_threads(4)` and not assume pool
state from other tests.

- [ ] **Step 4: Run and verify**

Run: `cargo nextest run -p inferno-core`
Expected: all PASS, including the new bit-identity test.

- [ ] **Step 5: Lint and commit**

Run: `mise run lint && mise run test`
Expected: clean, all PASS.

```bash
git add crates/inferno-core
git commit -m "feat(core): Engine thread count + global pool init; threaded forward bit-identity test"
```

---

### Task 7: CLI `--threads` on `run`; nightly speedup gate pinned t=1

**Files:**
- Modify: `/workspace/cli/src/main.rs` (Run command arg + call site)
- Modify: `/workspace/cli/src/run.rs` (`run`, `load_compiled` signatures)
- Modify: `/workspace/cli/src/bench.rs` (`bench_compiled`'s `load_compiled` call)
- Modify: `/workspace/cli/tests/run.rs` (thread-invariance test)

**Interfaces:**
- Consumes: `Engine::set_threads` (Task 6).
- Produces: `run::load_compiled(model: &Path, max_seq_len: usize, threads: u64) -> Result<Generator, Box<dyn std::error::Error>>` where `threads == 0` means "engine default (physical cores)"; `run::run(..., threads: u64, sampling: SamplerConfig)`.

- [ ] **Step 1: Thread the flag through `run.rs`**

`load_compiled` gains a `threads: u64` parameter:

```rust
pub(crate) fn load_compiled(
    model: &Path,
    max_seq_len: usize,
    threads: u64,
) -> Result<Generator, Box<dyn std::error::Error>> {
    let max_seq_len = clamp_max_seq_len(model, max_seq_len)?;
    let mut engine = Engine::load(model, max_seq_len)?;
    if threads != 0 {
        engine.set_threads(threads as usize);
    }
    let backend = engine.compiled_backend()?;
    let generator = Generator::load_with_backend(model, max_seq_len, Box::new(backend))?;
    Ok(generator)
}
```

`run` gains `threads: u64` (place it after `interp`) and forwards it:

```rust
        load_compiled(model, max_seq_len, threads).map_err(|e| e.to_string())
```

(the `--interp` branch ignores it — the interpreter is single-threaded by
design; say so in the arg's help text, not with a runtime warning).

- [ ] **Step 2: Add the clap arg**

In `/workspace/cli/src/main.rs`, in the `Run` variant after `max_seq_len`:

```rust
        /// Compiled-path thread count (0 = physical cores). The
        /// interpreter path (--interp) is single-threaded and ignores this.
        #[arg(long, default_value_t = 0)]
        threads: u64,
```

and pass it through the `Command::Run { .. } => run::run(...)` call site
(destructure `threads`, pass after `interp`).

- [ ] **Step 3: Pin the nightly speedup gate to t=1**

In `/workspace/cli/src/bench.rs`, `bench_compiled`'s compiled load becomes:

```rust
        // Pinned to 1 thread ON PURPOSE (M4b.1 spec): this gate measures
        // codegen quality against the interpreter; letting threading
        // inflate the ratio would hide codegen regressions behind
        // parallelism. Never "fix" a red nightly by unpinning this.
        let mut compiled = load_compiled(model, max_seq_len, 1)?;
```

- [ ] **Step 4: Write the CLI thread-invariance test**

In `/workspace/cli/tests/run.rs`, following the existing
`run_sampling_same_seed_is_reproducible` pattern (same `fixture` helper,
same `Command::cargo_bin` shape):

```rust
/// --threads must be invisible in the output: same fixture, same prompt,
/// t=1 vs t=4 produce byte-identical stdout (M4b.1 bit-identity contract
/// end-to-end through the real binary + dlopen'd artifact).
#[test]
fn run_threads_do_not_change_output() {
    let out = |threads: &str| {
        let a = Command::cargo_bin("inferno")
            .unwrap()
            .args([
                "run",
                &fixture("tiny.gguf"),
                "--prompt",
                "ab",
                "--max-tokens",
                "8",
                "--threads",
                threads,
            ])
            .assert()
            .success();
        a.get_output().stdout.clone()
    };
    assert_eq!(out("1"), out("4"));
}
```

(Match the argument spelling of the existing tests in that file — if they
pass the prompt positionally or via a different flag, copy their shape and
only add `--threads`.)

- [ ] **Step 5: Run and verify**

Run: `cargo nextest run -p inferno`
Expected: all PASS, including the new test.

- [ ] **Step 6: Lint and commit**

Run: `mise run lint && mise run test`
Expected: clean, all PASS.

```bash
git add cli/src/main.rs cli/src/run.rs cli/src/bench.rs cli/tests/run.rs
git commit -m "feat(cli): --threads on inferno run; speedup gate pinned to t=1"
```

---

### Task 8: `inferno bench` — matched-threads headline + inferno t=1 diagnostic

**Files:**
- Modify: `/workspace/cli/src/bench.rs` (`measure_inferno`, `BenchReport`, `render_table`, `bench`, tests)
- Modify: `/workspace/cli/src/main.rs` (Bench `--threads` help text)
- Modify: `/workspace/cli/Cargo.toml` (add `inferno-pool` dependency)
- Modify: `/workspace/cli/src/snapshots/inferno__bench__tests__bench_report_table.snap` (via insta)

**Interfaces:**
- Consumes: `inferno_pool::set_global_active_threads` (Task 3), `Engine::set_threads` (Task 6).
- Produces: `measure_inferno(model, pp, tg, reps, threads: usize, t1_diag: bool) -> Result<InfernoRun, ...>` with `pub struct InfernoRun { pub headline: InfernoNumbers, pub t1: Option<InfernoNumbers> }`; `BenchReport` gains `inferno_t1_pp_tok_s/_stddev`, `inferno_t1_tg_tok_s/_stddev: Option<f64>`.

- [ ] **Step 1: Add the cli → inferno-pool dependency**

In `/workspace/cli/Cargo.toml` `[dependencies]`, after `inferno-core`:

```toml
inferno-pool.workspace = true
```

- [ ] **Step 2: Extend `measure_inferno`**

Replace its signature/body changes as follows (the timed `run_once` closure
is unchanged):

```rust
pub struct InfernoRun {
    pub headline: InfernoNumbers,
    /// Per-thread parity diagnostic: same backend, active threads capped to
    /// 1. None when the headline itself ran at t=1.
    pub t1: Option<InfernoNumbers>,
}

/// Measure compiled prefill/decode throughput at `threads` lanes, plus an
/// optional t=1 diagnostic pass over the SAME process-global pool (capped
/// via `set_global_active_threads` — the pool is sized once per process).
pub fn measure_inferno(
    model: &Path,
    pp: usize,
    tg: usize,
    reps: usize,
    threads: usize,
    t1_diag: bool,
) -> Result<InfernoRun, Box<dyn std::error::Error>> {
```

Engine construction gains the thread count:

```rust
    let mut engine = Engine::load(model, max_seq_len)?;
    engine.set_threads(threads);
    let mut backend = engine.compiled_backend()?;
```

The existing warmup + reps loop and its `mean_stddev` reduction stay as-is,
but instead of returning, bind the result:

```rust
    let (pm, ps) = mean_stddev(&pp_samples);
    let (tm, ts) = mean_stddev(&tg_samples);
    let headline = InfernoNumbers {
        pp: Measurement { mean_tok_s: pm, stddev_tok_s: ps },
        tg: Measurement { mean_tok_s: tm, stddev_tok_s: ts },
    };
```

then add the diagnostic pass:

```rust
    let t1 = if t1_diag && engine.threads() > 1 {
        assert!(
            inferno_pool::set_global_active_threads(1),
            "pool initialized by compiled_backend above"
        );
        run_once(&mut backend)?; // warmup at the new lane count
        let mut pp_s = Vec::with_capacity(reps);
        let mut tg_s = Vec::with_capacity(reps);
        for _ in 0..reps {
            let (p, t) = run_once(&mut backend)?;
            pp_s.push(p);
            tg_s.push(t);
        }
        inferno_pool::set_global_active_threads(engine.threads());
        let (pm, ps) = mean_stddev(&pp_s);
        let (tm, ts) = mean_stddev(&tg_s);
        Some(InfernoNumbers {
            pp: Measurement { mean_tok_s: pm, stddev_tok_s: ps },
            tg: Measurement { mean_tok_s: tm, stddev_tok_s: ts },
        })
    } else {
        None
    };
    Ok(InfernoRun { headline, t1 })
```

(The inner `let (pm, ps) = ...` bindings in the diagnostic pass shadow the
headline ones — that's fine, or rename them; either way the reduction code
is `mean_stddev` twice, same as the headline.)

- [ ] **Step 3: Extend `BenchReport` + table**

Add after the existing `llama_t1_*` fields:

```rust
    /// inferno's own t=1 diagnostic (same pool, active threads capped to 1);
    /// None when the headline run was already t=1. Reads directly as the
    /// M4b.1 prefill-scaling measurement: headline pp / t1 pp.
    pub inferno_t1_pp_tok_s: Option<f64>,
    pub inferno_t1_pp_stddev: Option<f64>,
    pub inferno_t1_tg_tok_s: Option<f64>,
    pub inferno_t1_tg_stddev: Option<f64>,
```

Change the `inferno_threads` field's doc comment from «M3 generated code is
single-threaded; recorded so old data points stay interpretable after M4b
lands threading» to «Headline inferno thread count (matched to llama.cpp's
since M4b.1)».

In `render_table`, right after the `"inferno (compiled)"` row:

```rust
    if let (Some(pp), Some(pps), Some(tg), Some(tgs)) = (
        r.inferno_t1_pp_tok_s,
        r.inferno_t1_pp_stddev,
        r.inferno_t1_tg_tok_s,
        r.inferno_t1_tg_stddev,
    ) {
        row("inferno (t=1 diag)", 1, pp, pps, tg, tgs);
    }
```

- [ ] **Step 4: Wire `bench()`**

The `threads` resolution (0 → physical cores) already happens before
measurement — move the `measure_inferno` call AFTER it and pass the
resolved count:

```rust
        let inferno = measure_inferno(
            model,
            pp as usize,
            tg as usize,
            reps as usize,
            threads as usize,
            true,
        )?;
```

Report fields become:

```rust
            inferno_threads: threads,
            inferno_pp_tok_s: inferno.headline.pp.mean_tok_s,
            inferno_pp_stddev: inferno.headline.pp.stddev_tok_s,
            inferno_tg_tok_s: inferno.headline.tg.mean_tok_s,
            inferno_tg_stddev: inferno.headline.tg.stddev_tok_s,
            ...
            inferno_t1_pp_tok_s: inferno.t1.as_ref().map(|n| n.pp.mean_tok_s),
            inferno_t1_pp_stddev: inferno.t1.as_ref().map(|n| n.pp.stddev_tok_s),
            inferno_t1_tg_tok_s: inferno.t1.as_ref().map(|n| n.tg.mean_tok_s),
            inferno_t1_tg_stddev: inferno.t1.as_ref().map(|n| n.tg.stddev_tok_s),
```

In `/workspace/cli/src/main.rs`, update the Bench `--threads` help text:

```rust
        /// Thread count for BOTH engines (0 = physical cores). t=1
        /// diagnostic rows are recorded for each unless this is 1.
        #[arg(long, default_value_t = 0)]
        threads: u64,
```

- [ ] **Step 5: Update the bench unit tests + snapshot**

- `measure_inferno_smoke_on_fixture`: call becomes
  `measure_inferno(&model, 8, 4, 2, 2, true).unwrap()`; assert the headline
  numbers as before AND that `n.t1.is_some()` with finite positive means.
- `measure_inferno_rejects_prompt_beyond_context`: call becomes
  `measure_inferno(&model, 1 << 20, 4, 1, 1, false)`.
- `render_table_snapshot`: add to the struct literal
  `inferno_t1_pp_tok_s: Some(58.0), inferno_t1_pp_stddev: Some(0.9), inferno_t1_tg_tok_s: Some(21.4), inferno_t1_tg_stddev: Some(0.2),`
  then regenerate:

Run: `INSTA_UPDATE=always cargo nextest run -p inferno bench`
Then: `git diff cli/src/snapshots/` — the only change must be the new
`inferno (t=1 diag)` row between the inferno and llama.cpp rows. Revert and
fix if anything else moved.

- [ ] **Step 6: Run and verify**

Run: `cargo nextest run -p inferno`
Expected: all PASS.

- [ ] **Step 7: Lint and commit**

Run: `mise run lint && mise run test`
Expected: clean, all PASS.

```bash
git add cli/src/bench.rs cli/src/main.rs cli/Cargo.toml cli/src/snapshots Cargo.lock
git commit -m "feat(cli): bench runs inferno at matched threads + t=1 diagnostic rows"
```

---

### Task 9: docs — architecture + front door

**Files:**
- Modify: `/workspace/ARCHITECTURE.md`
- Modify: `/workspace/AGENTS.md`
- Modify: `/workspace/mise.toml` (bench task description)

**Interfaces:** none — documentation of what Tasks 1–8 built.

- [ ] **Step 1: ARCHITECTURE.md**

Add a crate bullet after `inferno-kernels`:

```markdown
- `crates/inferno-pool` — persistent fork-join thread pool + the
  `inferno_par_gemv` dispatcher generated code calls for every GEMV. This
  crate is "the caller" the kernel boundary rule refers to: it partitions
  row ranges into 8-row-aligned shards and calls the unchanged
  single-threaded kernels. Third sanctioned `unsafe` crate.
```

Replace the last boundary-rule bullet («Kernels are single-threaded and
row-range partitioned … splitting that range across a thread pool is
additive M4 work») with:

```markdown
- Kernels are single-threaded and row-range partitioned; parallelism lives
  in `inferno-pool`'s `inferno_par_gemv` dispatcher, which generated code
  calls with the full range (M4b.1). Shards are 8-row-aligned, so each
  output row is computed entirely by one thread with the kernel's fixed
  combine order — **thread count never changes output bits**, and the
  tests assert exact equality. A host that never initializes the pool runs
  serially (the dispatcher falls back to one direct kernel call).
```

- [ ] **Step 2: AGENTS.md**

In the non-obvious-constraints list, after the `inferno bench` bullet:

```markdown
- **The nightly speedup gate (`bench-compiled`) is pinned to `--threads 1`
  on purpose**: it measures codegen quality vs the interpreter, and
  threading would hide codegen regressions behind parallelism. Never "fix"
  a red nightly by unpinning it.
```

- [ ] **Step 3: mise.toml**

Update the `[tasks.bench]` description's «record data points in the M4a
spec» to «record data points in the current milestone spec's Amendments
(M4b.1: docs/superpowers/specs/2026-07-06-m4b1-threading-design.md)».

- [ ] **Step 4: Commit**

```bash
git add ARCHITECTURE.md AGENTS.md mise.toml
git commit -m "docs: inferno-pool crate + threading boundary rules; speedup-gate t=1 pin"
```

---

### Task 10: protocol run — scaling data point

**Files:**
- Modify: `docs/superpowers/specs/2026-07-06-m4b1-threading-design.md` (Amendments)

**Interfaces:** none — this executes the manual protocol.

**Hardware gate:** this task needs the quiet dev machine (AMD Ryzen 9 3900),
the devenv shell, and the pinned nightly model (Qwen2.5-0.5B-Instruct Q8_0 —
`scripts/nightly-speedup.sh` shows where it downloads from). **If you are
not on real, quiet hardware with that model available (e.g. you are a
sandboxed agent), STOP here and hand back to the human — never fabricate or
estimate a data point** (AGENTS.md: perf numbers come only from real runs).

- [ ] **Step 1 (quiet hardware only): Run the protocol**

Inside `devenv shell`, with the nightly model at `<MODEL>`:

```bash
mise run bench -- <MODEL>
mise run bench -- <MODEL> --json
```

Expected: the table now shows FOUR rows — inferno at 12 threads (headline),
inferno t=1 diag, llama.cpp at 12, llama.cpp t=1 diag — plus the ratio line
computed from the matched-threads rows.

- [ ] **Step 2 (quiet hardware only): Evaluate the exit criterion**

Prefill scaling = `inferno_pp_tok_s / inferno_t1_pp_tok_s` from the `--json`
blob. The M4b.1 target is **≥ 6x at 12 threads**.

- If it clears 6x: record and move on (Step 3).
- If it stalls below 6x: still record the data point honestly, then profile
  (e.g. `perf record` on `inferno run`) to attribute the ceiling. Serial
  attention → parallel attention becomes an explicit scoped amendment inside
  M4b.1 (spec §Risks). Memory-bandwidth saturation → record the finding;
  M4b.2 inherits it. Do NOT silently start optimizing.

- [ ] **Step 3 (quiet hardware only): Record the data point**

Append to the M4b.1 spec's `## Amendments`: date, machine (CPU model, core
counts), model file + quant, the full table output and the `--json` blob in
fenced blocks, the computed pp/tg scaling factors (headline vs inferno t=1),
and a plain statement of whether the ≥6x prefill-scaling exit criterion is
met. Never edit a previously recorded data point.

- [ ] **Step 4 (quiet hardware only): Commit**

```bash
git add docs/superpowers/specs/2026-07-06-m4b1-threading-design.md
git commit -m "docs(spec): M4b.1 threading data point (scaling protocol run)"
```
