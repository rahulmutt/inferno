# M4b.12 — Decode Attention Headroom Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the dispatch-split instrument that explains where M4b.11's head-sharding headroom went (publish / wake / kernel+alloc / drain), measure on quiet hardware, and land only the levers the pre-registered gates authorize (A: scratch reuse, D: publish slimming, W: decode wait-strategy).

**Architecture:** This is a **data-gated** milestone (M4b.11 pattern). Tasks 1–2 build the `pool-profile` instrument inside `inferno-pool` (feature-gated; the shipping hot loop is textually untouched with the feature off). Task 3 plumbs it into `inferno run --profile`. Task 4 adds the probe-only `INFERNO_ATTN_SHARDS` override. Task 5 builds the quiet-hw measurement surface (preflight TSC probe + three gate scripts). Task 6 runs the attribution sessions (manual, both machines). Task 7 evaluates the pre-registered gates — **no task before Task 7 may assume any gate's outcome.** Tasks 8–10 implement levers **A → D → W in that order**, each only if its gate fired, each carrying its own quiet-hw data point. Task 11 records the closing re-bench and closes the milestone.

**Tech Stack:** Rust (`inferno-pool`, `cli`), bash (`scripts/quiet-hw/`), mise tasks, PhoenixNAP bare metal via `mise run metal`.

**Spec:** [`docs/superpowers/specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md`](../specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md) (committed `254c48d`).

## Global Constraints

Copied verbatim from the spec. Every task's requirements implicitly include these.

- **Decode only.** Prefill (`inferno_par_gemm`, `inferno_par_attention`, `inferno_par_token_loop`, `run_attn_span`) is untouched.
- **Attribution-first, gated.** Levers ship only if their pre-registered gate fires (Task 7). An all-STOP with the memory-stall finding is a successful outcome.
- **Levers are pool-side only** — no `HOST_ABI_VERSION` bump, no recompile; a cached `model.so` benefits immediately.
- **Every lever is bit-invisible.** The full differential suite (`cargo test -p inferno-codegen --test differential`, `cargo test -p inferno-core --test artifact`) passes with **zero tolerance change** after every lever.
- **Landing order A → D → W**, one at a time, each with its own within-session data point; a lever that regresses tg on either machine triggers its pre-registered revert.
- **No new tuning constants beyond the two named ones** (`DECODE_SPIN_MULT` in Lever W — one named constant, same rule as `SPIN_ITERS`) and **no pos-/size-threshold heuristics**. An attention shard-count cap is explicitly out of scope even if the sweep shows a knee.
- **`INFERNO_ATTN_SHARDS` is probe-only** — measurement scripts only, never a tuning surface; inert when unset.
- **Instrument admissibility before any gate:** sum identity within 10%; perturbation A/B tg movement ≤ 1%. Gates are not evaluated on inadmissible data.
- **Never edit a recorded data point.** Session output is pasted verbatim into spec Amendments.
- **Scripts never write to `docs/`** — verdicts are pasted in by a human (`docs/runbooks/quiet-hw-verification.md`).
- **Workflows are mise tasks:** `mise run test` / `lint` / `metal`. Run `mise run lint` before every push (CI runs clippy `-D warnings`; `mise run test` alone skips it).
- Perf numbers come only from quiet bare metal (`mise run metal`); no CI perf gates.

## File Structure

| File | Responsibility |
|---|---|
| `crates/inferno-pool/Cargo.toml` (modify, Task 1) | `pool-profile` cargo feature. |
| `crates/inferno-pool/src/prof.rs` (create, Task 1) | The instrument: `PoolProfSnapshot` (always compiled), and — feature-gated — the `ProfState`/`LaneProf` accounting, `now()` (rdtsc), the enable flag, the `ALLOC_CYC` thread-local. |
| `crates/inferno-pool/src/lib.rs` (modify, Tasks 1) | `pub mod prof`, re-export, and the three no-op-without-feature API fns `set_pool_profiling` / `pool_prof_reset` / `pool_prof_snapshot`. |
| `crates/inferno-pool/src/pool.rs` (modify, Tasks 2, 4, 8, 9, 10) | Instrument hooks in `par_attention_heads` + `worker_loop` + `run_attn_heads_span` (Task 2); `INFERNO_ATTN_SHARDS` (Task 4); Lever A scratch (Task 8); Lever D publish slimming (Task 9); Lever W decode spin window (Task 10). |
| `mise.toml` (modify, Task 2) | `test`/`lint` gain the `--features pool-profile` invocations so the instrument can't rot. |
| `cli/Cargo.toml` (modify, Task 3) | Forwarding feature `pool-profile = ["inferno-pool/pool-profile"]`. |
| `cli/src/profile.rs` (modify, Task 3) | `render_pool()` — the `pool [decode attention]` section incl. the sum-identity line. |
| `cli/src/run.rs` (modify, Task 3) | `run_profile` enables/resets/snapshots pool profiling and prints the section. |
| `cli/src/main.rs` (modify, Task 3) | `INFERNO_POOL_PROF=1` enables recording outside `--profile` (the perturbation A/B benches with recording on). |
| `scripts/quiet-hw/preflight.sh` (modify, Task 5) | Probe 5: invariant-TSC cpuinfo flags. |
| `scripts/quiet-hw/preflight-selftest.sh` (modify, Task 5) | Fixture gains TSC flags; new UNFIT case without them. |
| `scripts/quiet-hw/gate-attn-split.sh` (create, Task 5) | Dispatch-split profile at best-t + the `INFERNO_ATTN_SHARDS` scaling sweep. |
| `scripts/quiet-hw/gate-attn-perturb.sh` (create, Task 5) | Admissibility #2: interleaved ship-vs-prof bench A/B. |
| `scripts/quiet-hw/gate-attn-perf.sh` (create, Task 5) | Rider: `perf stat` topdown/scheduler capture on the shipping build. |
| `scripts/quiet-hw/verify.sh` (modify, Task 5) | Wire the three new gates into the pass. |
| `docs/runbooks/quiet-hw-verification.md` (modify, Task 5) | Verdict-destination rows for the three gates. |
| `AGENTS.md` (modify, Tasks 5, 11) | Instrument/probe note (Task 5); lever notes at closure (Task 11). |
| `docs/superpowers/specs/2026-07-16-m4b12-...-design.md` (modify, Tasks 6, 7, 8, 9, 10, 11) | §Amendments: session records, gate verdicts, lever data points, closing verdict. |

---

### Task 1: `pool-profile` feature — the `prof` module and its API

The accounting types and the crate API, with no pool hooks yet. `PoolProfSnapshot` is always compiled (the CLI names it unconditionally); everything that costs anything is behind the feature.

**Files:**
- Modify: `crates/inferno-pool/Cargo.toml`
- Create: `crates/inferno-pool/src/prof.rs`
- Modify: `crates/inferno-pool/src/lib.rs`

**Interfaces:**
- Consumes: nothing new.
- Produces: `inferno_pool::PoolProfSnapshot` (pub fields below, plus `fn instr_total(&self) -> u64`); `inferno_pool::set_pool_profiling(on: bool)`, `inferno_pool::pool_prof_reset()`, `inferno_pool::pool_prof_snapshot() -> Option<PoolProfSnapshot>` (all three exist without the feature as no-ops / `None`); feature-gated internals `prof::ProfState`, `prof::LaneProf`, `prof::now()`, `prof::enabled()`, `prof::set_enabled()`, `prof::ALLOC_CYC` for Task 2's hooks.

- [ ] **Step 1: Declare the feature**

In `crates/inferno-pool/Cargo.toml`, after the `[dependencies]` section add:

```toml
[features]
# M4b.12 dispatch-split instrument (spec §The dispatch-split instrument).
# Off in every shipping/bench build; the quiet-hw gate scripts build with it.
pool-profile = []
```

- [ ] **Step 2: Write the failing unit tests**

Create `crates/inferno-pool/src/prof.rs` with ONLY the test module first:

```rust
#[cfg(all(test, feature = "pool-profile"))]
mod tests {
    use super::*;

    #[test]
    fn snapshot_instr_total_is_bucket_sum() {
        let st = ProfState::new(4);
        st.publish_cyc.fetch_add(10, std::sync::atomic::Ordering::Relaxed);
        st.kernel0_cyc.fetch_add(20, std::sync::atomic::Ordering::Relaxed);
        st.drain_cyc.fetch_add(30, std::sync::atomic::Ordering::Relaxed);
        let s = st.snapshot();
        assert_eq!(s.instr_total(), 60);
        assert_eq!(s.lane_kernel_cyc.len(), 4);
    }

    #[test]
    fn reset_zeroes_everything_but_capacity() {
        let st = ProfState::new(3);
        st.calls.fetch_add(7, std::sync::atomic::Ordering::Relaxed);
        st.lanes[1]
            .sum_wake
            .fetch_add(9, std::sync::atomic::Ordering::Relaxed);
        st.hist[5].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        st.reset();
        let s = st.snapshot();
        assert_eq!(s.calls, 0);
        assert_eq!(s.lane_wake_cyc, vec![0, 0, 0]);
        assert_eq!(s.hist_log2.iter().sum::<u64>(), 0);
    }

    #[test]
    fn enabled_flag_toggles() {
        set_enabled(true);
        assert!(enabled());
        set_enabled(false);
        assert!(!enabled());
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p inferno-pool --features pool-profile prof::`
Expected: FAIL to compile — `ProfState` not defined.

- [ ] **Step 4: Implement the module**

Fill in `crates/inferno-pool/src/prof.rs` above the test module:

```rust
//! M4b.12 dispatch-split instrument: per-call cycle accounting for
//! `Pool::par_attention_heads`, compiled only under the `pool-profile`
//! feature and recording only while enabled. Buckets (spec §The
//! dispatch-split instrument): publish, wake (per lane, with a parked
//! bit), kernel (per lane, with the scratch allocation bracketed as
//! H-alloc), drain. Self-measurement via invariant-TSC `rdtsc`; shares
//! guide scoping and never gate CI (M4b.2 rule).

/// The numbers `inferno run --profile` renders. Always compiled (the CLI
/// names this type without the feature); all counts are rdtsc cycles.
/// `lane_*` vectors are indexed by lane: 0 = the dispatching thread,
/// `i >= 1` = pool worker `i - 1`.
#[derive(Debug, Clone, Default)]
pub struct PoolProfSnapshot {
    pub calls: u64,
    pub publish_cyc: u64,
    pub kernel0_cyc: u64,
    pub drain_cyc: u64,
    /// Sum over calls of the max worker-lane wake latency.
    pub wake_max_cyc: u64,
    /// Same sum, restricted to calls whose max-wake lane had exhausted its
    /// spin window (park-eligible) while waiting — the P_W numerator.
    pub wake_parked_cyc: u64,
    /// Calls in which any participating lane was park-eligible.
    pub parked_calls: u64,
    /// Sum over calls of the max per-lane kernel cycles — C(n)'s numerator.
    pub kernel_max_cyc: u64,
    /// Sum over calls of the max per-lane scratch-alloc cycles — the P_A
    /// numerator (H-alloc).
    pub alloc_max_cyc: u64,
    pub lane_wake_cyc: Vec<u64>,
    pub lane_kernel_cyc: Vec<u64>,
    pub lane_alloc_cyc: Vec<u64>,
    pub lane_parked_calls: Vec<u64>,
    /// Per-call dispatcher-total histogram; bucket b counts calls whose
    /// total cycles had floor(log2) == b.
    pub hist_log2: Vec<u64>,
}

impl PoolProfSnapshot {
    /// Dispatcher-side identity: publish + kernel(shard 0) + drain
    /// partition each instrumented call exactly, so their sum is the
    /// instrument's whole-call total (admissibility check #1 compares it
    /// to the op profiler's attention cycles).
    pub fn instr_total(&self) -> u64 {
        self.publish_cyc + self.kernel0_cyc + self.drain_cyc
    }
}

#[cfg(feature = "pool-profile")]
pub(crate) use state::*;

#[cfg(feature = "pool-profile")]
mod state {
    use super::PoolProfSnapshot;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static ENABLED: AtomicBool = AtomicBool::new(false);

    pub fn set_enabled(on: bool) {
        ENABLED.store(on, Ordering::Relaxed);
    }

    #[inline]
    pub fn enabled() -> bool {
        ENABLED.load(Ordering::Relaxed)
    }

    /// Raw TSC read. Both target Xeons have invariant, synchronized TSC
    /// (the quiet-hw preflight asserts `constant_tsc nonstop_tsc`), so
    /// cross-thread deltas are meaningful there. Non-x86 builds return 0 —
    /// the instrument then records zeros, which the admissibility checks
    /// reject before any gate consumes them.
    #[inline]
    pub fn now() -> u64 {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: rdtsc has no memory or validity preconditions.
        unsafe {
            core::arch::x86_64::_rdtsc()
        }
        #[cfg(not(target_arch = "x86_64"))]
        0
    }

    thread_local! {
        /// Cycles the current thread spent obtaining the attention scratch
        /// in its most recent `run_attn_heads_span` call (the H-alloc
        /// bracket). Written by the span runner, consumed (and cleared) by
        /// the recording site on the same thread.
        pub static ALLOC_CYC: Cell<u64> = const { Cell::new(0) };
    }

    /// Per-lane accounting. `sum_*` accumulate across calls and are
    /// written ONLY by the dispatcher post-drain (single writer).
    /// `call_*` are the worker's per-call publication cells: the worker
    /// writes them (Relaxed) before its Release `remaining` decrement; the
    /// dispatcher reads them (Relaxed) only after its Acquire read of
    /// `remaining == 0` — the standing pool handshake orders them.
    #[derive(Default)]
    pub struct LaneProf {
        pub sum_wake: AtomicU64,
        pub sum_kernel: AtomicU64,
        pub sum_alloc: AtomicU64,
        pub sum_parked_calls: AtomicU64,
        pub call_wake: AtomicU64,
        pub call_kernel: AtomicU64,
        pub call_alloc: AtomicU64,
        pub call_parked: AtomicBool,
    }

    /// Pool-wide accounting; one per `Shared`, sized to the pool capacity.
    pub struct ProfState {
        /// TSC at publish, stored (SeqCst) before the epoch bump so any
        /// worker that observes the new epoch also observes this value.
        pub dispatch_tsc: AtomicU64,
        pub calls: AtomicU64,
        pub publish_cyc: AtomicU64,
        pub kernel0_cyc: AtomicU64,
        pub drain_cyc: AtomicU64,
        pub wake_max_cyc: AtomicU64,
        pub wake_parked_cyc: AtomicU64,
        pub parked_calls: AtomicU64,
        pub kernel_max_cyc: AtomicU64,
        pub alloc_max_cyc: AtomicU64,
        pub hist: [AtomicU64; 64],
        pub lanes: Vec<LaneProf>,
    }

    impl ProfState {
        pub fn new(capacity: usize) -> ProfState {
            ProfState {
                dispatch_tsc: AtomicU64::new(0),
                calls: AtomicU64::new(0),
                publish_cyc: AtomicU64::new(0),
                kernel0_cyc: AtomicU64::new(0),
                drain_cyc: AtomicU64::new(0),
                wake_max_cyc: AtomicU64::new(0),
                wake_parked_cyc: AtomicU64::new(0),
                parked_calls: AtomicU64::new(0),
                kernel_max_cyc: AtomicU64::new(0),
                alloc_max_cyc: AtomicU64::new(0),
                hist: std::array::from_fn(|_| AtomicU64::new(0)),
                lanes: (0..capacity).map(|_| LaneProf::default()).collect(),
            }
        }

        pub fn reset(&self) {
            for a in [
                &self.calls,
                &self.publish_cyc,
                &self.kernel0_cyc,
                &self.drain_cyc,
                &self.wake_max_cyc,
                &self.wake_parked_cyc,
                &self.parked_calls,
                &self.kernel_max_cyc,
                &self.alloc_max_cyc,
            ] {
                a.store(0, Ordering::Relaxed);
            }
            for b in &self.hist {
                b.store(0, Ordering::Relaxed);
            }
            for l in &self.lanes {
                l.sum_wake.store(0, Ordering::Relaxed);
                l.sum_kernel.store(0, Ordering::Relaxed);
                l.sum_alloc.store(0, Ordering::Relaxed);
                l.sum_parked_calls.store(0, Ordering::Relaxed);
            }
        }

        pub fn snapshot(&self) -> PoolProfSnapshot {
            let r = Ordering::Relaxed;
            PoolProfSnapshot {
                calls: self.calls.load(r),
                publish_cyc: self.publish_cyc.load(r),
                kernel0_cyc: self.kernel0_cyc.load(r),
                drain_cyc: self.drain_cyc.load(r),
                wake_max_cyc: self.wake_max_cyc.load(r),
                wake_parked_cyc: self.wake_parked_cyc.load(r),
                parked_calls: self.parked_calls.load(r),
                kernel_max_cyc: self.kernel_max_cyc.load(r),
                alloc_max_cyc: self.alloc_max_cyc.load(r),
                lane_wake_cyc: self.lanes.iter().map(|l| l.sum_wake.load(r)).collect(),
                lane_kernel_cyc: self.lanes.iter().map(|l| l.sum_kernel.load(r)).collect(),
                lane_alloc_cyc: self.lanes.iter().map(|l| l.sum_alloc.load(r)).collect(),
                lane_parked_calls: self
                    .lanes
                    .iter()
                    .map(|l| l.sum_parked_calls.load(r))
                    .collect(),
                hist_log2: self.hist.iter().map(|b| b.load(r)).collect(),
            }
        }

        /// Record the single-shard fast path (no publish, no drain): the
        /// whole call is dispatcher kernel time. Also feeds C(1) in the
        /// shard-count sweep.
        pub fn record_single(&self, t0: u64, t1: u64) {
            let k0 = t1.saturating_sub(t0);
            let alloc0 = ALLOC_CYC.with(|c| c.replace(0));
            let r = Ordering::Relaxed;
            self.calls.fetch_add(1, r);
            self.kernel0_cyc.fetch_add(k0, r);
            self.kernel_max_cyc.fetch_add(k0, r);
            self.alloc_max_cyc.fetch_add(alloc0, r);
            self.lanes[0].sum_kernel.fetch_add(k0, r);
            self.lanes[0].sum_alloc.fetch_add(alloc0, r);
            self.hist[Self::bucket(k0)].fetch_add(1, r);
        }

        /// Record a pooled call post-drain. `t0` = call entry, `t2` = after
        /// the unpark loop, `t3` = dispatcher's own span done, `t4` = drain
        /// observed zero. Reads lanes `1..=n_worker`'s publication cells —
        /// every one of them participated in THIS dispatch and published
        /// before decrementing `remaining`.
        pub fn record_call(&self, t0: u64, t2: u64, t3: u64, t4: u64, n_worker: usize) {
            let r = Ordering::Relaxed;
            let publish = t2.saturating_sub(t0);
            let k0 = t3.saturating_sub(t2);
            let drain = t4.saturating_sub(t3);
            let alloc0 = ALLOC_CYC.with(|c| c.replace(0));
            self.calls.fetch_add(1, r);
            self.publish_cyc.fetch_add(publish, r);
            self.kernel0_cyc.fetch_add(k0, r);
            self.drain_cyc.fetch_add(drain, r);
            self.lanes[0].sum_kernel.fetch_add(k0, r);
            self.lanes[0].sum_alloc.fetch_add(alloc0, r);
            let (mut wake_max, mut wake_max_parked) = (0u64, false);
            let mut kernel_max = k0;
            let mut alloc_max = alloc0;
            let mut any_parked = false;
            for lane in &self.lanes[1..=n_worker] {
                let w = lane.call_wake.load(r);
                let k = lane.call_kernel.load(r);
                let a = lane.call_alloc.load(r);
                let p = lane.call_parked.load(r);
                lane.sum_wake.fetch_add(w, r);
                lane.sum_kernel.fetch_add(k, r);
                lane.sum_alloc.fetch_add(a, r);
                if p {
                    lane.sum_parked_calls.fetch_add(1, r);
                    any_parked = true;
                }
                if w > wake_max {
                    wake_max = w;
                    wake_max_parked = p;
                }
                kernel_max = kernel_max.max(k);
                alloc_max = alloc_max.max(a);
            }
            self.wake_max_cyc.fetch_add(wake_max, r);
            if wake_max_parked {
                self.wake_parked_cyc.fetch_add(wake_max, r);
            }
            if any_parked {
                self.parked_calls.fetch_add(1, r);
            }
            self.kernel_max_cyc.fetch_add(kernel_max, r);
            self.alloc_max_cyc.fetch_add(alloc_max, r);
            self.hist[Self::bucket(t4.saturating_sub(t0))].fetch_add(1, r);
        }

        fn bucket(cycles: u64) -> usize {
            63 - cycles.max(1).leading_zeros() as usize
        }
    }
}
```

- [ ] **Step 5: Wire the crate API into `lib.rs`**

In `crates/inferno-pool/src/lib.rs`: add `pub mod prof;` to the module list (after `pub mod probe;`), add `pub use prof::PoolProfSnapshot;` after the existing `pub use` lines, and append after `set_global_decode_threads`:

```rust
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
```

`Pool::prof_reset` / `Pool::prof_snapshot` don't exist yet — add them now in `crates/inferno-pool/src/pool.rs` inside `impl Pool` (they need the `Shared.prof` field, which is also Task 2's hook anchor; add the field here so Task 1 compiles standalone). In `struct Shared`, after `slots`:

```rust
    /// M4b.12 dispatch-split accounting (feature-gated; spec §The
    /// dispatch-split instrument).
    #[cfg(feature = "pool-profile")]
    prof: crate::prof::ProfState,
```

In `Pool::new`, in the `Shared { ... }` initializer after the `slots` field:

```rust
            #[cfg(feature = "pool-profile")]
            prof: crate::prof::ProfState::new(capacity),
```

In `impl Pool`, after `decode_threads()`:

```rust
    /// Snapshot the M4b.12 dispatch-split accounting.
    #[cfg(feature = "pool-profile")]
    pub fn prof_snapshot(&self) -> crate::prof::PoolProfSnapshot {
        self.shared.prof.snapshot()
    }

    /// Zero the M4b.12 dispatch-split accounting.
    #[cfg(feature = "pool-profile")]
    pub fn prof_reset(&self) {
        self.shared.prof.reset();
    }
```

(`prof::PoolProfSnapshot` resolves — the type is defined at module top level; `pool_prof_snapshot`'s signature uses the re-export.)

- [ ] **Step 6: Run tests to verify they pass, both flavors**

Run: `cargo test -p inferno-pool --features pool-profile prof::`
Expected: 3 passed.
Run: `cargo test -p inferno-pool` and `cargo build -p inferno-pool`
Expected: existing tests pass; default build compiles with zero new code in the hot path.

- [ ] **Step 7: Commit**

```bash
git add crates/inferno-pool/Cargo.toml crates/inferno-pool/src/prof.rs crates/inferno-pool/src/lib.rs crates/inferno-pool/src/pool.rs
git commit -m "pool: M4b.12 pool-profile feature — dispatch-split accounting types + API"
```

---

### Task 2: Instrument hooks in the dispatcher, worker loop, and span runner

The four buckets get recorded. All hooks are `#[cfg(feature = "pool-profile")]`; with the feature off, `pool.rs`'s hot loop is byte-for-byte what it was.

**Files:**
- Modify: `crates/inferno-pool/src/pool.rs`
- Modify: `mise.toml`

**Interfaces:**
- Consumes: Task 1's `prof::{ProfState, enabled, now, ALLOC_CYC}`.
- Produces: recorded buckets retrievable via `Pool::prof_snapshot()`; Task 3 renders them.

- [ ] **Step 1: Write the failing tests**

Append to the `tests` module in `crates/inferno-pool/src/pool.rs` (these reuse the existing `attn_heads_dispatch`/`attn_heads_expected` and `dispatch`/`expected` helpers; nextest's process-per-test isolation keeps the global `ENABLED` flag from leaking between tests):

```rust
    #[cfg(feature = "pool-profile")]
    mod prof_hooks {
        use super::*;

        #[test]
        fn records_attn_heads_dispatches() {
            let pool = Pool::new(4);
            crate::prof::set_enabled(true);
            for pos in [0, 9, 100] {
                assert_eq!(attn_heads_dispatch(&pool, pos), attn_heads_expected(pos));
            }
            crate::prof::set_enabled(false);
            let s = pool.prof_snapshot();
            assert_eq!(s.calls, 3);
            assert_eq!(s.instr_total(), s.publish_cyc + s.kernel0_cyc + s.drain_cyc);
            assert!(s.kernel0_cyc > 0, "dispatcher span time must be recorded");
            assert!(s.kernel_max_cyc >= s.kernel0_cyc);
            assert_eq!(s.hist_log2.iter().sum::<u64>(), s.calls);
            // 14 heads over 4 lanes: every lane got a span every call.
            assert!(s.lane_kernel_cyc.iter().all(|&k| k > 0), "{:?}", s.lane_kernel_cyc);
        }

        #[test]
        fn disabled_records_nothing() {
            let pool = Pool::new(4);
            crate::prof::set_enabled(false);
            attn_heads_dispatch(&pool, 9);
            assert_eq!(pool.prof_snapshot().calls, 0);
        }

        #[test]
        fn gemv_dispatches_are_not_recorded() {
            let pool = Pool::new(4);
            crate::prof::set_enabled(true);
            assert_eq!(dispatch(&pool, 512, 1), expected(512, 1));
            crate::prof::set_enabled(false);
            assert_eq!(pool.prof_snapshot().calls, 0);
        }

        #[test]
        fn single_shard_path_records_kernel_only() {
            let pool = Pool::new(1);
            crate::prof::set_enabled(true);
            attn_heads_dispatch(&pool, 9);
            crate::prof::set_enabled(false);
            let s = pool.prof_snapshot();
            assert_eq!(s.calls, 1);
            assert_eq!(s.publish_cyc, 0);
            assert_eq!(s.drain_cyc, 0);
            assert!(s.kernel0_cyc > 0);
        }

        #[test]
        fn park_eligible_wait_sets_parked_bit() {
            let pool = Pool::new(4);
            crate::prof::set_enabled(true);
            attn_heads_dispatch(&pool, 9);
            // Well past the spin window: workers are park-eligible now.
            std::thread::sleep(std::time::Duration::from_millis(100));
            attn_heads_dispatch(&pool, 9);
            crate::prof::set_enabled(false);
            let s = pool.prof_snapshot();
            assert!(s.parked_calls >= 1, "parked_calls = {}", s.parked_calls);
            assert!(s.wake_parked_cyc > 0);
        }

        #[test]
        fn reset_zeroes_between_phases() {
            let pool = Pool::new(4);
            crate::prof::set_enabled(true);
            attn_heads_dispatch(&pool, 9);
            pool.prof_reset();
            attn_heads_dispatch(&pool, 9);
            crate::prof::set_enabled(false);
            assert_eq!(pool.prof_snapshot().calls, 1);
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p inferno-pool --features pool-profile prof_hooks`
Expected: FAIL — `calls` stays 0 (no hooks yet).

- [ ] **Step 3: Hook the span runner (H-alloc bracket)**

In `run_attn_heads_span`, bracket the scratch allocation:

```rust
pub(crate) unsafe fn run_attn_heads_span(j: &AttnHeadsJob, start: usize, end: usize) {
    #[cfg(feature = "pool-profile")]
    let a0 = crate::prof::now();
    let mut scores = vec![0f32; j.pos + 1];
    #[cfg(feature = "pool-profile")]
    crate::prof::ALLOC_CYC.with(|c| c.set(crate::prof::now().saturating_sub(a0)));
```

(rest of the function unchanged).

- [ ] **Step 4: Hook the dispatcher**

Replace `Pool::par_attention_heads`'s body with (the un-annotated lines are today's code, unchanged and in the same order):

```rust
    pub unsafe fn par_attention_heads(&self, job: &AttnHeadsJob) {
        let n_heads = job.n_heads;
        if n_heads == 0 {
            return;
        }
        #[cfg(feature = "pool-profile")]
        let rec = crate::prof::enabled();
        #[cfg(feature = "pool-profile")]
        let t0 = if rec { crate::prof::now() } else { 0 };
        let active = self.active_threads().min(self.decode_threads());
        let shards = shard_table_aligned(n_heads, active, 1);
        if shards.len() == 1 {
            // SAFETY: caller contract covers the full head range.
            unsafe { run_attn_heads_span(job, 0, n_heads) };
            #[cfg(feature = "pool-profile")]
            if rec {
                self.shared.prof.record_single(t0, crate::prof::now());
            }
            return;
        }
        let n_worker = shards.len() - 1;
        let (s0, e0) = shards[0];
        let kind = JobKind::AttnHeads(*job);
        // SAFETY (job write): the previous dispatch ended with
        // `remaining == 0`, and shardless workers never read `job`
        // (packed-epoch protocol) — no reader exists here.
        unsafe {
            *self.shared.job.get() = Job {
                kind: Some(kind),
                y: std::ptr::null_mut(),
                xq: std::ptr::null(),
                w: std::ptr::null(),
                k: 0,
                shards,
            };
        }
        self.shared.remaining.store(n_worker, Ordering::SeqCst);
        // Publish the dispatch timestamp BEFORE the epoch bump: both are
        // SeqCst, so a worker that observes the new epoch also observes
        // this value (single total order).
        #[cfg(feature = "pool-profile")]
        if rec {
            self.shared
                .prof
                .dispatch_tsc
                .store(crate::prof::now(), Ordering::SeqCst);
        }
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
        #[cfg(feature = "pool-profile")]
        let t2 = if rec { crate::prof::now() } else { 0 };
        // SAFETY: caller contract; shard 0's heads are disjoint from
        // worker shards.
        unsafe { run_attn_heads_span(job, s0, e0) };
        #[cfg(feature = "pool-profile")]
        let t3 = if rec { crate::prof::now() } else { 0 };
        let mut spins = 0u32;
        while self.shared.remaining.load(Ordering::Acquire) != 0 {
            if spins < SPIN_ITERS {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        #[cfg(feature = "pool-profile")]
        if rec {
            self.shared
                .prof
                .record_call(t0, t2, t3, crate::prof::now(), n_worker);
        }
    }
```

- [ ] **Step 5: Hook the worker loop**

In `worker_loop`, three insertions (surrounding code unchanged):

Before the inner wait loop (right after `loop {` / `let mut spins = 0u32;`):

```rust
        #[cfg(feature = "pool-profile")]
        let mut spun_out = false;
```

Inside the wait loop's park-eligible branch (the `else` arm, right after `let slot = &shared.slots[idx];`):

```rust
                #[cfg(feature = "pool-profile")]
                {
                    spun_out = true;
                }
```

After `seen = epoch;` and before `let n_shards = ...`:

```rust
        #[cfg(feature = "pool-profile")]
        let t_start = crate::prof::now();
```

Then, replacing the bare `unsafe { run_shard(...) };` at the bottom (the decrement stays last — workers publish their cells BEFORE the Release decrement, which is what makes the dispatcher's post-drain read sound):

```rust
        #[cfg(feature = "pool-profile")]
        let rec = crate::prof::enabled() && matches!(kind, JobKind::AttnHeads(_));
        #[cfg(feature = "pool-profile")]
        if rec {
            crate::prof::ALLOC_CYC.with(|c| c.set(0));
        }
        // SAFETY: dispatcher's caller contract covers this disjoint range.
        unsafe { run_shard(&kind, y, xq, w, k, start, end) };
        #[cfg(feature = "pool-profile")]
        if rec {
            let t_end = crate::prof::now();
            let lane = &shared.prof.lanes[idx + 1];
            let wake =
                t_start.saturating_sub(shared.prof.dispatch_tsc.load(Ordering::SeqCst));
            lane.call_wake.store(wake, Ordering::Relaxed);
            lane.call_kernel
                .store(t_end.saturating_sub(t_start), Ordering::Relaxed);
            lane.call_alloc
                .store(crate::prof::ALLOC_CYC.with(|c| c.get()), Ordering::Relaxed);
            lane.call_parked.store(spun_out, Ordering::Relaxed);
        }
        shared.remaining.fetch_sub(1, Ordering::Release);
```

- [ ] **Step 6: Run tests to verify they pass, both flavors**

Run: `cargo test -p inferno-pool --features pool-profile`
Expected: all pass, including `prof_hooks::*`.
Run: `cargo test -p inferno-pool`
Expected: all existing tests pass (feature off — no behavior change).

- [ ] **Step 7: Keep the feature build honest in CI**

In `mise.toml` (note: editing this file invalidates the GitHub Actions tool cache — one ~10 min CI run, then it self-heals; accepted, the spec requires CI coverage of the instrument):

```toml
[tasks.test]
description = "Blocking-tier tests (fast; what PR CI runs)"
run = [
  "cargo nextest run --workspace --no-tests=pass",
  "cargo nextest run -p inferno-pool --features pool-profile --no-tests=pass",
]
```

and in `[tasks.lint]`, append to the `run` list:

```toml
  "cargo clippy -p inferno-pool --features pool-profile --all-targets -- -D warnings",
```

- [ ] **Step 8: Run the full check and commit**

Run: `mise run test && mise run lint`
Expected: both pass.

```bash
git add crates/inferno-pool/src/pool.rs mise.toml
git commit -m "pool: M4b.12 dispatch-split hooks — publish/wake/kernel/drain + H-alloc bracket"
```

---

### Task 3: CLI plumbing — `pool [decode attention]` section under `--profile`

**Files:**
- Modify: `cli/Cargo.toml`
- Modify: `cli/src/profile.rs`
- Modify: `cli/src/run.rs`
- Modify: `cli/src/main.rs`

**Interfaces:**
- Consumes: `inferno_pool::{set_pool_profiling, pool_prof_reset, pool_prof_snapshot, PoolProfSnapshot}` (Task 1).
- Produces: `crate::profile::render_pool(snap: &PoolProfSnapshot, op_attention_cyc: u64) -> String`; the printed section the gate scripts parse (`^pool \[decode attention\]`); `INFERNO_POOL_PROF=1` recording for non-`--profile` commands.

- [ ] **Step 1: Write the failing render test**

Append to the `tests` module in `cli/src/profile.rs`:

```rust
    #[test]
    fn render_pool_prints_buckets_and_identity() {
        let snap = inferno_pool::PoolProfSnapshot {
            calls: 3,
            publish_cyc: 100,
            kernel0_cyc: 700,
            drain_cyc: 200,
            wake_max_cyc: 90,
            wake_parked_cyc: 60,
            parked_calls: 1,
            kernel_max_cyc: 750,
            alloc_max_cyc: 30,
            lane_wake_cyc: vec![0, 90],
            lane_kernel_cyc: vec![700, 740],
            lane_alloc_cyc: vec![30, 25],
            lane_parked_calls: vec![0, 1],
            hist_log2: {
                let mut h = vec![0u64; 64];
                h[9] = 3;
                h
            },
        };
        let out = super::render_pool(&snap, 1100);
        assert!(out.starts_with("pool [decode attention] 3 calls"), "{out}");
        // instr_total = 1000, op attention = 1100 → 90.9%.
        assert!(out.contains("90.9%"), "{out}");
        assert!(out.contains("publish"), "{out}");
        assert!(out.contains("kernel(shard0)"), "{out}");
        assert!(out.contains("drain"), "{out}");
        assert!(out.contains("wake-parked 60 (1 calls)"), "{out}");
        assert!(out.contains("2^9:3"), "{out}");
        // publish share of instr total: 100/1000.
        assert!(out.contains("10.0%"), "{out}");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p inferno render_pool`
Expected: FAIL to compile — `render_pool` not defined.

- [ ] **Step 3: Implement `render_pool`**

Append to `cli/src/profile.rs` (below `render`, above the tests):

```rust
/// Render the M4b.12 `pool [decode attention]` dispatch-split section.
/// `op_attention_cyc` is the op profiler's decode attention cycle count,
/// for the sum-identity admissibility line (spec: within 10%). Cycle
/// numbers are printed raw — decode-wall shares and gate arithmetic are
/// controller work in the spec's Amendments, never computed here.
pub fn render_pool(s: &inferno_pool::PoolProfSnapshot, op_attention_cyc: u64) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let total = s.instr_total();
    writeln!(
        out,
        "pool [decode attention] {} calls, {} cyc instrumented",
        s.calls, total
    )
    .unwrap();
    let identity = if op_attention_cyc > 0 {
        total as f64 / op_attention_cyc as f64 * 100.0
    } else {
        0.0
    };
    writeln!(
        out,
        "  sum identity vs op-profiler attention: {identity:.1}% (admissible: 90-110%)"
    )
    .unwrap();
    let share = |c: u64| {
        if total > 0 {
            c as f64 / total as f64 * 100.0
        } else {
            0.0
        }
    };
    writeln!(out, "  {:<16} {:>14} {:>7}", "bucket", "cycles", "share").unwrap();
    for (name, c) in [
        ("publish", s.publish_cyc),
        ("kernel(shard0)", s.kernel0_cyc),
        ("drain", s.drain_cyc),
    ] {
        writeln!(out, "  {:<16} {:>14} {:>6.1}%", name, c, share(c)).unwrap();
    }
    writeln!(
        out,
        "  per-call max-lane sums: wake {} | wake-parked {} ({} calls) | kernel-max {} | alloc-max {}",
        s.wake_max_cyc, s.wake_parked_cyc, s.parked_calls, s.kernel_max_cyc, s.alloc_max_cyc
    )
    .unwrap();
    writeln!(
        out,
        "  {:<6} {:>14} {:>14} {:>14} {:>13}",
        "lane", "wake", "kernel", "alloc", "parked-calls"
    )
    .unwrap();
    for i in 0..s.lane_kernel_cyc.len() {
        writeln!(
            out,
            "  {:<6} {:>14} {:>14} {:>14} {:>13}",
            i, s.lane_wake_cyc[i], s.lane_kernel_cyc[i], s.lane_alloc_cyc[i],
            s.lane_parked_calls[i]
        )
        .unwrap();
    }
    let mut hist = String::from("  per-call cycles histogram:");
    for (b, &n) in s.hist_log2.iter().enumerate() {
        if n > 0 {
            write!(hist, " 2^{b}:{n}").unwrap();
        }
    }
    writeln!(out, "{hist}").unwrap();
    out
}
```

- [ ] **Step 4: Enable, reset, snapshot, print in `run_profile`**

In `cli/src/run.rs`, `fn run_profile`, three insertions:

After `engine.set_profile(true);`:

```rust
    // M4b.12: dispatch-split recording (no-op unless built with
    // --features pool-profile).
    inferno_pool::set_pool_profiling(true);
```

After `backend.profile_reset();` (the between-phases reset):

```rust
    inferno_pool::pool_prof_reset();
```

At the end, after the decode-table `print!`, before `Ok(())`:

```rust
    // M4b.12 dispatch-split section (only prints on a pool-profile build).
    if let Some(snap) = inferno_pool::pool_prof_snapshot() {
        if snap.calls > 0 {
            let attn_cyc = slots
                .iter()
                .position(|s| s == "attention")
                .map(|i| decode_counts[i])
                .unwrap_or(0);
            print!("{}", crate::profile::render_pool(&snap, attn_cyc));
        }
    }
```

- [ ] **Step 5: `INFERNO_POOL_PROF=1` for non-`--profile` commands**

At the top of `fn main()` in `cli/src/main.rs` (first statement):

```rust
    // M4b.12: the instrument-perturbation A/B benches with recording ON but
    // without --profile; no-op unless built with --features pool-profile.
    inferno_pool::set_pool_profiling(std::env::var("INFERNO_POOL_PROF").is_ok_and(|v| v == "1"));
```

- [ ] **Step 6: Forward the feature**

In `cli/Cargo.toml`, add:

```toml
[features]
pool-profile = ["inferno-pool/pool-profile"]
```

- [ ] **Step 7: Run tests to verify they pass, both flavors**

Run: `cargo test -p inferno render_pool && cargo build --release -p inferno && cargo build --release -p inferno --features pool-profile`
Expected: test passes; both builds compile (the CLI code is unconditional — only `inferno-pool` has cfg).

- [ ] **Step 8: End-to-end smoke**

Run (devpod is fine — plumbing only; fetch the model with `bash scripts/fetch-qwen-gguf.sh` if absent):

```bash
cargo run --release -q -p inferno --features pool-profile -- run models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "hello there" --max-tokens 8 --threads 4 --profile | tail -20
```

Expected: after the `profile [decode]` table, a `pool [decode attention] N calls...` section with a sum-identity line and nonzero kernel cycles. Then confirm the shipping build prints no section:

```bash
cargo run --release -q -p inferno -- run models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "hello there" --max-tokens 8 --threads 4 --profile | grep -c "^pool" || true
```

Expected: `0`.

- [ ] **Step 9: Commit**

```bash
git add cli/Cargo.toml cli/src/profile.rs cli/src/run.rs cli/src/main.rs
git commit -m "cli: M4b.12 pool [decode attention] dispatch-split section under --profile"
```

---

### Task 4: `INFERNO_ATTN_SHARDS` — the probe-only shard-count override

Forces the decode-attention shard count without touching the GEMV decode cap (which `INFERNO_DECODE_THREADS` would move, confounding the sweep). Probe-only; inert when unset.

**Files:**
- Modify: `crates/inferno-pool/src/pool.rs`

**Interfaces:**
- Consumes: nothing new.
- Produces: the env-var behavior `gate-attn-split.sh` (Task 5) sweeps; internal `Pool::par_attention_heads_at(&self, job, active)` used by tests.

- [ ] **Step 1: Write the failing tests**

Append to the `tests` module in `crates/inferno-pool/src/pool.rs`:

```rust
    #[test]
    fn parse_attn_shards_accepts_positive_ints_only() {
        assert_eq!(parse_attn_shards(None), None);
        assert_eq!(parse_attn_shards(Some("")), None);
        assert_eq!(parse_attn_shards(Some("0")), None);
        assert_eq!(parse_attn_shards(Some("abc")), None);
        assert_eq!(parse_attn_shards(Some("-3")), None);
        assert_eq!(parse_attn_shards(Some("7")), Some(7));
        assert_eq!(parse_attn_shards(Some(" 14 ")), Some(14));
    }

    #[test]
    fn forced_shard_counts_reproduce_unsharded_output() {
        // The probe can only regroup heads: any forced count is bit-identical.
        let pool = Pool::new(16);
        for forced in [1, 2, 4, 7, 14, 16, 99] {
            for pos in [0, 9, 100] {
                let q: Vec<f32> = (0..AH_NH * AH_HD).map(|i| i as f32).collect();
                let mut out = vec![f32::NAN; AH_NH * AH_HD];
                let mut kv = [0f32; 1];
                let job = AttnHeadsJob {
                    kernel: stamp_attn_heads,
                    out: out.as_mut_ptr(),
                    q: q.as_ptr(),
                    kv: kv.as_mut_ptr(),
                    pos,
                    kv_base: 0,
                    v_off: 0,
                    kv_dim: 0,
                    n_heads: AH_NH,
                    n_kv_heads: 2,
                    head_dim: AH_HD,
                };
                // SAFETY: buffers sized per stamp_attn_heads' expectations.
                unsafe { pool.par_attention_heads_at(&job, forced.min(pool.capacity())) };
                assert_eq!(out, attn_heads_expected(pos), "forced={forced} pos={pos}");
            }
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p inferno-pool attn_shards`
Expected: FAIL to compile — `parse_attn_shards` / `par_attention_heads_at` not defined.

- [ ] **Step 3: Implement**

In `crates/inferno-pool/src/pool.rs`, above `impl Pool` add (needs `use std::sync::OnceLock;` — already imported in this file):

```rust
/// M4b.12 probe-only override: `INFERNO_ATTN_SHARDS=N` forces the
/// decode-attention lane count, so the attribution shard sweep can move
/// attention parallelism without touching the GEMV decode cap
/// (`INFERNO_DECODE_THREADS` would confound the curve). Read once; unset
/// or unparsable = no effect. Measurement scripts only — never a tuning
/// surface (spec §Explicitly out of scope: no shard-count cap).
fn attn_shards_override() -> Option<usize> {
    static V: OnceLock<Option<usize>> = OnceLock::new();
    *V.get_or_init(|| parse_attn_shards(std::env::var("INFERNO_ATTN_SHARDS").ok().as_deref()))
}

fn parse_attn_shards(v: Option<&str>) -> Option<usize> {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
}
```

Split `par_attention_heads` into the public wrapper plus a lane-count-explicit body — the wrapper keeps today's doc comment plus one added sentence, and the body is today's function verbatim from `let shards = ...` down, with `active` now a parameter:

```rust
    /// (existing doc comment stays; append:)
    /// `INFERNO_ATTN_SHARDS` (M4b.12, probe-only) forces the lane count.
    ///
    /// # Safety
    /// (unchanged)
    pub unsafe fn par_attention_heads(&self, job: &AttnHeadsJob) {
        let active = self.active_threads().min(self.decode_threads());
        let active = attn_shards_override()
            .map(|n| n.min(self.capacity))
            .unwrap_or(active);
        // SAFETY: forwarding the caller's contract.
        unsafe { self.par_attention_heads_at(job, active) };
    }

    /// [`Self::par_attention_heads`] with an explicit lane count — the
    /// testable seam under the env override (env is process-global, so
    /// tests force counts here instead).
    ///
    /// # Safety
    /// As [`Self::par_attention_heads`]; `active >= 1`.
    pub(crate) unsafe fn par_attention_heads_at(&self, job: &AttnHeadsJob, active: usize) {
        let n_heads = job.n_heads;
        if n_heads == 0 {
            return;
        }
        #[cfg(feature = "pool-profile")]
        let rec = crate::prof::enabled();
        #[cfg(feature = "pool-profile")]
        let t0 = if rec { crate::prof::now() } else { 0 };
        let shards = shard_table_aligned(n_heads, active, 1);
        // ... today's body from here down, verbatim (single-shard early
        // return, publish, dispatch-tsc store, epoch, unpark, own span,
        // drain, record_call) ...
    }
```

(The `n_heads == 0` check and the two profiling lines move INTO `par_attention_heads_at`; do not retype the rest — cut and paste it.)

- [ ] **Step 4: Run tests to verify they pass, both flavors**

Run: `cargo test -p inferno-pool && cargo test -p inferno-pool --features pool-profile`
Expected: all pass (the new two plus every existing test — the wrapper path with the env unset is behavior-identical).

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-pool/src/pool.rs
git commit -m "pool: M4b.12 INFERNO_ATTN_SHARDS probe-only shard-count override"
```

---

### Task 5: Quiet-hw measurement surface — preflight TSC probe + three gates

**Files:**
- Modify: `scripts/quiet-hw/preflight.sh`
- Modify: `scripts/quiet-hw/preflight-selftest.sh`
- Create: `scripts/quiet-hw/gate-attn-split.sh`
- Create: `scripts/quiet-hw/gate-attn-perturb.sh`
- Create: `scripts/quiet-hw/gate-attn-perf.sh`
- Modify: `scripts/quiet-hw/verify.sh`
- Modify: `docs/runbooks/quiet-hw-verification.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: `lib.sh` helpers `smoke_header`, `machine_block`, `phys_cores` (existing); Task 3's `--profile` pool section and `INFERNO_POOL_PROF`; Task 4's `INFERNO_ATTN_SHARDS`.
- Produces: `gate-attn-split.out` / `gate-attn-perturb.out` / `gate-attn-perf.out` — the raw records Task 7's arithmetic consumes.

- [ ] **Step 1: Preflight Probe 5 — invariant TSC**

In `scripts/quiet-hw/preflight.sh`, after Probe 4's `fi` and before `machine_block`:

```bash
# Probe 5 — invariant TSC (M4b.12: the dispatch-split instrument compares
# rdtsc across threads; only meaningful with constant+nonstop TSC).
tsc_flags=$(awk '/^flags/ { print; exit }' "$PROC/cpuinfo")
tsc_summary=ok
for f in constant_tsc nonstop_tsc; do
  case " $tsc_flags " in
    *" $f "*) ;;
    *) fails+=("tsc: cpuinfo flags lack $f"); tsc_summary=missing ;;
  esac
done
```

and extend the probes summary line:

```bash
echo "probes: cpus=$NPROC quota=$quota_summary psi_some_avg10=${psi:-?} throttled_delta=$((after - before)) calib=${CALIB_SECS}s tsc=$tsc_summary"
```

- [ ] **Step 2: Update the selftest fixture and add the TSC UNFIT case**

In `scripts/quiet-hw/preflight-selftest.sh`, change `mktree`'s cpuinfo line to include the flags (otherwise the FIT case now fails):

```bash
  printf 'vendor_id\t: FakeVendor\nmodel name\t: Fake CPU\nflags\t\t: fpu constant_tsc nonstop_tsc\n' > "$root/proc/cpuinfo"
```

and append before the final `echo "preflight-selftest: OK"`:

```bash
# UNFIT: missing invariant-TSC flags (M4b.12 probe 5).
root=$(mktree "max 100000" "0.10")
printf 'vendor_id\t: FakeVendor\nmodel name\t: Fake CPU\nflags\t\t: fpu sse2\n' > "$root/proc/cpuinfo"
out=$(run_pf "$root" 2>&1) && fail "missing-TSC expected nonzero exit"
echo "$out" | grep -q "tsc: cpuinfo flags lack constant_tsc" || fail "missing-TSC: no tsc reason: $out"
```

Run: `bash scripts/quiet-hw/preflight-selftest.sh`
Expected: `preflight-selftest: OK`.

- [ ] **Step 3: `gate-attn-split.sh` — dispatch-split profile + shard sweep**

Create `scripts/quiet-hw/gate-attn-split.sh` (then `chmod +x` it):

```bash
#!/usr/bin/env bash
# M4b.12 attribution gate — the dispatch-split profile the pre-registered
# gates consume, then the INFERNO_ATTN_SHARDS scaling sweep (menu guard's
# C(n) curve). Both run on a pool-profile build; prints the op tables and
# `pool [decode attention]` sections verbatim. VERDICTS ARE HUMAN: paste
# into the M4b.12 spec §Amendments and compute decode-wall shares, the
# menu guard C(max) vs C(1)/2, and P_W/P_A/P_D there, per the spec's
# pre-registered formulas (docs/runbooks/quiet-hw-verification.md).
# C(n) = kernel-max cycles / calls, from each sweep point's pool section.
# Usage: gate-attn-split.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-attn-split.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PROMPT_BYTES=256; MAXTOK=8; else PROMPT_BYTES=2048; MAXTOK=64; fi

smoke_header "gate-attn-split (M4b.12 attribution: dispatch-split profile + shard sweep)"
machine_block
echo

PROMPT="$(head -c "$PROMPT_BYTES" /dev/urandom | base64 | tr -d '\n')"
TBEST="$(phys_cores)"

echo "--- dispatch-split profile at --threads $TBEST ---"
cargo run --release -q -p inferno --features pool-profile -- run "$MODEL" \
  --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads "$TBEST" --profile \
  > "$OUT/attn-split-t$TBEST.txt" 2>&1
sed -n '/^profile \[/,$p' "$OUT/attn-split-t$TBEST.txt"
echo

echo "--- shard sweep (INFERNO_ATTN_SHARDS; pool sections only) ---"
for S in 1 2 4 7 "$TBEST"; do
  echo "--- INFERNO_ATTN_SHARDS=$S ---"
  INFERNO_ATTN_SHARDS="$S" cargo run --release -q -p inferno --features pool-profile -- run "$MODEL" \
    --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads "$TBEST" --profile \
    > "$OUT/attn-sweep-s$S.txt" 2>&1
  sed -n '/^pool \[decode attention\]/,$p' "$OUT/attn-sweep-s$S.txt"
  echo
done
```

Smoke it: `QHW_SMOKE=1 bash scripts/quiet-hw/gate-attn-split.sh models/qwen2.5-0.5b-instruct-q8_0.gguf`
Expected: op tables + a pool section, then five sweep sections with differing lane tables. If the `sed` extraction prints nothing, fix it against the actual section header before proceeding.

- [ ] **Step 4: `gate-attn-perturb.sh` — admissibility #2**

Create `scripts/quiet-hw/gate-attn-perturb.sh` (then `chmod +x`):

```bash
#!/usr/bin/env bash
# M4b.12 admissibility check #2 — instrument perturbation: shipping build
# vs pool-profile build with recording ON (INFERNO_POOL_PROF=1), the M4a
# bench protocol, interleaved rep pairs in one session. VERDICTS ARE
# HUMAN: paste into the M4b.12 spec §Amendments; if the within-session tg
# ratio moves more than 1%, the instrumentation is reworked before any
# attribution is trusted (spec §The dispatch-split instrument).
# Usage: gate-attn-perturb.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-attn-perturb.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PP=16; TG=8; REPS=1; else PP=512; TG=128; REPS=5; fi

smoke_header "gate-attn-perturb (M4b.12 admissibility: ship vs pool-profile-recording A/B)"
machine_block
echo

REPO=$(git rev-parse --show-toplevel)
cargo build --release -q -p inferno
cp "$REPO/target/release/inferno" "$OUT/inferno-ship"
cargo build --release -q -p inferno --features pool-profile
cp "$REPO/target/release/inferno" "$OUT/inferno-prof"

: > "$OUT/perturb-ship.jsonl"; : > "$OUT/perturb-prof.jsonl"
for r in $(seq "$REPS"); do
  echo "--- rep $r: ship ---"
  "$OUT/inferno-ship" bench "$MODEL" --pp "$PP" --tg "$TG" --reps 1 --threads 0 --json \
    | tee -a "$OUT/perturb-ship.jsonl"
  echo "--- rep $r: prof (recording on) ---"
  INFERNO_POOL_PROF=1 "$OUT/inferno-prof" bench "$MODEL" --pp "$PP" --tg "$TG" --reps 1 --threads 0 --json \
    | tee -a "$OUT/perturb-prof.jsonl"
done

echo
echo "inferno tg per interleaved rep (ship | prof-recording):"
paste \
  <(grep -o '"inferno_tg_tok_s": *[0-9.]*' "$OUT/perturb-ship.jsonl" | grep -o '[0-9.]*$') \
  <(grep -o '"inferno_tg_tok_s": *[0-9.]*' "$OUT/perturb-prof.jsonl" | grep -o '[0-9.]*$')
```

Smoke: `QHW_SMOKE=1 bash scripts/quiet-hw/gate-attn-perturb.sh models/qwen2.5-0.5b-instruct-q8_0.gguf`
Expected: one interleaved rep pair and a final two-column tg line. (If `bench --json`'s key differs from `inferno_tg_tok_s`, fix the grep against the actual output — the M4a protocol JSONs recorded in the M4b.11 spec §Amendments use that key.)

- [ ] **Step 5: `gate-attn-perf.sh` — perf-counter rider**

Create `scripts/quiet-hw/gate-attn-perf.sh` (then `chmod +x`):

```bash
#!/usr/bin/env bash
# M4b.12 rider — perf-counter capture on the SHIPPING build: topdown (or
# -d fallback) + scheduler events around a decode-dominant run. This is
# the worker-side view calling-thread self-measurement can't see, and the
# escalation evidence if the menu guard fires. Whole-process counters
# (prefill included) — the workload is shaped decode-heavy (short prompt,
# long generation); interpretation is controller work. VERDICTS ARE HUMAN.
# Exit: 0 completed, 3 SKIPPED (no perf), else failure.
# Usage: gate-attn-perf.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }
command -v perf >/dev/null || { echo "SKIPPED: perf not on PATH"; exit 3; }

MODEL="${1:?usage: gate-attn-perf.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then MAXTOK=8; else MAXTOK=256; fi

smoke_header "gate-attn-perf (M4b.12 rider: topdown + scheduler counters, shipping build)"
machine_block
echo

REPO=$(git rev-parse --show-toplevel)
cargo build --release -q -p inferno
BIN="$REPO/target/release/inferno"
PROMPT="$(head -c 256 /dev/urandom | base64 | tr -d '\n')"

if perf stat --topdown -- true >/dev/null 2>&1; then TD=(--topdown); else TD=(-d); fi
echo "--- perf stat ${TD[*]} ---"
perf stat "${TD[@]}" -o "$OUT/attn-perf-topdown.txt" -- \
  "$BIN" run "$MODEL" --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads 0 >/dev/null
cat "$OUT/attn-perf-topdown.txt"
echo
echo "--- perf stat scheduler events ---"
perf stat -e task-clock,context-switches,cpu-migrations -o "$OUT/attn-perf-sched.txt" -- \
  "$BIN" run "$MODEL" --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads 0 >/dev/null
cat "$OUT/attn-perf-sched.txt"
```

Smoke: `QHW_SMOKE=1 bash scripts/quiet-hw/gate-attn-perf.sh models/qwen2.5-0.5b-instruct-q8_0.gguf`
Expected: two perf tables, or `SKIPPED: perf not on PATH` (exit 3) on a box without perf.

- [ ] **Step 6: Wire into `verify.sh`**

In `scripts/quiet-hw/verify.sh`, after the `run_gate decode-attr` line:

```bash
run_gate attn-split      bash "$HERE/gate-attn-split.sh" "$MODEL"
run_gate attn-perturb    bash "$HERE/gate-attn-perturb.sh" "$MODEL"
run_gate attn-perf       bash "$HERE/gate-attn-perf.sh" "$MODEL"
```

The summary loop becomes:

```bash
  for g in prefill-scaling decode-cap bw-curve pf-dist bench-protocol decode-attr attn-split attn-perturb attn-perf intel-ab; do
```

The hard-fail loop becomes (attn-perf may legitimately SKIP, so it joins intel-ab's FAILED-only check):

```bash
for g in prefill-scaling decode-cap bw-curve pf-dist bench-protocol decode-attr attn-split attn-perturb; do
  [ "${status[$g]}" = PASS ] || exit 1
done
[ "${status[intel-ab]}" = FAILED ] && exit 1
[ "${status[attn-perf]}" = FAILED ] && exit 1
exit 0
```

- [ ] **Step 7: Runbook verdict rows**

In `docs/runbooks/quiet-hw-verification.md`, append to the verdict-destination table (after the `gate-decode-attr.out` row, matching its format):

```markdown
| `gate-attn-split.out` | [M4b.12 spec](../superpowers/specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md) §Amendments | blame table (publish/wake/kernel/drain + parked bits + H-alloc), sum identity (admissible 90–110%), C(n) sweep, menu guard, P_W/P_A/P_D arithmetic and verdicts |
| `gate-attn-perturb.out` | M4b.12 spec §Amendments | admissibility #2: within-session ship-vs-recording tg ratio (rework instrument if >1%) |
| `gate-attn-perf.out` | M4b.12 spec §Amendments | topdown + scheduler counters (worker-side view; escalation evidence if the menu guard fires) |
```

- [ ] **Step 8: AGENTS.md instrument note**

In `AGENTS.md`, in the decode-threading bullet, after the M4b.11 head-sharding sentence (the one ending "hspan tiling tests are the guard)."), append:

```markdown
  The pool's `pool-profile` cargo feature is the M4b.12 dispatch-split
  instrument (off in every shipping/bench build; quiet-hw gate scripts
  build with it), and `INFERNO_ATTN_SHARDS` is its probe-only shard-count
  override — neither is a tuning surface.
```

- [ ] **Step 9: Full smoke pass and commit**

Run: `bash scripts/quiet-hw/verify.sh models/qwen2.5-0.5b-instruct-q8_0.gguf --smoke`
Expected: summary table gains `attn-split`, `attn-perturb`, `attn-perf` rows, each PASS or (attn-perf only) SKIPPED; preflight UNFIT is fine under `--smoke`.

```bash
git add scripts/quiet-hw/ docs/runbooks/quiet-hw-verification.md AGENTS.md
git commit -m "quiet-hw: M4b.12 gates — attn-split/perturb/perf + invariant-TSC preflight probe"
```

---

### Task 6: Attribution sessions (manual, quiet hardware — one per machine)

A **manual protocol run**, not code. Follow `docs/runbooks/metal.md` (provisioning) and `docs/runbooks/quiet-hw-verification.md` (session discipline) exactly. PR and merge Tasks 1–5 first — sessions run against `main`.

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md` (§Amendments)

**Interfaces:**
- Consumes: Task 5's gates via `mise run metal` + `verify.sh`.
- Produces: per machine, the recorded amendments Task 7 consumes: dispatch-split blame table, C(n) sweep, perturbation A/B, perf capture, and both admissibility checks' raw numbers.

- [ ] **Step 1: Session on the 16c primary (`d2.c1.medium`)**

Provision via `mise run metal` per the runbook. On the box, inside the devenv shell: smoke pass first (`verify.sh <model> --smoke`), then the real pass (`verify.sh <model>`). Preflight UNFIT is a hard stop — reschedule, don't override.

- [ ] **Step 2: Record the 16c amendments**

Paste `gate-attn-split.out`, `gate-attn-perturb.out`, and `gate-attn-perf.out` verbatim into the M4b.12 spec §Amendments under a date-stamped "attribution session A" heading. State both admissibility results in the heading paragraph (sum identity %, perturbation tg ratio). Commit:

```bash
git add docs/superpowers/specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md
git commit -m "specs: M4b.12 attribution session A (d2.c1.medium, 16c) — dispatch split + sweep + admissibility"
```

- [ ] **Step 3: Session on the 8c check (`s2.c2.medium`) — same protocol**

Repeat Steps 1–2; commit as session B. (PHX stock permitting — the metal runbook's location fallback applies.)

---

### Task 7: Gate verdicts — the pre-registered arithmetic

Controller work: compute the gates from the recorded sessions, exactly as the spec pre-registers them. **Do not look at the lever tasks while doing this; the formulas were fixed before the data.**

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md` (§Amendments)

**Interfaces:**
- Consumes: Task 6's recorded amendments.
- Produces: the menu-guard verdict and the P_W / P_A / P_D verdicts that decide whether Tasks 8–10 run.

- [ ] **Step 1: Admissibility first**

Per machine: sum identity = `instr_total / op-profiler decode attention cycles` must be within 90–110%; perturbation tg ratio must be within 1%. **If either fails on either machine, stop — fix the instrument, re-run Task 6. No gate is evaluated on inadmissible data.**

- [ ] **Step 2: Menu guard**

From the sweep sections: `C(n) = kernel_max_cyc / calls` at each shard count. If `C(max shards) > C(1) / 2` on **both** machines, all three gates STOP: record the finding with the perf capture as evidence — this is the flash-decoding escalation record (M4b.11 §Scope). Skip to Task 11.

- [ ] **Step 3: Per-lever projections and thresholds**

Decode-wall shares from the best-t dispatch-split profile (decode total cycles = the op table's total):

```
P_W = wake_parked_cyc / decode_total_cyc
P_A = alloc_max_cyc  / decode_total_cyc
P_D = publish_cyc    / decode_total_cyc
```

Thresholds per lever (M4b.6 STOP gate, verbatim): **≥3% on both machines → authorized; <3% on both → STOP; split → judgment call, argument recorded.**

- [ ] **Step 4: Record the verdict amendment and commit**

The amendment shows: both admissibility results, the C(n) table and menu-guard arithmetic, each P value per machine, and each gate's verdict with the threshold applied.

```bash
git add docs/superpowers/specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md
git commit -m "specs: M4b.12 gate verdicts — menu guard + P_W/P_A/P_D arithmetic and lever authorization"
```

**⛔ GATE: Tasks 8, 9, 10 run only if their lever's gate is authorized, in the order A → D → W, one at a time (each lever's data point lands before the next lever's implementation starts). All STOP → straight to Task 11.**

---

### Task 8 (GATED on P_A): Lever A — per-lane scratch reuse

**Files:**
- Modify: `crates/inferno-pool/src/pool.rs`
- Modify: `docs/superpowers/specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md` (§Amendments, data point)

**Interfaces:**
- Consumes: `run_attn_heads_span` (Tasks 2's bracketed version).
- Produces: allocation-free steady-state decode attention; the H-alloc bracket now measures the resize check (~0), which the data point's post-lever profile confirms.

- [ ] **Step 1: Write the failing test**

Append to the `tests` module in `crates/inferno-pool/src/pool.rs`:

```rust
    #[test]
    fn attn_heads_scratch_reuse_is_bit_invisible_across_pos_sequence() {
        // Lever A: grow-only per-lane scratch. A long → short → long pos
        // sequence exercises reuse with stale bytes beyond pos+1 present;
        // outputs must match the fresh-scratch expectation bit-for-bit.
        let pool = Pool::new(4);
        for pos in [100, 0, 100, 7, 100, 0] {
            assert_eq!(
                attn_heads_dispatch(&pool, pos),
                attn_heads_expected(pos),
                "pos={pos}"
            );
        }
    }
```

(This passes even pre-lever — it pins the behavior the lever must preserve. The lever's own signal is the H-alloc bracket going to ~0, checked in the data point.)

- [ ] **Step 2: Run the test — it passes pre-lever; that is the point**

Run: `cargo test -p inferno-pool attn_heads_scratch_reuse`
Expected: PASS (baseline pinned).

- [ ] **Step 3: Implement the scratch**

In `crates/inferno-pool/src/pool.rs`, add near `run_attn_heads_span`:

```rust
thread_local! {
    /// Lever A (M4b.12): grow-only per-lane decode-attention scratch. The
    /// hspan kernel writes `scores[t]` for every `t <= pos` before any
    /// read of it, so stale contents from a longer earlier call are
    /// unreachable — reuse is bit-invisible. One buffer per thread covers
    /// every lane (dispatcher, workers, and the pool-less serial
    /// fallback), since a lane runs one span at a time.
    static ATTN_SCRATCH: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}
```

and rewrite `run_attn_heads_span`'s body to use it (kernel call args unchanged):

```rust
pub(crate) unsafe fn run_attn_heads_span(j: &AttnHeadsJob, start: usize, end: usize) {
    ATTN_SCRATCH.with(|cell| {
        let mut scores = cell.borrow_mut();
        #[cfg(feature = "pool-profile")]
        let a0 = crate::prof::now();
        if scores.len() < j.pos + 1 {
            scores.resize(j.pos + 1, 0.0);
        }
        #[cfg(feature = "pool-profile")]
        crate::prof::ALLOC_CYC.with(|c| c.set(crate::prof::now().saturating_sub(a0)));
        // SAFETY: forwarding the caller's contract for the head span;
        // scores covers pos + 1 entries.
        unsafe {
            (j.kernel)(
                j.out,
                j.q,
                j.kv,
                scores.as_mut_ptr(),
                j.kv_base,
                j.v_off,
                j.pos,
                j.kv_dim,
                j.n_heads,
                j.n_kv_heads,
                j.head_dim,
                start,
                end,
            );
        }
    })
}
```

Also update the function's doc comment: replace the "same Vec-per-lane reasoning" sentence with a pointer to the `ATTN_SCRATCH` rationale. `run_attn_span` (prefill) keeps its `vec!` — prefill is out of scope.

- [ ] **Step 4: Full verification**

Run: `mise run test && mise run lint`
Expected: pass — including the pos-sequence test, all existing pool tests, and the differential suites (`inferno-codegen --test differential`, `inferno-core --test artifact` run inside `mise run test`) with zero tolerance change.

- [ ] **Step 5: Commit, PR, merge**

```bash
git add crates/inferno-pool/src/pool.rs
git commit -m "M4b.12 Lever A: grow-only per-lane decode-attention scratch (gated: P_A authorized)"
```

- [ ] **Step 6: Quiet-hw data point (manual, both machines)**

Per machine, per the metal runbook: within one session, run the M4a protocol on the parent commit and the lever commit, interleaved (the M4b.11 Task 7 pattern):

```bash
cargo run --release -q -p inferno -- bench <model> --pp 512 --tg 128 --reps 5 --threads 0 --json
cargo run --release -q -p inferno -- bench <model> --pp 16 --tg 32 --reps 5 --threads 0 --json
```

Also capture one post-lever `gate-attn-split.sh` profile — the H-alloc column must have collapsed. Paste verbatim into the M4b.12 spec §Amendments with within-session tg ratios vs P_A's projection. **Revert rule:** tg regression on either machine → revert this lever's commit, record the finding. Commit the amendment:

```bash
git add docs/superpowers/specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md
git commit -m "specs: M4b.12 Lever A data point (both machines)"
```

---

### Task 9 (GATED on P_D): Lever D — publish slimming

Reuses the published `shards` vector when the dispatch geometry is unchanged (decode calls this 24×/token with identical geometry), eliminating the per-call `Vec` build+drop from the publish path. **Ordering audit included; the epoch/remaining SAFETY argument must survive verbatim — the default outcome of the audit is "no ordering change".**

**Files:**
- Modify: `crates/inferno-pool/src/pool.rs`
- Modify: `docs/superpowers/specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md` (§Amendments, data point)

**Interfaces:**
- Consumes: Task 4's `par_attention_heads_at` structure.
- Produces: an allocation-free publish for repeated decode-attention geometry.

- [ ] **Step 1: Write the failing invariant test**

The reuse condition relies on one invariant: the align-1 shard-table length is exactly `min(threads, rows)`, so `(n_heads, shards.len())` fully determines the table. Pin it. Append to the `tests` module in `crates/inferno-pool/src/pool.rs`:

```rust
    #[test]
    fn attn_shard_table_len_is_min_threads_heads() {
        // Lever D's reuse key is (n_heads, shards.len()); sound only if the
        // align-1 table is a pure function of those two. len == min(t, h)
        // makes the key complete.
        for h in 1..=64usize {
            for t in 1..=32usize {
                let tbl = shard_table_aligned(h, t, 1);
                assert_eq!(tbl.len(), t.min(h), "h={h} t={t}");
                assert_eq!(tbl.first().map(|s| s.0), Some(0));
                assert_eq!(tbl.last().map(|s| s.1), Some(h));
            }
        }
    }

    #[test]
    fn attn_heads_geometry_changes_reshard_correctly() {
        // Alternate lane counts against one pool: every dispatch must hit
        // either the reuse path or a fresh table and stay bit-identical.
        let pool = Pool::new(8);
        for active in [8, 8, 2, 8, 1, 3, 3, 8] {
            pool.set_active_threads(active);
            assert_eq!(
                attn_heads_dispatch(&pool, 9),
                attn_heads_expected(9),
                "active={active}"
            );
        }
        pool.set_active_threads(8);
    }
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p inferno-pool attn_shard_table_len attn_heads_geometry`
Expected: both PASS pre-lever (they pin invariants the lever relies on; `attn_heads_geometry_changes_reshard_correctly` will catch a broken reuse key after Step 3).

- [ ] **Step 3: Implement the reuse**

In `par_attention_heads_at`, replace everything from `let shards = shard_table_aligned(...)` through the `unsafe { *self.shared.job.get() = Job { ... }; }` block with:

```rust
        let lanes = active.max(1).min(n_heads);
        if lanes == 1 {
            // SAFETY: caller contract covers the full head range.
            unsafe { run_attn_heads_span(job, 0, n_heads) };
            #[cfg(feature = "pool-profile")]
            if rec {
                self.shared.prof.record_single(t0, crate::prof::now());
            }
            return;
        }
        // SAFETY (job write): the previous dispatch ended with
        // `remaining == 0`, and shardless workers never read `job`
        // (packed-epoch protocol) — no reader exists here. Lever D
        // (M4b.12): the shards vector is rebuilt only when the dispatch
        // geometry changed — the align-1 table is a pure function of
        // (n_heads, lanes) and len == min(lanes, n_heads) (pinned by
        // attn_shard_table_len_is_min_threads_heads), so an AttnHeads job
        // with the same n_heads and the same table length IS the same
        // table. Workers read `shards` only under a new epoch, and its
        // contents are identical either way.
        let (n_worker, s0, e0) = unsafe {
            let slot = &mut *self.shared.job.get();
            let reuse = matches!(slot.kind, Some(JobKind::AttnHeads(prev)) if prev.n_heads == n_heads)
                && slot.shards.len() == lanes;
            if !reuse {
                slot.shards = shard_table_aligned(n_heads, lanes, 1);
            }
            slot.kind = Some(JobKind::AttnHeads(*job));
            slot.y = std::ptr::null_mut();
            slot.xq = std::ptr::null();
            slot.w = std::ptr::null();
            slot.k = 0;
            let (s0, e0) = slot.shards[0];
            (slot.shards.len() - 1, s0, e0)
        };
```

(the `remaining.store`, dispatch-tsc, epoch, unpark, own-span, drain, and record_call code below is unchanged; the old `let kind = ...` line is gone — `slot.kind` is written directly. Passing `lanes` instead of `active` to `shard_table_aligned` is value-identical: the table clamps to `min(threads, strips)` internally.)

- [ ] **Step 4: The ordering audit (documentation step, default = no change)**

Re-read the publish sequence against the epoch/remaining SAFETY comments (`PACKED_SHARD_BITS` doc, the job-write SAFETY, the unpark lost-wakeup argument at `par_gemv`). Record the audit's outcome as a code comment only if something was changed; the pre-registered default is **no ordering change** — every SeqCst in the publish sequence is load-bearing for the lost-wakeup argument. Do not trim orderings unless you can restate the full argument with the trim in place; if you do, the restated argument replaces the old comment in the same location.

- [ ] **Step 5: Full verification**

Run: `mise run test && mise run lint`
Expected: pass — including `stress_repeated_dispatches` (GEMV path unaffected), both new tests, both differential suites, zero tolerance change. Also run the feature flavor explicitly: `cargo test -p inferno-pool --features pool-profile` (the prof hooks sit inside the edited function).

- [ ] **Step 6: Commit, PR, merge; data point**

```bash
git add crates/inferno-pool/src/pool.rs
git commit -m "M4b.12 Lever D: reuse the published attention shard table across unchanged geometry (gated: P_D authorized)"
```

Then the same manual data-point protocol as Task 8 Step 6 (parent = post-Lever-A commit; post-lever `gate-attn-split.sh` capture — the publish bucket must have shrunk), same revert rule, amendment committed as:

```bash
git commit -m "specs: M4b.12 Lever D data point (both machines)"
```

---

### Task 10 (GATED on P_W): Lever W — decode wait-strategy

Workers that just ran a decode-kind shard (`Gemv` / `AttnHeads`) extend their spin window before parking. One new named constant; park/unpark protocol untouched.

**Files:**
- Modify: `crates/inferno-pool/src/pool.rs`
- Modify: `docs/superpowers/specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md` (§Amendments, data point)

**Interfaces:**
- Consumes: `worker_loop`, `SPIN_ITERS`.
- Produces: fewer park-eligible waits inside decode (the parked-bit column in the post-lever profile is the direct check).

- [ ] **Step 1: Write the failing stress test**

Append to the `tests` module in `crates/inferno-pool/src/pool.rs`:

```rust
    /// Lever W liveness: decode- and prefill-kind jobs across window
    /// boundaries, with sleeps past the base spin window — the park/unpark
    /// handshake must never lose a wakeup regardless of which window a
    /// worker was in.
    #[test]
    fn decode_window_boundary_stress() {
        let pool = Pool::new(4);
        for i in 0..200usize {
            let rows = 64 + (i * 37) % 512;
            assert_eq!(dispatch(&pool, rows, i % 13), expected(rows, i % 13)); // Gemv: decode-kind
            assert_eq!(attn_dispatch(&pool, 8, 0), attn_expected(8, 0)); // Attention: prefill-kind
            assert_eq!(attn_heads_dispatch(&pool, 9), attn_heads_expected(9)); // AttnHeads: decode-kind
            if i % 50 == 0 {
                // Past the base window; with the lever, past the extended
                // window too on the final iteration's sleep.
                std::thread::sleep(std::time::Duration::from_millis(60));
            }
        }
    }
```

- [ ] **Step 2: Run it — passes pre-lever (baseline pinned); keep it**

Run: `cargo test -p inferno-pool decode_window_boundary_stress`
Expected: PASS.

- [ ] **Step 3: Implement the window**

In `crates/inferno-pool/src/pool.rs`, after `SPIN_ITERS`:

```rust
/// Lever W (M4b.12): a worker that just ran a decode-kind shard
/// (`Gemv`/`AttnHeads`) waits `SPIN_ITERS * DECODE_SPIN_MULT` before
/// becoming park-eligible — decode issues hundreds of dispatches per token
/// with serial gaps (sampling, layer tails) that can exceed the base
/// window, and a parked lane pays scheduler wake latency on the next
/// dispatch (the M4b.12 wake bucket). Prefill kinds reset to the base
/// window; an idle host pays at most one extended window after its last
/// decode dispatch, then parks exactly as before. Named, like SPIN_ITERS,
/// so an embedding host that objects can tune one constant.
const DECODE_SPIN_MULT: u32 = 20;
```

In `worker_loop`: before the outer `loop {`, add `let mut decode_hot = false;`. Change the wait loop's spin bound: at the top of the outer loop, after `let mut spins = 0u32;` (and the Task-2 `spun_out` line), add:

```rust
        let window = if decode_hot {
            SPIN_ITERS.saturating_mul(DECODE_SPIN_MULT)
        } else {
            SPIN_ITERS
        };
```

and in the wait loop replace `if spins < SPIN_ITERS {` with `if spins < window {`. (The park branch is untouched: same `parked` handshake, same epoch re-check, same `spins = 0` reset — the lost-wakeup argument depends on the handshake, not the iteration count. Restate this in the park branch's comment.)

After the job read (the `let (kind, ...) = unsafe { ... };` block), before the Task-2 recording block, add:

```rust
        decode_hot = matches!(kind, JobKind::Gemv { .. } | JobKind::AttnHeads(_));
```

(Shardless epochs `continue` above the job read and keep the previous `decode_hot` — they never read `job`, per the packed-epoch protocol.)

- [ ] **Step 4: Full verification**

Run: `mise run test && mise run lint && cargo test -p inferno-pool --features pool-profile`
Expected: pass — including `workers_park_and_wake_correctly` (100 ms sleep is past even the extended ~1 ms window), the boundary stress, both differential suites, zero tolerance change.

- [ ] **Step 5: Commit, PR, merge; data point**

```bash
git add crates/inferno-pool/src/pool.rs
git commit -m "M4b.12 Lever W: extended spin window after decode-kind shards (gated: P_W authorized)"
```

Then the same manual data-point protocol as Task 8 Step 6 (parent = the previous landed commit; the post-lever `gate-attn-split.sh` capture must show `parked_calls` collapsing and the wake bucket shrinking), same revert rule, amendment committed as:

```bash
git commit -m "specs: M4b.12 Lever W data point (both machines)"
```

---

### Task 11: Closing re-bench + milestone closure

- [ ] **Step 1: Closing protocol run (manual, both machines)**

If any lever landed: run the M4a protocol (`gate-bench-protocol.sh` via `verify.sh`, or the manual `inferno bench` commands) on both machines against the final commit. Record tg vs llama.cpp best-of as **v1 context, never the gate** (spec §Exit criteria 4), next to the M4b.11 closing baseline (0.96x / 0.86x). If all gates STOP'd, this step instead re-records nothing new — the closing amendment states the milestone closed as a diagnostic.

- [ ] **Step 2: Record and close**

Bench outputs verbatim into the M4a spec §Amendments (protocol home), cross-referenced from the M4b.12 spec §Amendments in a closing-verdict section that walks the four exit criteria: (1) blame table + sweep + perf capture recorded with admissibility passing; (2) gate verdicts recorded once with arithmetic; (3) every authorized lever's data point or revert, every STOP; (4) the closing tg context. If the menu guard fired, the closing verdict names the recorded finding as the flash-decoding escalation record.

```bash
git add docs/superpowers/specs/
git commit -m "specs: M4b.12 closing — exit-criteria walk and verdict"
```

---

## Self-Review

- **Spec coverage:** instrument (four buckets, parked bit, H-alloc bracket, histogram, per-lane) → Tasks 1–2; CLI section + sum identity → Task 3; `INFERNO_ATTN_SHARDS` → Task 4; preflight TSC assert, sweep, perturbation A/B, perf rider, runbook, AGENTS.md → Task 5; attribution protocol → Task 6; admissibility-first + menu guard + P_W/P_A/P_D thresholds → Task 7; Lever A → Task 8; Lever D incl. the ordering-audit-default-no-change → Task 9; Lever W → Task 10; exit criteria 1–4 → Tasks 6, 7, 8–10, 11. Out-of-scope items (flash-decoding, F16 KV, shard-count cap, NUMA, prefill, CI gates) appear only as constraints. ✔
- **Placeholder scan:** every code step carries complete code; the two "do not retype" instructions (Task 4 Step 3, Task 9 Step 3) move existing verbatim code with the exact seam stated. Manual-protocol tasks (6, 8/9/10 data points, 11) reference the runbooks that own those procedures, per the M4b.11 precedent. ✔
- **Type consistency:** `PoolProfSnapshot` field set is identical in Task 1 (definition), Task 2 (assertions), and Task 3 (test literal + `render_pool`); `record_call(t0, t2, t3, t4, n_worker)` matches its call site; `par_attention_heads_at(&self, job, active)` is defined in Task 4 and edited (not re-signed) in Task 9; `attn_shards_override`/`parse_attn_shards` names match their tests; gate script names match the `verify.sh` wiring and runbook rows. ✔
