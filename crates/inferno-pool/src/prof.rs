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

#[cfg(all(test, feature = "pool-profile"))]
mod tests {
    use super::*;

    #[test]
    fn snapshot_instr_total_is_bucket_sum() {
        let st = ProfState::new(4);
        st.publish_cyc
            .fetch_add(10, std::sync::atomic::Ordering::Relaxed);
        st.kernel0_cyc
            .fetch_add(20, std::sync::atomic::Ordering::Relaxed);
        st.drain_cyc
            .fetch_add(30, std::sync::atomic::Ordering::Relaxed);
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
