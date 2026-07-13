//! Bandwidth-saturation probe (M4b.10). Times `par_gemv` at a range of lane
//! counts and derives the lane count at which aggregate streaming bandwidth
//! saturates — the physically motivated decode cap
//! (`total_DRAM_bandwidth / per_core_streaming_bandwidth`).
//!
//! Generic over the kernel on purpose: the caller supplies the `GemvFn` and
//! the packed buffers, so this crate gains no dependency on `inferno-kernels`.

use crate::pool::{GemvFn, Pool};
use std::time::Instant;

/// The smallest lane count in `curve` reaching `frac` of the curve's peak
/// rate — the saturation knee. An empty curve knees at 1 lane.
///
/// Deliberately reads the *first* lane count at or above the threshold, not
/// the argmax: past saturation the curve is flat-to-noisy, and the cheapest
/// lane count on the plateau is the one we want.
pub fn knee_at_fraction(curve: &[(usize, f64)], frac: f64) -> usize {
    let peak = curve.iter().map(|&(_, r)| r).fold(f64::MIN, f64::max);
    if curve.is_empty() {
        return 1;
    }
    let target = peak * frac;
    curve
        .iter()
        .find(|&&(_, r)| r >= target)
        .map(|&(lanes, _)| lanes)
        .unwrap_or_else(|| curve.last().map(|&(l, _)| l).unwrap_or(1))
}

/// Restores the pool's decode cap on drop, whether the sweep in
/// `bandwidth_curve` completes normally or unwinds (e.g. a panic from a
/// caller-supplied kernel, or anywhere else mid-sweep). Private to this
/// module — `bandwidth_curve` is the only thing that constructs one.
struct CapGuard<'a> {
    pool: &'a Pool,
    restore: usize,
}

impl Drop for CapGuard<'_> {
    fn drop(&mut self) {
        self.pool.set_decode_threads(self.restore);
    }
}

/// Time `reps` full-range `par_gemv` dispatches at each lane count in
/// `lanes`, returning `(lanes, GB/s)` per entry. `stream_bytes` is the number
/// of bytes the kernel streams per dispatch (the packed weight image), which
/// is what makes the rate a *bandwidth* rather than a throughput.
///
/// Takes the median of `reps` timings per lane count, so one descheduled
/// iteration cannot move the curve. Saves and restores the pool's decode cap
/// around the sweep — the cap is restored even if a dispatch unwinds, since
/// this probe is meant to run against the process-global pool and must never
/// strand it at a mid-sweep lane count.
///
/// `reps == 0` returns an empty curve without dispatching anything: there is
/// no timing data to take a median of, so "zero reps" and "no lane counts
/// swept" collapse to the same empty result. This composes with
/// [`knee_at_fraction`], which already treats an empty curve as knee-at-1.
///
/// # Safety
/// Same contract as [`Pool::par_gemv`] for `(kernel, y, xq, w, k, rows)`:
/// `y` valid for `rows` f32 writes, `xq`/`w` valid packed buffers built for
/// this exact `k` and `rows`, and `kernel` a valid GEMV-ABI pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe fn bandwidth_curve(
    pool: &Pool,
    lanes: &[usize],
    reps: usize,
    stream_bytes: usize,
    kernel: GemvFn,
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    rows: usize,
) -> Vec<(usize, f64)> {
    let _guard = CapGuard {
        pool,
        restore: pool.decode_threads(),
    };

    debug_assert!(
        reps > 0,
        "bandwidth_curve called with reps == 0 — likely a caller bug (a \
         mis-specified sweep silently collapses to knee-at-1, i.e. maximum \
         decode throttling); release behavior (empty curve) is unchanged"
    );
    if reps == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(lanes.len());

    for &n in lanes {
        pool.set_decode_threads(n);

        // Warm the lanes and the weight image into whatever caches will hold
        // it, so the first timed rep is not paying for a cold pool.
        // SAFETY: forwarding the caller's contract unchanged.
        unsafe { pool.par_gemv(kernel, y, xq, w, k, rows) };

        let mut secs: Vec<f64> = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            // SAFETY: forwarding the caller's contract unchanged.
            unsafe { pool.par_gemv(kernel, y, xq, w, k, rows) };
            secs.push(t0.elapsed().as_secs_f64());
        }
        secs.sort_by(f64::total_cmp);
        let med = secs[secs.len() / 2].max(f64::EPSILON);
        out.push((n, stream_bytes as f64 / med / 1e9));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pool;

    #[test]
    fn knee_is_the_first_lane_count_reaching_the_fraction_of_peak() {
        // Saturates at 2 lanes: peak 21.0, 95% of peak = 19.95, and 2 lanes
        // already delivers 20.0.
        let curve = [(1, 10.0), (2, 20.0), (4, 21.0), (8, 21.0)];
        assert_eq!(knee_at_fraction(&curve, 0.95), 2);
    }

    #[test]
    fn a_curve_that_never_saturates_knees_at_the_top() {
        let curve = [(1, 10.0), (2, 20.0), (4, 40.0)];
        assert_eq!(knee_at_fraction(&curve, 0.95), 4);
    }

    #[test]
    fn a_non_monotonic_tail_does_not_move_the_knee_below_the_peak_fraction() {
        // 8 lanes regresses; the knee is still where 95% of peak is first hit.
        let curve = [(1, 10.0), (2, 20.0), (4, 21.0), (8, 18.0)];
        assert_eq!(knee_at_fraction(&curve, 0.95), 2);
    }

    #[test]
    fn degenerate_curves_knee_at_one_lane() {
        assert_eq!(knee_at_fraction(&[], 0.95), 1);
        assert_eq!(knee_at_fraction(&[(3, 12.0)], 0.95), 3);
    }

    /// A stub kernel: writes each row so the dispatcher's work is real but
    /// trivially fast. The GB/s values are meaningless here — this test is
    /// about the curve's *shape contract*, not its numbers.
    unsafe extern "C" fn stub_gemv(
        y: *mut f32,
        _xq: *const u8,
        _w: *const u8,
        _k: usize,
        row_start: usize,
        row_end: usize,
    ) {
        for r in row_start..row_end {
            // SAFETY: the dispatcher only ever passes rows within `y`'s length.
            unsafe { *y.add(r) = r as f32 };
        }
    }

    #[test]
    fn bandwidth_curve_returns_one_entry_per_lane_and_restores_the_cap() {
        let pool = Pool::new(4);
        pool.set_decode_threads(3);
        let mut y = vec![0f32; 64];
        let xq = [0u8; 8];
        let w = [0u8; 8];
        let lanes = [1usize, 2, 4];

        // SAFETY: stub_gemv only writes y[row_start..row_end]; rows == y.len().
        let curve = unsafe {
            bandwidth_curve(
                &pool,
                &lanes,
                2,
                1 << 20,
                stub_gemv,
                y.as_mut_ptr(),
                xq.as_ptr(),
                w.as_ptr(),
                8,
                64,
            )
        };

        assert_eq!(curve.len(), 3);
        assert_eq!(
            curve.iter().map(|&(l, _)| l).collect::<Vec<_>>(),
            vec![1, 2, 4]
        );
        assert!(
            curve.iter().all(|&(_, gbps)| gbps > 0.0),
            "every lane count must record a positive rate: {curve:?}"
        );
        assert_eq!(
            pool.decode_threads(),
            3,
            "the probe must restore the caller's decode cap"
        );
    }

    // `reps == 0` is a caller-bug shape (Finding 2, M4b.10 final review): a
    // mis-specified sweep would otherwise silently collapse to knee-at-1,
    // i.e. maximum decode throttling, with no signal anything went wrong.
    // The `debug_assert!` beside the early return makes that loud in
    // dev/test builds; the release contract (empty curve, cap untouched) is
    // deliberately unchanged. These two variants cover exactly those two
    // profiles — only one of them compiles into any given test run, so the
    // total test count is unaffected by which profile is active.

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "reps == 0")]
    fn reps_zero_trips_the_debug_assert_in_dev_and_test_builds() {
        let pool = Pool::new(4);
        let mut y = vec![0f32; 64];
        let xq = [0u8; 8];
        let w = [0u8; 8];
        let lanes = [1usize, 2, 4];

        // SAFETY: the debug_assert fires before any dispatch; the buffers
        // just need to satisfy the (unused) contract shape.
        let _ = unsafe {
            bandwidth_curve(
                &pool,
                &lanes,
                0,
                1 << 20,
                stub_gemv,
                y.as_mut_ptr(),
                xq.as_ptr(),
                w.as_ptr(),
                8,
                64,
            )
        };
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn reps_zero_returns_an_empty_curve_in_release_builds() {
        let pool = Pool::new(4);
        pool.set_decode_threads(3);
        let mut y = vec![0f32; 64];
        let xq = [0u8; 8];
        let w = [0u8; 8];
        let lanes = [1usize, 2, 4];

        // SAFETY: reps == 0 means the sweep returns before any dispatch;
        // the buffers just need to satisfy the (unused) contract shape.
        let curve = unsafe {
            bandwidth_curve(
                &pool,
                &lanes,
                0,
                1 << 20,
                stub_gemv,
                y.as_mut_ptr(),
                xq.as_ptr(),
                w.as_ptr(),
                8,
                64,
            )
        };

        assert!(
            curve.is_empty(),
            "reps == 0 must yield an empty curve, got {curve:?}"
        );
        assert_eq!(
            pool.decode_threads(),
            3,
            "reps == 0 must still leave the caller's decode cap untouched"
        );
    }

    #[test]
    fn the_decode_cap_is_restored_after_a_mid_sweep_unwind() {
        let pool = Pool::new(4);
        pool.set_decode_threads(3);

        // `GemvFn` is `unsafe extern "C" fn(...)`, and `extern "C"` is a
        // non-unwinding ABI in Rust: a panic thrown from inside a
        // caller-supplied kernel aborts the process instead of unwinding
        // (verified empirically — swapping in a panicking `extern "C"`
        // kernel here takes down the whole test binary rather than being
        // caught by `catch_unwind`, since Rust inserts an abort at the FFI
        // boundary per RFC 2945). So the only *reachable* mid-sweep unwind
        // is from Rust-side code in the loop body — exactly the shape of
        // the finding-1 bug (`secs[secs.len() / 2]` on an empty `Vec`)
        // before it was fixed. This test exercises the real mechanism
        // directly: open a `CapGuard` exactly as `bandwidth_curve` does,
        // mutate the cap exactly as the loop does for its first lane count,
        // then panic from plain Rust while the guard is alive.
        //
        // `Pool` holds an `UnsafeCell` behind its `Arc<Shared>`, so `&Pool`
        // is not `RefUnwindSafe`; `AssertUnwindSafe` is sound here because
        // the panic happens without calling into any pool method that
        // mutates through that cell, so no partially-mutated invariant can
        // leak across the catch.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = CapGuard {
                pool: &pool,
                restore: pool.decode_threads(),
            };
            pool.set_decode_threads(1);
            panic!("forced panic to exercise CapGuard's unwind cleanup");
        }));

        assert!(
            result.is_err(),
            "the forced panic must propagate through catch_unwind"
        );
        assert_eq!(
            pool.decode_threads(),
            3,
            "the probe must restore the caller's decode cap even after an unwind"
        );
    }
}
