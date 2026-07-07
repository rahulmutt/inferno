//! Concurrent dispatch race over the safe API: two threads both calling
//! `inferno_par_gemv` against the SAME initialized global pool at the same
//! time. `Pool::par_gemv`'s protocol is single-dispatcher (its `# Safety`
//! forbids overlapping calls), but two `CompiledBackend`s on two user
//! threads can reach `inferno_par_gemv` concurrently through nothing but
//! safe code (`Engine::compiled_backend()` + `Backend::forward()`). This is
//! the M4b.1 merge-gate fix: a static dispatch guard in `inferno_par_gemv`
//! must make this data-race-free — the loser of the guard's `compare_exchange`
//! runs a fully serial kernel call instead of touching the pool, so it can
//! never corrupt the shared job/epoch/remaining protocol.
//!
//! Own test binary (own process, under nextest): the global pool is
//! process-global state, so this can't share a binary with any other test
//! that initializes it (see `tests/global.rs`'s own note).
//!
//! This is a REAL race, not a sequenced simulation: both threads start from
//! a `Barrier` at (as close to) the same instant, and each runs hundreds of
//! dispatches back to back — enough for the two threads' CAS attempts to
//! actually contend for the guard, not just interleave without overlap.

use std::sync::{Arc, Barrier};
use std::thread;

use inferno_formats::{DType, quant};
use inferno_kernels::{AlignedBuf, reference_kernels};
use inferno_pool::GemvFn;

/// Deterministic pseudo-random f32s in [-1, 1) (same generator as
/// `tests/par_rig.rs` and the kernels' own rig).
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

/// Pack weights + quantize the activation for f32 via the scalar KernelSet.
fn prep(rows: usize, k: usize, seed: u64) -> (AlignedBuf, Vec<u8>) {
    let set = reference_kernels(&DType::F32).expect("scalar set always available");
    let wvals = pseudo(seed, rows * k);
    let wbytes = quant::pack(&DType::F32, &wvals).unwrap();
    let w = set.pack(&wbytes, rows, k).unwrap();
    let x = pseudo(seed ^ 0x9e3779b97f4a7c15, k);
    let xq = set.quantize_row(&x).unwrap();
    (w, xq)
}

const ROWS: usize = 1003; // deliberately not a multiple of 8
const K: usize = 33;
const ITERS: usize = 300;

#[test]
fn concurrent_par_gemv_dispatches_stay_correct() {
    let kernel: GemvFn = inferno_kernels::inferno_gemv_f32_rs8_scalar;
    let (w, xq) = prep(ROWS, K, 0xC0FFEE_u64);

    // Reference: one direct serial kernel call over the full range — the
    // exact bits every dispatch (pooled OR guard-loser-serial) must match.
    let mut want = vec![f32::NAN; ROWS];
    // SAFETY: w/xq built by prep() for exactly (ROWS, K); want has ROWS f32s.
    unsafe { kernel(want.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), K, 0, ROWS) };

    // A real multi-lane pool: pool dispatches genuinely parallelize
    // internally while the two threads below race for the dispatch guard.
    inferno_pool::init_global(4).expect("first (only) init_global in this process");

    let w = Arc::new(w);
    let xq = Arc::new(xq);
    let want = Arc::new(want);
    let barrier = Arc::new(Barrier::new(2));

    let spawn_worker =
        |w: Arc<AlignedBuf>, xq: Arc<Vec<u8>>, want: Arc<Vec<f32>>, barrier: Arc<Barrier>| {
            thread::spawn(move || {
                barrier.wait(); // both threads start hammering at the same instant
                for iter in 0..ITERS {
                    // Distinct output buffer every call (and between threads —
                    // each is a fresh, thread-local allocation).
                    let mut y = vec![f32::NAN; ROWS];
                    // SAFETY: buffers sized/built per `kernel`'s contract for
                    // (ROWS, K); `y` is live and exclusively owned by this call.
                    unsafe {
                        inferno_pool::inferno_par_gemv(
                            kernel,
                            y.as_mut_ptr(),
                            xq.as_ptr(),
                            w.as_ptr(),
                            K,
                            ROWS,
                        )
                    };
                    for (i, (g, s)) in y.iter().zip(want.iter()).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            s.to_bits(),
                            "iter {iter} row {i}: {g} != {s} (racing dispatch corrupted output)"
                        );
                    }
                }
            })
        };

    let t1 = spawn_worker(
        Arc::clone(&w),
        Arc::clone(&xq),
        Arc::clone(&want),
        Arc::clone(&barrier),
    );
    let t2 = spawn_worker(w, xq, want, barrier);

    t1.join().expect("thread 1 must not panic");
    t2.join().expect("thread 2 must not panic");
}
