//! Global init semantics + the extern entry over an initialized pool.
//! `init_dispatch_and_mismatch_semantics` chains steps that share the
//! process-global OnceLock, so their order within that fn must stay fixed
//! regardless of test-runner parallelism. Other tests here init the global
//! pool with the SAME count (4) so they compose under `cargo nextest`'s
//! process-per-test isolation (and remain idempotent under plain
//! `cargo test`).

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
    assert!(
        inferno_pool::init_global(4).is_ok(),
        "same count: idempotent"
    );
    assert_eq!(
        inferno_pool::init_global(2),
        Err(PoolError::AlreadyInitialized {
            current: 4,
            requested: 2
        })
    );

    let run = || {
        let mut y = vec![f32::NAN; 1000];
        let (xq, w) = ([0u8], [0u8]);
        let kernel: GemvFn = stamp;
        // SAFETY: buffers sized per stamp's expectations.
        unsafe {
            inferno_pool::inferno_par_gemv(kernel, y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), 9, 1000)
        };
        y
    };
    let want: Vec<f32> = (0..1000).map(|r| (r * 3 + 9) as f32).collect();
    assert_eq!(run(), want, "threaded");

    assert!(inferno_pool::set_global_active_threads(1));
    assert_eq!(run(), want, "t=1 via active-threads cap");
    assert!(inferno_pool::set_global_active_threads(4));
}

#[test]
fn set_global_decode_threads_reports_init_state() {
    // Before any init in a fresh test process this could be either state
    // depending on test ordering within the binary; assert the post-init
    // contract explicitly by initializing first.
    inferno_pool::init_global(4).unwrap();
    assert!(inferno_pool::set_global_decode_threads(2));
}
