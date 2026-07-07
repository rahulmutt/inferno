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
    // rows == 0 is a no-op even uninitialized: y must come back unchanged.
    let before = y.clone();
    // SAFETY: rows == 0 → no writes.
    unsafe {
        inferno_pool::inferno_par_gemv(kernel, y.as_mut_ptr(), xq.as_ptr(), w.as_ptr(), 5, 0)
    };
    assert_eq!(y, before, "rows == 0 must not touch y");
}
