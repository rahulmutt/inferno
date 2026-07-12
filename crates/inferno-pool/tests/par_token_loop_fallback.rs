//! `inferno_par_token_loop` without an initialized global pool: the entry
//! point must degrade to the serial full-range body call (and take the
//! m == 1 direct path) — this file never calls `init_global`, and an
//! integration test binary is its own process, so the pool is guaranteed
//! absent. The stub's coercion to `TokenBodyFn` in these calls is also the
//! ABI drift guard: there is no inferno-kernels symbol to coerce (bodies
//! are codegen-emitted), so this stub plays that role.
#![allow(unsafe_code)] // FFI entry-point tests; same justification as the sibling test files.

use inferno_pool::{TokenBodyFn, inferno_par_token_loop};

unsafe extern "C" fn stamp_tokens(ctx: *const u8, t0: usize, t1: usize) {
    let words = ctx as *const usize;
    // SAFETY: tests pass a 2-word ctx pack, live for the call.
    let out = unsafe { *words } as *mut f32;
    let stride = unsafe { *words.add(1) };
    for t in t0..t1 {
        for i in 0..stride {
            // SAFETY: out has m*stride elements and t < m per contract.
            unsafe { *out.add(t * stride + i) = (t * 31 + i) as f32 };
        }
    }
}

const STRIDE: usize = 5;

fn dispatch(m: usize) -> Vec<f32> {
    let body: TokenBodyFn = stamp_tokens; // ABI coercion is part of the test
    let mut out = vec![f32::NAN; m * STRIDE];
    let ctx = [out.as_mut_ptr() as usize, STRIDE];
    // SAFETY: ctx/out sized per stamp_tokens' expectations, live for the call.
    unsafe { inferno_par_token_loop(body, ctx.as_ptr() as *const u8, m) };
    out
}

fn expected(m: usize) -> Vec<f32> {
    (0..m * STRIDE)
        .map(|j| ((j / STRIDE) * 31 + j % STRIDE) as f32)
        .collect()
}

#[test]
fn uninitialized_pool_falls_back_to_serial_full_range() {
    for m in [2, 7, 64] {
        assert_eq!(dispatch(m), expected(m), "m={m}");
    }
}

#[test]
fn m1_takes_the_direct_path() {
    // Decode-shaped span (and the T=1 prefill tile): one token, computed
    // correctly with no pool involvement by construction.
    assert_eq!(dispatch(1), expected(1));
}

#[test]
fn m0_is_a_noop() {
    assert!(dispatch(0).is_empty());
}
