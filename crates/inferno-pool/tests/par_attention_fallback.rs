//! `inferno_par_attention` without an initialized global pool: the entry
//! point must degrade to the serial full-range loop (and take the m == 1
//! direct path) — this file never calls `init_global`, and an integration
//! test binary is its own process, so the pool is guaranteed absent.
#![allow(unsafe_code)] // FFI entry-point tests; same justification as the sibling test files.

use inferno_pool::inferno_par_attention;

unsafe extern "C" fn stamp_attn(
    out: *mut f32,
    q: *const f32,
    _kv: *mut f32,
    scores: *mut f32,
    _kv_base: usize,
    _v_off: usize,
    pos: usize,
    _kv_dim: usize,
    n_heads: usize,
    _n_kv_heads: usize,
    head_dim: usize,
) {
    // SAFETY: the dispatcher sizes scores to max pos + 1 for its span.
    unsafe { *scores.add(pos) = pos as f32 };
    for i in 0..n_heads * head_dim {
        // SAFETY: out/q rows are n_heads*head_dim per the AttnFn contract.
        unsafe { *out.add(i) = *q.add(i) + (pos * 31 + i) as f32 };
    }
}

const NH: usize = 3;
const HD: usize = 4;
const STRIDE: usize = NH * HD;

fn dispatch(m: usize, pos0: usize) -> Vec<f32> {
    let q: Vec<f32> = (0..m * STRIDE).map(|i| i as f32).collect();
    let mut out = vec![f32::NAN; m * STRIDE];
    let mut kv = [0f32; 1];
    // SAFETY: buffers sized per stamp_attn's expectations, live for the call.
    unsafe {
        inferno_par_attention(
            stamp_attn,
            out.as_mut_ptr(),
            q.as_ptr(),
            kv.as_mut_ptr(),
            pos0,
            m,
            0,
            0,
            0,
            NH,
            1,
            HD,
            STRIDE,
            STRIDE,
        );
    }
    out
}

fn expected(m: usize, pos0: usize) -> Vec<f32> {
    (0..m * STRIDE)
        .map(|j| {
            let (t, i) = (j / STRIDE, j % STRIDE);
            j as f32 + ((pos0 + t) * 31 + i) as f32
        })
        .collect()
}

#[test]
fn uninitialized_pool_falls_back_to_serial_full_range() {
    for m in [2, 7, 64] {
        assert_eq!(dispatch(m, 3), expected(m, 3), "m={m}");
    }
}

#[test]
fn m1_takes_the_direct_path() {
    // Decode-shaped call (and the T=1 prefill tile): one token, computed
    // correctly with no pool involvement by construction.
    assert_eq!(dispatch(1, 41), expected(1, 41));
}

#[test]
fn m0_is_a_noop() {
    assert!(dispatch(0, 0).is_empty());
}
