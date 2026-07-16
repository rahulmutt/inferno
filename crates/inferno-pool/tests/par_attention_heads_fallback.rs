//! `inferno_par_attention_heads` without an initialized global pool: the
//! entry point must degrade to one serial full-range hspan call — this
//! file never calls `init_global`, and an integration test binary is its
//! own process, so the pool is guaranteed absent.
#![allow(unsafe_code)] // FFI entry-point tests; same justification as the sibling test files.

use inferno_pool::inferno_par_attention_heads;

unsafe extern "C" fn stamp_attn_heads(
    out: *mut f32,
    q: *const f32,
    _kv: *mut f32,
    scores: *mut f32,
    _kv_base: usize,
    _v_off: usize,
    pos: usize,
    _kv_dim: usize,
    _n_heads: usize,
    _n_kv_heads: usize,
    head_dim: usize,
    h_start: usize,
    h_end: usize,
) {
    // SAFETY: the dispatcher sizes scores to pos + 1.
    unsafe { *scores.add(pos) = pos as f32 };
    for h in h_start..h_end {
        for d in 0..head_dim {
            let i = h * head_dim + d;
            // SAFETY: out/q rows are n_heads*head_dim per the contract.
            unsafe { *out.add(i) = *q.add(i) + (h * 31 + d + pos) as f32 };
        }
    }
}

const NH: usize = 14;
const HD: usize = 4;

fn dispatch(n_heads: usize, pos: usize) -> Vec<f32> {
    let q: Vec<f32> = (0..NH * HD).map(|i| i as f32).collect();
    let mut out = vec![f32::NAN; NH * HD];
    let mut kv = [0f32; 1];
    // SAFETY: buffers sized per stamp_attn_heads' expectations.
    unsafe {
        inferno_par_attention_heads(
            stamp_attn_heads,
            out.as_mut_ptr(),
            q.as_ptr(),
            kv.as_mut_ptr(),
            pos,
            0,
            0,
            0,
            n_heads,
            2,
            HD,
        );
    }
    out
}

fn expected(pos: usize) -> Vec<f32> {
    (0..NH * HD)
        .map(|i| {
            let (h, d) = (i / HD, i % HD);
            i as f32 + (h * 31 + d + pos) as f32
        })
        .collect()
}

#[test]
fn uninitialized_pool_falls_back_to_serial_full_range() {
    for pos in [0, 41] {
        assert_eq!(dispatch(NH, pos), expected(pos), "pos={pos}");
    }
}

#[test]
fn zero_heads_is_a_noop() {
    let out = dispatch(0, 0);
    assert!(out.iter().all(|v| v.is_nan()), "no head row may be written");
}
