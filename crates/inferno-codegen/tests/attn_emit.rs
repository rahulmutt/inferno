//! M4b.16 bit-exactness harness: the codegen-emitted geometry-specialized
//! decode attention function vs the Rust hspan kernels (the rig oracle).
//! Exact bit equality, every geometry/span/pos — this is the third lane of
//! the scalar≡AVX2 discipline.
//!
//! # unsafe
//! Same justification as tests/differential.rs: integration test dlopens a
//! shared object and calls a C-ABI fn pointer; production codegen stays
//! unsafe-free.
#![allow(unsafe_code)]

use inferno_codegen::compile_attn_probe;

/// The 13-arg AttnFn ABI (see inferno-kernels registry::AttnFn).
type AttnFn = unsafe extern "C" fn(
    *mut f32,   // out
    *const f32, // q
    *mut f32,   // kv
    *mut f32,   // scores
    usize,      // kv_base
    usize,      // v_off
    usize,      // pos
    usize,      // kv_dim
    usize,      // n_heads
    usize,      // n_kv_heads
    usize,      // head_dim
    usize,      // h_start
    usize,      // h_end
);

/// Deterministic pseudo-random f32 in [-1, 1) — no rand dep, reproducible.
fn lcg_fill(buf: &mut [f32], seed: &mut u64) {
    for v in buf.iter_mut() {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
    }
}

struct Probe {
    // Field order is drop order: the Library must close before its backing
    // tempdir (holding the .so) is removed.
    _lib: libloading::Library,
    f: AttnFn,
    _dir: tempfile::TempDir,
}

fn build_probe(head_dim: usize, n_heads: usize, n_kv_heads: usize, features: &str) -> Probe {
    let dir = tempfile::tempdir().unwrap();
    let so = compile_attn_probe(head_dim, n_heads, n_kv_heads, features, dir.path()).unwrap();
    let lib = unsafe { libloading::Library::new(&so) }.expect("dlopen probe");
    let f = *unsafe { lib.get::<AttnFn>(b"inferno_attn_probe") }.expect("probe symbol");
    Probe {
        f,
        _lib: lib,
        _dir: dir,
    }
}

/// Run emitted-vs-reference for one geometry/pos/span over random inputs.
#[allow(clippy::too_many_arguments)]
fn assert_bit_identical(
    probe: &Probe,
    reference: AttnFn,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    pos: usize,
    h_start: usize,
    h_end: usize,
    seed: u64,
) {
    let kv_dim = n_kv_heads * head_dim;
    let seq_len = pos + 1;
    let v_off = seq_len * kv_dim;
    let kv_base = 0usize;
    let mut seed = seed;
    let mut kv = vec![0f32; 2 * v_off];
    let mut q = vec![0f32; n_heads * head_dim];
    lcg_fill(&mut kv, &mut seed);
    lcg_fill(&mut q, &mut seed);

    let mut out_e = vec![0f32; n_heads * head_dim];
    let mut out_r = vec![0f32; n_heads * head_dim];
    let mut sc_e = vec![0f32; seq_len];
    let mut sc_r = vec![0f32; seq_len];

    unsafe {
        (probe.f)(
            out_e.as_mut_ptr(),
            q.as_ptr(),
            kv.as_mut_ptr(),
            sc_e.as_mut_ptr(),
            kv_base,
            v_off,
            pos,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            h_start,
            h_end,
        );
        reference(
            out_r.as_mut_ptr(),
            q.as_ptr(),
            kv.as_mut_ptr(),
            sc_r.as_mut_ptr(),
            kv_base,
            v_off,
            pos,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            h_start,
            h_end,
        );
    }
    for h in h_start..h_end {
        for d in 0..head_dim {
            let i = h * head_dim + d;
            assert_eq!(
                out_e[i].to_bits(),
                out_r[i].to_bits(),
                "hd={head_dim} nh={n_heads} nkv={n_kv_heads} pos={pos} span=[{h_start},{h_end}) h={h} d={d}: {} vs {}",
                out_e[i],
                out_r[i]
            );
        }
    }
}

#[test]
fn emitted_matches_scalar_protocol_geometry() {
    // qwen2.5-0.5b decode geometry: head_dim 64, 14 heads, 2 kv heads.
    let p = build_probe(64, 14, 2, "+avx2,+fma");
    assert_bit_identical(
        &p,
        inferno_kernels::inferno_attention_f32_scalar_hspan,
        64,
        14,
        2,
        /*pos*/ 63,
        /*span*/ 0,
        14,
        0x4b16,
    );
}
