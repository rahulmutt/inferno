//! Phase-marginal µbench of the decode attention kernel (M4b.15).
//!
//! Attribution by *marginals*: bench-local variants isolate the kernel's
//! phases (dot / max+exp / AV), counterfactual probes are the Lever 1
//! candidates, roofline anchors bound each phase. Admissibility rules and
//! gate formulas: the M4b.15 spec §The instrument / §Pre-registered gates.
//!
//! `full_local` is the FROZEN pre-Lever-1 copy of `attn_core_avx2`'s
//! loops. After Lever 1 lands, `full` (the public symbol) is the new
//! kernel and the per-box kernel reduction is r = 1 − time(full) /
//! time(full_local). `assert_bit_identity()` runs before every bench
//! session and therefore doubles as a standing Lever 1 bit-neutrality
//! check: if `full` ever diverges bitwise from the frozen copy, the run
//! aborts (instrument inadmissible).
//!
//! `bexpf_*` are verbatim copies of `crate::expf` (pub(crate), benches
//! can't reach it); copy drift is caught by the same bitwise assert.
//! Numbers are only meaningful from quiet hardware; local runs feed the
//! Lever 1 gates as *ratios*, honestly labeled non-quiet.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group};
use std::arch::x86_64::*;
use std::hint::black_box;

const N_HEADS: usize = 14;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
const SEQ_LEN: usize = 2048;
/// 511/639 bracket the bench protocol's decode range (pp 512 + tg 128);
/// the rest are for understanding, not gating (spec §The instrument).
const POSITIONS: [usize; 5] = [127, 511, 639, 1023, 2047];

// ---- bench-local expf (verbatim from src/expf.rs; drift caught by the
// bitwise assert against the public symbol) ----

const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;
const C: [f32; 7] = [
    1.0,
    1.0,
    0.5,
    0.166_666_67,
    0.041_666_67,
    0.008_333_34,
    0.001_388_888_9,
];

#[inline]
fn bexpf_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * LOG2E).round_ties_even();
    let r = n.mul_add(-LN2_LO, n.mul_add(-LN2_HI, x));
    let mut p = C[6];
    p = p.mul_add(r, C[5]);
    p = p.mul_add(r, C[4]);
    p = p.mul_add(r, C[3]);
    p = p.mul_add(r, C[2]);
    p = p.mul_add(r, C[1]);
    p = p.mul_add(r, C[0]);
    let pow2n = f32::from_bits((((n as i32) + 127) << 23) as u32);
    p * pow2n
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn bexpf_avx2(x: __m256) -> __m256 {
    let x = _mm256_min_ps(
        _mm256_set1_ps(88.0),
        _mm256_max_ps(_mm256_set1_ps(-88.0), x),
    );
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(LOG2E),
    ));
    let r = _mm256_fmadd_ps(
        n,
        _mm256_set1_ps(-LN2_LO),
        _mm256_fmadd_ps(n, _mm256_set1_ps(-LN2_HI), x),
    );
    let mut p = _mm256_set1_ps(C[6]);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[5]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[4]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[3]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[2]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[1]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[0]));
    let ni = _mm256_cvtps_epi32(n);
    let bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(ni, _mm256_set1_epi32(127)));
    _mm256_mul_ps(p, _mm256_castsi256_ps(bits))
}

/// hsum8, verbatim from src/attention.rs (same tree — bit-identity).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bhsum8(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let s = _mm_add_ps(lo, hi);
    let sh = _mm_movehl_ps(s, s);
    let s2 = _mm_add_ps(s, sh);
    let s3 = _mm_add_ss(s2, _mm_shuffle_ps::<0x55>(s2, s2));
    _mm_cvtss_f32(s3)
}

// ---- buffers ----

/// Deterministic varied fill (spread exp inputs; no RNG dependency).
fn lcg_fill(mut seed: u64, buf: &mut [f32]) {
    for v in buf.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
    }
}

struct Buffers {
    q: Vec<f32>,
    kv: Vec<f32>,
    out: Vec<f32>,
    scores: Vec<f32>,
}

impl Buffers {
    fn new() -> Self {
        let mut q = vec![0f32; N_HEADS * HEAD_DIM];
        let mut kv = vec![0f32; 2 * SEQ_LEN * KV_DIM];
        lcg_fill(0x4b15_0001, &mut q);
        lcg_fill(0x4b15_0002, &mut kv);
        Buffers {
            q,
            kv,
            out: vec![0f32; N_HEADS * HEAD_DIM],
            scores: vec![0f32; SEQ_LEN],
        }
    }
    const V_OFF: usize = SEQ_LEN * KV_DIM;
}

// ---- phase passes (bench-local copies of attn_core_avx2's loops; each
// takes the per-head slice pointers the kernel derives) ----

/// Scores pass for one head: scores[t] = hsum8(dot(qh, k[t,g])) * scale.
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::missing_safety_doc)]
unsafe fn dot_pass(
    qh: *const f32,
    kv: *const f32,
    kreg: usize,
    g: usize,
    scale: f32,
    visible: usize,
    scores: *mut f32,
) {
    for t in 0..visible {
        let kb = unsafe { kv.add(kreg + t * KV_DIM + g * HEAD_DIM) };
        let mut acc = _mm256_setzero_ps();
        let mut d = 0;
        while d < HEAD_DIM {
            let qv = unsafe { _mm256_loadu_ps(qh.add(d)) };
            let kvv = unsafe { _mm256_loadu_ps(kb.add(d)) };
            acc = _mm256_fmadd_ps(qv, kvv, acc);
            d += 8;
        }
        unsafe { *scores.add(t) = bhsum8(acc) * scale };
    }
}

/// Max pass (the kernel's scalar max fold).
#[inline]
#[allow(clippy::missing_safety_doc)]
unsafe fn max_pass(scores: *const f32, visible: usize) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for t in 0..visible {
        max = max.max(unsafe { *scores.add(t) });
    }
    max
}

/// Exp + denom pass (8-wide blocks + scalar tail), in-place over scores.
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::missing_safety_doc)]
unsafe fn exp_pass(scores: *mut f32, visible: usize, max: f32) -> f32 {
    let maxv = _mm256_set1_ps(max);
    let mut denom = 0f32;
    let mut t = 0;
    while t + 8 <= visible {
        let s = unsafe { _mm256_loadu_ps(scores.add(t)) };
        let e = unsafe { bexpf_avx2(_mm256_sub_ps(s, maxv)) };
        unsafe { _mm256_storeu_ps(scores.add(t), e) };
        denom += unsafe { bhsum8(e) };
        t += 8;
    }
    while t < visible {
        let e = bexpf_scalar(unsafe { *scores.add(t) } - max);
        unsafe { *scores.add(t) = e };
        denom += e;
        t += 1;
    }
    denom
}

/// AV pass (the store-reload loop, verbatim).
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::missing_safety_doc)]
unsafe fn av_pass(
    oh: *mut f32,
    kv: *const f32,
    vreg: usize,
    g: usize,
    scores: *const f32,
    denom: f32,
    visible: usize,
) {
    for d in (0..HEAD_DIM).step_by(8) {
        unsafe { _mm256_storeu_ps(oh.add(d), _mm256_setzero_ps()) };
    }
    for t in 0..visible {
        let wn = unsafe { _mm256_set1_ps(*scores.add(t) / denom) };
        let vb = unsafe { kv.add(vreg + t * KV_DIM + g * HEAD_DIM) };
        for d in (0..HEAD_DIM).step_by(8) {
            let cur = unsafe { _mm256_loadu_ps(oh.add(d)) };
            let vv = unsafe { _mm256_loadu_ps(vb.add(d)) };
            unsafe { _mm256_storeu_ps(oh.add(d), _mm256_fmadd_ps(wn, vv, cur)) };
        }
    }
}

/// Blocked scores pass: NB positions in flight on independent
/// accumulators. Each position's fmadd order over d and its hsum8 tree
/// are exactly `dot_pass`'s — bit-identical per position (spec §Lever 1).
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::missing_safety_doc)]
unsafe fn dot_pass_blocked<const NB: usize>(
    qh: *const f32,
    kv: *const f32,
    kreg: usize,
    g: usize,
    scale: f32,
    visible: usize,
    scores: *mut f32,
) {
    let mut t = 0;
    while t + NB <= visible {
        let kb = unsafe { kv.add(kreg + t * KV_DIM + g * HEAD_DIM) };
        let mut acc = [_mm256_setzero_ps(); NB];
        let mut d = 0;
        while d < HEAD_DIM {
            let qv = unsafe { _mm256_loadu_ps(qh.add(d)) };
            for (j, a) in acc.iter_mut().enumerate() {
                let kvv = unsafe { _mm256_loadu_ps(kb.add(j * KV_DIM + d)) };
                *a = _mm256_fmadd_ps(qv, kvv, *a);
            }
            d += 8;
        }
        for (j, a) in acc.iter().enumerate() {
            unsafe { *scores.add(t + j) = bhsum8(*a) * scale };
        }
        t += NB;
    }
    // Tail: the original single-position loop.
    while t < visible {
        let kb = unsafe { kv.add(kreg + t * KV_DIM + g * HEAD_DIM) };
        let mut acc = _mm256_setzero_ps();
        let mut d = 0;
        while d < HEAD_DIM {
            let qv = unsafe { _mm256_loadu_ps(qh.add(d)) };
            let kvv = unsafe { _mm256_loadu_ps(kb.add(d)) };
            acc = _mm256_fmadd_ps(qv, kvv, acc);
            d += 8;
        }
        unsafe { *scores.add(t) = bhsum8(acc) * scale };
        t += 1;
    }
}

/// Register-resident AV pass: the 64-float out row lives in 8 YMM
/// accumulators across the whole visible loop, stored once. Per-element
/// accumulation order stays t-ascending — bit-identical (spec §Lever 1).
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::missing_safety_doc)]
unsafe fn av_pass_regacc(
    oh: *mut f32,
    kv: *const f32,
    vreg: usize,
    g: usize,
    scores: *const f32,
    denom: f32,
    visible: usize,
) {
    let mut acc = [_mm256_setzero_ps(); 8];
    for t in 0..visible {
        let wn = unsafe { _mm256_set1_ps(*scores.add(t) / denom) };
        let vb = unsafe { kv.add(vreg + t * KV_DIM + g * HEAD_DIM) };
        for (j, a) in acc.iter_mut().enumerate() {
            let vv = unsafe { _mm256_loadu_ps(vb.add(j * 8)) };
            *a = _mm256_fmadd_ps(wn, vv, *a);
        }
    }
    for (j, a) in acc.iter().enumerate() {
        unsafe { _mm256_storeu_ps(oh.add(j * 8), *a) };
    }
}

// ---- variants ----

/// FROZEN pre-Lever-1 copy of attn_core_avx2 over heads [h0, h1).
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn full_local(b: &mut Buffers, pos: usize, h0: usize, h1: usize) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let group = N_HEADS / N_KV_HEADS;
    let visible = pos + 1;
    for h in h0..h1 {
        let g = h / group;
        let qh = unsafe { b.q.as_ptr().add(h * HEAD_DIM) };
        unsafe {
            dot_pass(
                qh,
                b.kv.as_ptr(),
                0,
                g,
                scale,
                visible,
                b.scores.as_mut_ptr(),
            )
        };
        let max = unsafe { max_pass(b.scores.as_ptr(), visible) };
        let denom = unsafe { exp_pass(b.scores.as_mut_ptr(), visible, max) };
        unsafe {
            av_pass(
                b.out.as_mut_ptr().add(h * HEAD_DIM),
                b.kv.as_ptr(),
                Buffers::V_OFF,
                g,
                b.scores.as_ptr(),
                denom,
                visible,
            )
        };
    }
}

/// dot + max only (no exp, no AV). The max is black_boxed so the pass
/// isn't dead-code-eliminated; scores stores are the kernel's own.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn dot_only(b: &mut Buffers, pos: usize) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let group = N_HEADS / N_KV_HEADS;
    let visible = pos + 1;
    for h in 0..N_HEADS {
        let g = h / group;
        let qh = unsafe { b.q.as_ptr().add(h * HEAD_DIM) };
        unsafe {
            dot_pass(
                qh,
                b.kv.as_ptr(),
                0,
                g,
                scale,
                visible,
                b.scores.as_mut_ptr(),
            )
        };
        black_box(unsafe { max_pass(b.scores.as_ptr(), visible) });
    }
}

/// dot + max + exp + denom (no AV).
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn no_av(b: &mut Buffers, pos: usize) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let group = N_HEADS / N_KV_HEADS;
    let visible = pos + 1;
    for h in 0..N_HEADS {
        let g = h / group;
        let qh = unsafe { b.q.as_ptr().add(h * HEAD_DIM) };
        unsafe {
            dot_pass(
                qh,
                b.kv.as_ptr(),
                0,
                g,
                scale,
                visible,
                b.scores.as_mut_ptr(),
            )
        };
        let max = unsafe { max_pass(b.scores.as_ptr(), visible) };
        black_box(unsafe { exp_pass(b.scores.as_mut_ptr(), visible, max) });
    }
}

/// Whole kernel with the blocked dot pass (Lever 1a candidate).
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn dot_blocked<const NB: usize>(b: &mut Buffers, pos: usize) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let group = N_HEADS / N_KV_HEADS;
    let visible = pos + 1;
    for h in 0..N_HEADS {
        let g = h / group;
        let qh = unsafe { b.q.as_ptr().add(h * HEAD_DIM) };
        unsafe {
            dot_pass_blocked::<NB>(
                qh,
                b.kv.as_ptr(),
                0,
                g,
                scale,
                visible,
                b.scores.as_mut_ptr(),
            )
        };
        let max = unsafe { max_pass(b.scores.as_ptr(), visible) };
        let denom = unsafe { exp_pass(b.scores.as_mut_ptr(), visible, max) };
        unsafe {
            av_pass(
                b.out.as_mut_ptr().add(h * HEAD_DIM),
                b.kv.as_ptr(),
                Buffers::V_OFF,
                g,
                b.scores.as_ptr(),
                denom,
                visible,
            )
        };
    }
}

/// Whole kernel with the register-resident AV pass (Lever 1b candidate).
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn av_regacc(b: &mut Buffers, pos: usize) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let group = N_HEADS / N_KV_HEADS;
    let visible = pos + 1;
    for h in 0..N_HEADS {
        let g = h / group;
        let qh = unsafe { b.q.as_ptr().add(h * HEAD_DIM) };
        unsafe {
            dot_pass(
                qh,
                b.kv.as_ptr(),
                0,
                g,
                scale,
                visible,
                b.scores.as_mut_ptr(),
            )
        };
        let max = unsafe { max_pass(b.scores.as_ptr(), visible) };
        let denom = unsafe { exp_pass(b.scores.as_mut_ptr(), visible, max) };
        unsafe {
            av_pass_regacc(
                b.out.as_mut_ptr().add(h * HEAD_DIM),
                b.kv.as_ptr(),
                Buffers::V_OFF,
                g,
                b.scores.as_ptr(),
                denom,
                visible,
            )
        };
    }
}

/// Both levers together (sanity row for the "both fire" case).
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn combined(b: &mut Buffers, pos: usize) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let group = N_HEADS / N_KV_HEADS;
    let visible = pos + 1;
    for h in 0..N_HEADS {
        let g = h / group;
        let qh = unsafe { b.q.as_ptr().add(h * HEAD_DIM) };
        unsafe {
            dot_pass_blocked::<4>(
                qh,
                b.kv.as_ptr(),
                0,
                g,
                scale,
                visible,
                b.scores.as_mut_ptr(),
            )
        };
        let max = unsafe { max_pass(b.scores.as_ptr(), visible) };
        let denom = unsafe { exp_pass(b.scores.as_mut_ptr(), visible, max) };
        unsafe {
            av_pass_regacc(
                b.out.as_mut_ptr().add(h * HEAD_DIM),
                b.kv.as_ptr(),
                Buffers::V_OFF,
                g,
                b.scores.as_ptr(),
                denom,
                visible,
            )
        };
    }
}

/// The shipping kernel via its public symbol.
fn full(b: &mut Buffers, pos: usize) {
    // SAFETY: buffers sized per the kernel contract (scores >= pos+1,
    // kv covers K and V regions, out/q are N_HEADS*HEAD_DIM).
    unsafe {
        inferno_kernels::inferno_attention_f32_avx2(
            b.out.as_mut_ptr(),
            b.q.as_ptr(),
            b.kv.as_mut_ptr(),
            b.scores.as_mut_ptr(),
            0,
            Buffers::V_OFF,
            pos,
            KV_DIM,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
        );
    }
}

// ---- admissibility: bitwise identity (spec §The instrument) ----

/// Aborts the whole bench run if any variant that claims bit-identity
/// diverges from the public symbol. Checked at a tail-heavy pos (9) and a
/// protocol pos (639).
fn assert_bit_identity() {
    assert!(
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"),
        "attn_decode_phases requires avx2+fma"
    );
    for &pos in &[9usize, 639] {
        let mut want = Buffers::new();
        full(&mut want, pos);
        let mut got = Buffers::new();
        // SAFETY: avx2+fma detected above; buffer contract as in `full`.
        unsafe { full_local(&mut got, pos, 0, N_HEADS) };
        for i in 0..N_HEADS * HEAD_DIM {
            assert_eq!(
                want.out[i].to_bits(),
                got.out[i].to_bits(),
                "full_local diverges from public symbol at pos={pos} idx={i} — \
                 instrument inadmissible (copy drift or a non-bit-neutral lever)"
            );
        }
        for (name, f) in [
            (
                "dot_blocked2",
                dot_blocked::<2> as unsafe fn(&mut Buffers, usize),
            ),
            ("dot_blocked4", dot_blocked::<4>),
            ("dot_blocked8", dot_blocked::<8>),
            ("av_regacc", av_regacc),
            ("combined", combined),
        ] {
            let mut got = Buffers::new();
            // SAFETY: avx2+fma detected at the top of this fn.
            unsafe { f(&mut got, pos) };
            for i in 0..N_HEADS * HEAD_DIM {
                assert_eq!(
                    want.out[i].to_bits(),
                    got.out[i].to_bits(),
                    "{name} diverges at pos={pos} idx={i} — probe is NOT bit-neutral"
                );
            }
        }
    }
}

// ---- criterion groups ----

fn bench_phases(c: &mut Criterion) {
    let mut group = c.benchmark_group("phases");
    for &pos in &POSITIONS {
        let mut b = Buffers::new();
        group.bench_with_input(BenchmarkId::new("full", pos), &pos, |bch, &pos| {
            bch.iter(|| full(black_box(&mut b), pos))
        });
        let mut b = Buffers::new();
        group.bench_with_input(BenchmarkId::new("full_local", pos), &pos, |bch, &pos| {
            // SAFETY: avx2+fma asserted in main; buffer contract as `full`.
            bch.iter(|| unsafe { full_local(black_box(&mut b), pos, 0, N_HEADS) })
        });
        let mut b = Buffers::new();
        group.bench_with_input(BenchmarkId::new("dot_only", pos), &pos, |bch, &pos| {
            // SAFETY: avx2+fma asserted in main.
            bch.iter(|| unsafe { dot_only(black_box(&mut b), pos) })
        });
        let mut b = Buffers::new();
        group.bench_with_input(BenchmarkId::new("no_av", pos), &pos, |bch, &pos| {
            // SAFETY: avx2+fma asserted in main.
            bch.iter(|| unsafe { no_av(black_box(&mut b), pos) })
        });
    }
    group.finish();
}

fn bench_probes(c: &mut Criterion) {
    let mut group = c.benchmark_group("probes");
    for &pos in &POSITIONS {
        macro_rules! probe {
            ($name:literal, $f:expr) => {{
                let mut b = Buffers::new();
                group.bench_with_input(BenchmarkId::new($name, pos), &pos, |bch, &pos| {
                    // SAFETY: avx2+fma asserted in main.
                    bch.iter(|| unsafe { $f(black_box(&mut b), pos) })
                });
            }};
        }
        probe!("dot_blocked2", dot_blocked::<2>);
        probe!("dot_blocked4", dot_blocked::<4>);
        probe!("dot_blocked8", dot_blocked::<8>);
        probe!("av_regacc", av_regacc);
        probe!("combined", combined);
    }
    group.finish();
}

fn bench_anchors(c: &mut Criterion) {
    let mut group = c.benchmark_group("anchors");
    // Bandwidth bound: stream the K and V regions the kernel touches.
    for &pos in &POSITIONS {
        let b = Buffers::new();
        let visible = pos + 1;
        group.throughput(Throughput::Bytes((2 * visible * KV_DIM * 4) as u64));
        group.bench_with_input(BenchmarkId::new("kv_stream", pos), &pos, |bch, _| {
            bch.iter(|| {
                // SAFETY: avx2 asserted in main; reads stay inside b.kv.
                unsafe {
                    let mut s = _mm256_setzero_ps();
                    for base in [0usize, Buffers::V_OFF] {
                        let p = b.kv.as_ptr().add(base);
                        let n = visible * KV_DIM;
                        let mut i = 0;
                        while i < n {
                            s = _mm256_add_ps(s, _mm256_loadu_ps(p.add(i)));
                            i += 8;
                        }
                    }
                    black_box(bhsum8(s))
                }
            })
        });
    }
    // Compute bound: 8 independent FMA chains, 4096 iterations.
    group.throughput(Throughput::Elements(8 * 8 * 4096));
    group.bench_function("fma_peak", |bch| {
        bch.iter(|| {
            // SAFETY: avx2+fma asserted in main.
            unsafe {
                let mut a = [_mm256_set1_ps(1.000_001); 8];
                let m = _mm256_set1_ps(0.999_999);
                for _ in 0..4096 {
                    for x in a.iter_mut() {
                        *x = _mm256_fmadd_ps(*x, m, *x);
                    }
                }
                black_box(bhsum8(a[0]))
            }
        })
    });
    group.finish();
}

fn bench_cold_warm(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_warm");
    // One head (h=0), KV evicted vs resident — bounds how much cache
    // state can explain the 16c per-lane spread (spec §The instrument).
    let mut thrash = vec![0u8; 64 * 1024 * 1024];
    for &pos in &[639usize, 2047] {
        let mut b = Buffers::new();
        group.bench_with_input(BenchmarkId::new("warm", pos), &pos, |bch, &pos| {
            // SAFETY: avx2+fma asserted in main.
            bch.iter(|| unsafe { full_local(black_box(&mut b), pos, 0, 1) })
        });
        let mut b = Buffers::new();
        group.bench_with_input(BenchmarkId::new("cold", pos), &pos, |bch, &pos| {
            bch.iter_batched(
                || {
                    for (i, x) in thrash.iter_mut().enumerate() {
                        *x = x.wrapping_add(i as u8);
                    }
                },
                // SAFETY: avx2+fma asserted in main.
                |_| unsafe { full_local(black_box(&mut b), pos, 0, 1) },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_phases,
    bench_probes,
    bench_anchors,
    bench_cold_warm
);

fn main() {
    assert_bit_identity();
    benches();
    Criterion::default().configure_from_args().final_summary();
}
