# M4b.15 — Decode Attention Kernel Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain where the decode-attention hspan kernel's cycles go (phase-marginal µbench), then ship only the gate-authorized bit-neutral kernel fixes and judge closing tg against a headroom-set target on both quiet-hw boxes.

**Architecture:** A new criterion bench (`attn_decode_phases`) times phase-isolating bench-local copies of `attn_core_avx2`'s loops plus counterfactual probes that double as the Lever 1 candidates. Gates 1a/1b (pre-registered in the spec) decide which of two bit-identical kernel changes land in `attn_core_avx2`. Two quiet-hw sessions (16c + 8c) fix the headroom target, feed Gate 2 (KV-split kernel), and close the milestone.

**Tech Stack:** Rust, AVX2/FMA intrinsics, criterion, the M2 rig (`inferno-kernels/tests/rig.rs`), quiet-hw gate scripts (`scripts/quiet-hw/`), PhoenixNAP metal boxes via `mise run metal`.

**Spec:** `docs/superpowers/specs/2026-07-17-m4b15-decode-attention-kernel-design.md` — the pre-registered gates, thresholds, and formulas live THERE. This plan never restates a threshold as authority; when in doubt the spec wins.

## Global Constraints

- **Bit-identity:** every Lever 1 change must leave every kernel output bit-identical. The scalar kernel is NOT touched. `attn_rel_tol` / `logits_abs_tol` are NOT touched.
- **`inferno-kernels` allows `unsafe` (own `[lints.rust]` table); everything else in the workspace denies it.** The bench and kernel edits stay inside `inferno-kernels` except the two session scripts.
- **No lever ships without its gate verdict recorded first** in the spec's §Amendments, arithmetic shown. Gate verdicts are computed by a human/controller from recorded data, never inferred.
- **Recorded data points are never edited** (erratum pattern for corrections).
- **Verification commands:** `mise run test`, `mise run lint` (clippy -D warnings — run it, CI does), `cargo test -p inferno-kernels --test rig`, `cargo test -p inferno-codegen --test differential`, `cargo test -p inferno-core --test artifact`.
- **Quiet-hw sessions:** never provision two PNAP servers in parallel; after ANY failed session run `mise run metal-gc` and confirm zero servers. Commit and push before `mise run metal` (the box clones committed HEAD).
- Branch: work on `m4b15-design` (already exists, holds the spec); PR to `main` at the end.

---

### Task 1: µbench scaffolding — `full` vs `full_local` + bit-identity check

**Files:**
- Create: `crates/inferno-kernels/benches/attn_decode_phases.rs`
- Modify: `crates/inferno-kernels/Cargo.toml` (add `[[bench]]`)

**Interfaces:**
- Consumes: `inferno_kernels::inferno_attention_f32_avx2` (public C-ABI symbol).
- Produces: bench-local helpers later tasks extend: `Buffers`, `lcg_fill`, `bexpf_avx2`/`bexpf_scalar`, pass fns `dot_pass`, `max_pass`, `exp_pass`, `av_pass`, variant fn `full_local`, and `assert_bit_identity()`. Geometry consts `N_HEADS=14, N_KV_HEADS=2, HEAD_DIM=64, KV_DIM=128, SEQ_LEN=2048`, `POSITIONS = [127, 511, 639, 1023, 2047]`.

- [ ] **Step 1: Register the bench**

In `crates/inferno-kernels/Cargo.toml`, after the existing `[[bench]] name = "attention"` block, add:

```toml
[[bench]]
name = "attn_decode_phases"
harness = false
```

- [ ] **Step 2: Create the bench file with helpers, the frozen kernel copy, and the bit-identity assert**

Create `crates/inferno-kernels/benches/attn_decode_phases.rs`:

```rust
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

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
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
        let kb = kv.add(kreg + t * KV_DIM + g * HEAD_DIM);
        let mut acc = _mm256_setzero_ps();
        let mut d = 0;
        while d < HEAD_DIM {
            let qv = _mm256_loadu_ps(qh.add(d));
            let kvv = _mm256_loadu_ps(kb.add(d));
            acc = _mm256_fmadd_ps(qv, kvv, acc);
            d += 8;
        }
        *scores.add(t) = bhsum8(acc) * scale;
    }
}

/// Max pass (the kernel's scalar max fold).
#[inline]
#[allow(clippy::missing_safety_doc)]
unsafe fn max_pass(scores: *const f32, visible: usize) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for t in 0..visible {
        max = max.max(*scores.add(t));
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
        let s = _mm256_loadu_ps(scores.add(t));
        let e = bexpf_avx2(_mm256_sub_ps(s, maxv));
        _mm256_storeu_ps(scores.add(t), e);
        denom += bhsum8(e);
        t += 8;
    }
    while t < visible {
        let e = bexpf_scalar(*scores.add(t) - max);
        *scores.add(t) = e;
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
        _mm256_storeu_ps(oh.add(d), _mm256_setzero_ps());
    }
    for t in 0..visible {
        let wn = _mm256_set1_ps(*scores.add(t) / denom);
        let vb = kv.add(vreg + t * KV_DIM + g * HEAD_DIM);
        for d in (0..HEAD_DIM).step_by(8) {
            let cur = _mm256_loadu_ps(oh.add(d));
            let vv = _mm256_loadu_ps(vb.add(d));
            _mm256_storeu_ps(oh.add(d), _mm256_fmadd_ps(wn, vv, cur));
        }
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
        let qh = b.q.as_ptr().add(h * HEAD_DIM);
        dot_pass(qh, b.kv.as_ptr(), 0, g, scale, visible, b.scores.as_mut_ptr());
        let max = max_pass(b.scores.as_ptr(), visible);
        let denom = exp_pass(b.scores.as_mut_ptr(), visible, max);
        av_pass(
            b.out.as_mut_ptr().add(h * HEAD_DIM),
            b.kv.as_ptr(),
            Buffers::V_OFF,
            g,
            b.scores.as_ptr(),
            denom,
            visible,
        );
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
    }
    group.finish();
}

criterion_group!(benches, bench_phases);

fn main() {
    assert_bit_identity();
    benches();
    Criterion::default().configure_from_args().final_summary();
}
```

- [ ] **Step 3: Smoke-run the bench (fast mode) and verify the assert + both benchmarks run**

Run: `cargo bench -p inferno-kernels --bench attn_decode_phases -- --test`
Expected: exits 0; lists `phases/full/127` … `phases/full_local/2047` as "Testing … Success". A bitwise divergence would panic before any benchmark.

- [ ] **Step 4: Lint + workspace tests still green**

Run: `mise run lint && cargo test -p inferno-kernels`
Expected: clippy clean (benches are linted with `-D warnings`), tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/Cargo.toml crates/inferno-kernels/benches/attn_decode_phases.rs
git commit -m "M4b.15 Task 1: attn_decode_phases µbench scaffolding (full vs frozen copy, bit-identity assert)"
```

---

### Task 2: Marginal variants — `dot_only`, `no_av`

**Files:**
- Modify: `crates/inferno-kernels/benches/attn_decode_phases.rs`

**Interfaces:**
- Consumes: Task 1's passes and `Buffers`.
- Produces: `dot_only(b, pos)`, `no_av(b, pos)` bench-local fns; criterion IDs `phases/dot_only/<pos>`, `phases/no_av/<pos>`. Marginals are computed by the human from criterion means: dot = `dot_only`, softmax = `no_av − dot_only`, AV = `full_local − no_av` (spec §The instrument).

- [ ] **Step 1: Add the two variants** (after `full_local` in the bench file):

```rust
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
        let qh = b.q.as_ptr().add(h * HEAD_DIM);
        dot_pass(qh, b.kv.as_ptr(), 0, g, scale, visible, b.scores.as_mut_ptr());
        black_box(max_pass(b.scores.as_ptr(), visible));
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
        let qh = b.q.as_ptr().add(h * HEAD_DIM);
        dot_pass(qh, b.kv.as_ptr(), 0, g, scale, visible, b.scores.as_mut_ptr());
        let max = max_pass(b.scores.as_ptr(), visible);
        black_box(exp_pass(b.scores.as_mut_ptr(), visible, max));
    }
}
```

- [ ] **Step 2: Register them in `bench_phases`** (inside the `for &pos in &POSITIONS` loop, after the `full_local` entry):

```rust
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
```

- [ ] **Step 3: Smoke-run**

Run: `cargo bench -p inferno-kernels --bench attn_decode_phases -- --test`
Expected: exits 0, now also listing `phases/dot_only/*` and `phases/no_av/*`.

- [ ] **Step 4: Commit**

```bash
git add crates/inferno-kernels/benches/attn_decode_phases.rs
git commit -m "M4b.15 Task 2: dot_only / no_av marginal variants"
```

---

### Task 3: Counterfactual probes, roofline anchors, cold/warm axis

**Files:**
- Modify: `crates/inferno-kernels/benches/attn_decode_phases.rs`

**Interfaces:**
- Consumes: Tasks 1–2 helpers.
- Produces: `dot_pass_blocked::<NB>`, `av_pass_regacc`, variants `dot_blocked::<NB>` (NB ∈ 2,4,8), `av_regacc`, `combined`; criterion IDs `probes/dot_blocked{2,4,8}/<pos>`, `probes/av_regacc/<pos>`, `probes/combined/<pos>`, `anchors/kv_stream/<pos>`, `anchors/fma_peak`, `cold_warm/{cold,warm}/<pos>`. The probe loop bodies are EXACTLY what Tasks 5–6 promote into `attn_core_avx2`.

- [ ] **Step 1: Add the blocked dot pass and register-resident AV pass** (after `av_pass`):

```rust
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
        let kb = kv.add(kreg + t * KV_DIM + g * HEAD_DIM);
        let mut acc = [_mm256_setzero_ps(); NB];
        let mut d = 0;
        while d < HEAD_DIM {
            let qv = _mm256_loadu_ps(qh.add(d));
            for (j, a) in acc.iter_mut().enumerate() {
                let kvv = _mm256_loadu_ps(kb.add(j * KV_DIM + d));
                *a = _mm256_fmadd_ps(qv, kvv, *a);
            }
            d += 8;
        }
        for (j, a) in acc.iter().enumerate() {
            *scores.add(t + j) = bhsum8(*a) * scale;
        }
        t += NB;
    }
    // Tail: the original single-position loop.
    while t < visible {
        let kb = kv.add(kreg + t * KV_DIM + g * HEAD_DIM);
        let mut acc = _mm256_setzero_ps();
        let mut d = 0;
        while d < HEAD_DIM {
            let qv = _mm256_loadu_ps(qh.add(d));
            let kvv = _mm256_loadu_ps(kb.add(d));
            acc = _mm256_fmadd_ps(qv, kvv, acc);
            d += 8;
        }
        *scores.add(t) = bhsum8(acc) * scale;
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
        let wn = _mm256_set1_ps(*scores.add(t) / denom);
        let vb = kv.add(vreg + t * KV_DIM + g * HEAD_DIM);
        for (j, a) in acc.iter_mut().enumerate() {
            let vv = _mm256_loadu_ps(vb.add(j * 8));
            *a = _mm256_fmadd_ps(wn, vv, *a);
        }
    }
    for (j, a) in acc.iter().enumerate() {
        _mm256_storeu_ps(oh.add(j * 8), *a);
    }
}
```

- [ ] **Step 2: Add the probe variants** (after `no_av`):

```rust
/// Whole kernel with the blocked dot pass (Lever 1a candidate).
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
unsafe fn dot_blocked<const NB: usize>(b: &mut Buffers, pos: usize) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let group = N_HEADS / N_KV_HEADS;
    let visible = pos + 1;
    for h in 0..N_HEADS {
        let g = h / group;
        let qh = b.q.as_ptr().add(h * HEAD_DIM);
        dot_pass_blocked::<NB>(qh, b.kv.as_ptr(), 0, g, scale, visible, b.scores.as_mut_ptr());
        let max = max_pass(b.scores.as_ptr(), visible);
        let denom = exp_pass(b.scores.as_mut_ptr(), visible, max);
        av_pass(
            b.out.as_mut_ptr().add(h * HEAD_DIM),
            b.kv.as_ptr(),
            Buffers::V_OFF,
            g,
            b.scores.as_ptr(),
            denom,
            visible,
        );
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
        let qh = b.q.as_ptr().add(h * HEAD_DIM);
        dot_pass(qh, b.kv.as_ptr(), 0, g, scale, visible, b.scores.as_mut_ptr());
        let max = max_pass(b.scores.as_ptr(), visible);
        let denom = exp_pass(b.scores.as_mut_ptr(), visible, max);
        av_pass_regacc(
            b.out.as_mut_ptr().add(h * HEAD_DIM),
            b.kv.as_ptr(),
            Buffers::V_OFF,
            g,
            b.scores.as_ptr(),
            denom,
            visible,
        );
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
        let qh = b.q.as_ptr().add(h * HEAD_DIM);
        dot_pass_blocked::<4>(qh, b.kv.as_ptr(), 0, g, scale, visible, b.scores.as_mut_ptr());
        let max = max_pass(b.scores.as_ptr(), visible);
        let denom = exp_pass(b.scores.as_mut_ptr(), visible, max);
        av_pass_regacc(
            b.out.as_mut_ptr().add(h * HEAD_DIM),
            b.kv.as_ptr(),
            Buffers::V_OFF,
            g,
            b.scores.as_ptr(),
            denom,
            visible,
        );
    }
}
```

- [ ] **Step 3: Extend `assert_bit_identity`** — inside the `for &pos` loop, after the `full_local` check, add the same element-loop comparison for each probe:

```rust
        for (name, f) in [
            ("dot_blocked2", dot_blocked::<2> as unsafe fn(&mut Buffers, usize)),
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
```

- [ ] **Step 4: Add probe/anchor/cold-warm criterion groups** (new fns, and extend the `criterion_group!` list):

```rust
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
```

Change the group registration to:

```rust
criterion_group!(benches, bench_phases, bench_probes, bench_anchors, bench_cold_warm);
```

- [ ] **Step 5: Smoke-run + lint**

Run: `cargo bench -p inferno-kernels --bench attn_decode_phases -- --test && mise run lint`
Expected: exits 0 (all groups listed; bit-identity assert covers every probe), clippy clean. A probe divergence here means the probe code is wrong — fix the probe, never weaken the assert.

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-kernels/benches/attn_decode_phases.rs
git commit -m "M4b.15 Task 3: counterfactual probes, roofline anchors, cold/warm axis"
```

---

### Task 4: Quiet-hw wrapper script for the µbench

**Files:**
- Create: `scripts/quiet-hw/gate-decode-kernel-ubench.sh` (mode 755)

**Interfaces:**
- Consumes: `scripts/quiet-hw/lib.sh` (`smoke_header`, `machine_block`), the Task 1–3 bench.
- Produces: the session artifact both boxes record (criterion output + machine block, verbatim into §Amendments). Tasks 8–9 run it.

- [ ] **Step 1: Write the script** (pattern: `gate-decode-attr.sh`):

```bash
#!/usr/bin/env bash
# M4b.15 instrument — the phase-marginal decode-attention µbench on quiet
# hardware. Prints the machine block then the criterion output verbatim.
# VERDICTS ARE HUMAN: paste into the M4b.15 spec §Amendments and compute
# the marginals, admissibility, and (post-Lever-1) r there per the spec's
# pre-registered formulas. QHW_SMOKE=1 runs criterion's --test mode
# (plumbing check only, no numbers).
# Usage: gate-decode-kernel-ubench.sh   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

OUT="${QHW_OUT:-$(mktemp -d)}"
smoke_header "gate-decode-kernel-ubench (M4b.15 instrument: phase-marginal µbench)"
machine_block
echo

ARGS=()
if [ "${QHW_SMOKE:-0}" = 1 ]; then ARGS+=(--test); fi
cargo bench -p inferno-kernels --bench attn_decode_phases -- "${ARGS[@]}" \
  | tee "$OUT/attn-decode-phases.txt"
echo
echo "raw criterion data: target/criterion/ (collected by mise run metal)"
```

- [ ] **Step 2: Smoke it locally**

Run: `chmod +x scripts/quiet-hw/gate-decode-kernel-ubench.sh && QHW_SMOKE=1 bash scripts/quiet-hw/gate-decode-kernel-ubench.sh`
Expected: machine block prints, bench runs in `--test` mode, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/quiet-hw/gate-decode-kernel-ubench.sh
git commit -m "M4b.15 Task 4: gate-decode-kernel-ubench.sh session wrapper"
```

---

### Task 5: Local µbench run — record the table, admissibility, and Gates 1a/1b

**This task produces a VERDICT that gates Tasks 6 and 7. No shipping code changes here.**

**Files:**
- Modify: `docs/superpowers/specs/2026-07-17-m4b15-decode-attention-kernel-design.md` (§Amendments only)

- [ ] **Step 1: Full local run**

Run: `cargo bench -p inferno-kernels --bench attn_decode_phases 2>&1 | tee /tmp/m4b15-task5.txt`
Expected: bit-identity assert passes, all groups produce estimates (~10–20 min).

- [ ] **Step 2: Compute and record.** Append to the spec's §Amendments a dated entry `### YYYY-MM-DD — Task 5: local µbench + Lever 1 gate verdicts (non-quiet dev box)` containing, verbatim per the spec's pre-registered rules:
  1. The machine (CPU model, core count, "NON-QUIET dev box" label) and the criterion mean times for every benchmark ID.
  2. **Admissibility:** at each pos — monotonicity `dot_only ≤ no_av ≤ full_local`; marginals `dot = dot_only`, `softmax = no_av − dot_only`, `AV = full_local − no_av`, each ≥ 0, sum within ±10% of `full_local`; `full_local` within ±5% of `full`. Any failure → **STOP (instrument finding)**: record which check failed and skip to Task 8 (the sessions still run to record the diagnostic); Tasks 6–7 are SKIPPED.
  3. **Gate 1a:** dot marginal ≥ 15% of `full_local` at pos 511 AND 639; best `dot_blockedN` (N over 2/4/8) beats `full_local` by ≥ 10% whole-call at both. Record the winning N. PASS/STOP.
  4. **Gate 1b:** AV marginal ≥ 15% at both; `av_regacc` ≥ 10% whole-call at both. PASS/STOP.
  5. **Softmax escape check:** if the softmax marginal exceeds both other marginals at pos ≥ 511 → Lever 1 STOPs entirely (spec §Pre-registered gates); Tasks 6–7 SKIPPED.
  6. Roofline context: each phase marginal vs the `kv_stream` and `fma_peak` anchors; cold/warm delta. (Context for the record, feeds no gate.)

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-07-17-m4b15-decode-attention-kernel-design.md
git commit -m "specs: M4b.15 Task 5 — local µbench table, admissibility, Lever 1 gate verdicts"
```

---

### Task 6: Lever 1a — blocked QK dot in `attn_core_avx2` (SKIP unless Gate 1a PASS)

**Files:**
- Modify: `crates/inferno-kernels/src/attention.rs` (`attn_core_avx2` scores loop only)
- Test: `crates/inferno-kernels/tests/rig.rs` (new edge-case test)

**Interfaces:**
- Consumes: Gate 1a's winning N from Task 5.
- Produces: `const DOT_BLOCK: usize` in `attention.rs`; the public symbols' signatures and outputs are UNCHANGED (bit-identical).

- [ ] **Step 1: Write the invariant test FIRST and confirm it passes against the CURRENT kernel.** This is the rig's exact-equality discipline, not red-green TDD — the test encodes an invariant the change must preserve, so it must pass both before and after (a "fail first" here would mean the current kernel is broken). Add to `crates/inferno-kernels/tests/rig.rs`, after the `mod attention_hspan` block:

```rust
mod attention_m4b15_edges {
    //! M4b.15: the blocked-dot / register-AV kernel must stay bitwise
    //! equal to scalar at the protocol head_dim (64 — the register-AV
    //! path) across visible counts straddling every block boundary.
    use super::{attn_kernel_avx2, attn_kernel_scalar};

    #[test]
    fn avx2_bitwise_matches_scalar_at_block_boundary_visibles() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let (n_heads, n_kv_heads, head_dim) = (14usize, 2usize, 64usize);
        let kv_dim = n_kv_heads * head_dim;
        for pos in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65] {
            let seq_len = pos + 1;
            let mut q = vec![0f32; n_heads * head_dim];
            let mut k = vec![0f32; kv_dim];
            let mut v = vec![0f32; kv_dim];
            let mut fill = |seed: u64, buf: &mut [f32]| {
                let mut s = seed;
                for x in buf.iter_mut() {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    *x = ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
                }
            };
            fill(0x4b15 + pos as u64, &mut q);
            fill(0x4b16 + pos as u64, &mut k);
            fill(0x4b17 + pos as u64, &mut v);
            let mut kv_s = vec![0f32; 2 * seq_len * kv_dim];
            let mut kv_a = kv_s.clone();
            // Backfill history rows so every visible position is nonzero.
            fill(0x4b18 + pos as u64, &mut kv_s[..pos * kv_dim]);
            fill(0x4b19 + pos as u64, &mut kv_s[seq_len * kv_dim..seq_len * kv_dim + pos * kv_dim]);
            kv_a.copy_from_slice(&kv_s);
            let want = attn_kernel_scalar(
                &q, &k, &v, &mut kv_s, seq_len, pos, kv_dim, n_heads, n_kv_heads, head_dim,
            );
            let got = attn_kernel_avx2(
                &q, &k, &v, &mut kv_a, seq_len, pos, kv_dim, n_heads, n_kv_heads, head_dim,
            );
            for (i, (w, g)) in want.iter().zip(&got).enumerate() {
                assert_eq!(w.to_bits(), g.to_bits(), "pos={pos} idx={i}");
            }
        }
    }
}
```

(If `attn_kernel_scalar` / `attn_kernel_avx2` are not visible from a child module, hoist the calls — match how `attention_hspan` resolves its imports in the existing file.)

- [ ] **Step 2: Run it — must PASS (pre-change)**

Run: `cargo test -p inferno-kernels --test rig attention_m4b15_edges -- --nocapture`
Expected: PASS.

- [ ] **Step 3: Apply the kernel change.** In `crates/inferno-kernels/src/attention.rs`, add at module scope (near the top, after the imports):

```rust
/// M4b.15 Lever 1a: query positions in flight per scores-pass step.
/// Value = the Gate 1a winner (spec §Amendments, Task 5). Independent
/// accumulators break the FMA latency chain; each position's fmadd order
/// and hsum8 tree are exactly the single-position loop's, so scores are
/// bit-identical position by position.
const DOT_BLOCK: usize = 4; // ← replace with the Gate 1a winning N
```

Then in `attn_core_avx2`, replace the scores loop:

```rust
            // scores[t] = reduce8(sum_d qh[d]*kcache) * scale
            for t in 0..visible {
                let kb = kv.add(kreg + t * kv_dim + g * head_dim);
                let mut acc = _mm256_setzero_ps();
                let mut d = 0;
                while d < head_dim {
                    let qv = _mm256_loadu_ps(qh.add(d));
                    let kvv = _mm256_loadu_ps(kb.add(d));
                    acc = _mm256_fmadd_ps(qv, kvv, acc);
                    d += 8;
                }
                *scores.add(t) = hsum8(acc) * scale;
            }
```

with:

```rust
            // scores[t] = reduce8(sum_d qh[d]*kcache) * scale — DOT_BLOCK
            // positions in flight (M4b.15 Lever 1a); per-position order
            // unchanged, so scores are bit-identical.
            let mut t = 0;
            while t + DOT_BLOCK <= visible {
                let kb = kv.add(kreg + t * kv_dim + g * head_dim);
                let mut acc = [_mm256_setzero_ps(); DOT_BLOCK];
                let mut d = 0;
                while d < head_dim {
                    let qv = _mm256_loadu_ps(qh.add(d));
                    for (j, a) in acc.iter_mut().enumerate() {
                        let kvv = _mm256_loadu_ps(kb.add(j * kv_dim + d));
                        *a = _mm256_fmadd_ps(qv, kvv, *a);
                    }
                    d += 8;
                }
                for (j, a) in acc.iter().enumerate() {
                    *scores.add(t + j) = hsum8(*a) * scale;
                }
                t += DOT_BLOCK;
            }
            while t < visible {
                let kb = kv.add(kreg + t * kv_dim + g * head_dim);
                let mut acc = _mm256_setzero_ps();
                let mut d = 0;
                while d < head_dim {
                    let qv = _mm256_loadu_ps(qh.add(d));
                    let kvv = _mm256_loadu_ps(kb.add(d));
                    acc = _mm256_fmadd_ps(qv, kvv, acc);
                    d += 8;
                }
                *scores.add(t) = hsum8(acc) * scale;
                t += 1;
            }
```

Do NOT touch `attn_core_scalar`, the qblock kernels, or any signature.

- [ ] **Step 4: Full verification sweep**

Run, in order; ALL must pass with zero tolerance edits:
```
cargo test -p inferno-kernels --test rig
cargo bench -p inferno-kernels --bench attn_decode_phases -- --test   # bit-identity assert vs frozen copy
cargo test -p inferno-codegen --test differential
cargo test -p inferno-core --test artifact
mise run test && mise run lint
```
Expected: all green. The bench's `assert_bit_identity` now checks the NEW shipping kernel against the frozen pre-lever copy — that is the Lever 1 bit-neutrality proof running on every future bench invocation.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/src/attention.rs crates/inferno-kernels/tests/rig.rs
git commit -m "M4b.15 Lever 1a: blocked QK dot in attn_core_avx2 (bit-identical, gate-authorized)"
```

---

### Task 7: Lever 1b — register-resident AV in `attn_core_avx2` (SKIP unless Gate 1b PASS)

**Files:**
- Modify: `crates/inferno-kernels/src/attention.rs` (`attn_core_avx2` AV loop only)

**Interfaces:**
- Consumes: Task 6's edge test (already committed if Gate 1a passed; if Task 6 was skipped, do its Steps 1–2 here first — the test is lever-independent).
- Produces: unchanged public symbols, bit-identical outputs.

- [ ] **Step 1: Ensure the edge test from Task 6 Step 1 exists and passes.** If Task 6 was skipped, add it now (same code, same commit discipline) and run it: must PASS pre-change.

- [ ] **Step 2: Apply the kernel change.** In `attn_core_avx2`, replace the AV section:

```rust
            // AV: oh[d] += (scores[t]/denom) * vcache
            let oh = out.add(h * head_dim);
            for d in (0..head_dim).step_by(8) {
                _mm256_storeu_ps(oh.add(d), _mm256_setzero_ps());
            }
            for t in 0..visible {
                let wn = _mm256_set1_ps(*scores.add(t) / denom);
                let vb = kv.add(vreg + t * kv_dim + g * head_dim);
                for d in (0..head_dim).step_by(8) {
                    let cur = _mm256_loadu_ps(oh.add(d));
                    let vv = _mm256_loadu_ps(vb.add(d));
                    _mm256_storeu_ps(oh.add(d), _mm256_fmadd_ps(wn, vv, cur));
                }
            }
```

with:

```rust
            // AV: oh[d] += (scores[t]/denom) * vcache — out row held in
            // YMM registers per 64-float chunk, stored once (M4b.15 Lever
            // 1b). Per-element accumulation stays t-ascending, so the row
            // is bit-identical to the store-reload loop; wn is a pure
            // function of (scores[t], denom), so recomputing it per chunk
            // changes nothing. head_dim < 64 (rig shapes) takes the tail
            // branch, which IS the old loop.
            let oh = out.add(h * head_dim);
            let mut d0 = 0;
            while d0 + 64 <= head_dim {
                let mut acc = [_mm256_setzero_ps(); 8];
                for t in 0..visible {
                    let wn = _mm256_set1_ps(*scores.add(t) / denom);
                    let vb = kv.add(vreg + t * kv_dim + g * head_dim + d0);
                    for (j, a) in acc.iter_mut().enumerate() {
                        let vv = _mm256_loadu_ps(vb.add(j * 8));
                        *a = _mm256_fmadd_ps(wn, vv, *a);
                    }
                }
                for (j, a) in acc.iter().enumerate() {
                    _mm256_storeu_ps(oh.add(d0 + j * 8), *a);
                }
                d0 += 64;
            }
            if d0 < head_dim {
                for d in (d0..head_dim).step_by(8) {
                    _mm256_storeu_ps(oh.add(d), _mm256_setzero_ps());
                }
                for t in 0..visible {
                    let wn = _mm256_set1_ps(*scores.add(t) / denom);
                    let vb = kv.add(vreg + t * kv_dim + g * head_dim);
                    for d in (d0..head_dim).step_by(8) {
                        let cur = _mm256_loadu_ps(oh.add(d));
                        let vv = _mm256_loadu_ps(vb.add(d));
                        _mm256_storeu_ps(oh.add(d), _mm256_fmadd_ps(wn, vv, cur));
                    }
                }
            }
```

- [ ] **Step 3: Full verification sweep** — same five commands as Task 6 Step 4, all green, zero tolerance edits. Pay attention to the bench `--test` run: the `assert_bit_identity` against the frozen copy is the proof.

- [ ] **Step 4: Commit**

```bash
git add crates/inferno-kernels/src/attention.rs crates/inferno-kernels/tests/rig.rs
git commit -m "M4b.15 Lever 1b: register-resident AV in attn_core_avx2 (bit-identical, gate-authorized)"
```

---

### Task 8: Local data point (post-Lever-1)

**Files:**
- Modify: `docs/superpowers/specs/2026-07-17-m4b15-decode-attention-kernel-design.md` (§Amendments only)

- [ ] **Step 1: µbench re-run.** `cargo bench -p inferno-kernels --bench attn_decode_phases 2>&1 | tee /tmp/m4b15-task8.txt` — `full` is now the shipped kernel; `full_local` the frozen baseline. Compute local `r = 1 − full/full_local` averaged over pos {511, 639}. If BOTH levers shipped, also check the spec's sanity rule: `combined` (≈ the new `full`) beats each single probe; if it regresses against the better single probe, record it — the spec says the better single lever ships alone (revert the other lever's commit, re-run Task 6/7 verification, record the revert).

- [ ] **Step 2: Local e2e data point (context only, non-quiet).**

```bash
MODEL=$(bash scripts/fetch-qwen-gguf.sh)
cargo run --release -p inferno -- bench "$MODEL"
```

- [ ] **Step 3: Record** both under a dated `Task 8 — local post-Lever-1 data point (non-quiet)` amendment: the µbench table, local `r`, the e2e numbers, honestly labeled. Commit:

```bash
git add docs/superpowers/specs/2026-07-17-m4b15-decode-attention-kernel-design.md
git commit -m "specs: M4b.15 Task 8 — local post-Lever-1 data point (non-quiet)"
```

- [ ] **Step 4: Push the branch** (sessions clone committed HEAD): `git push -u origin m4b15-design`

---

### Task 9: Quiet-hw session A — 16c `d2.c1.medium` (6336Y)

**Files:**
- Modify: `docs/superpowers/specs/2026-07-17-m4b15-decode-attention-kernel-design.md` (§Amendments), `docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md` (§Amendments, protocol data points only)

**Session inputs fixed BEFORE provisioning:** `BASE_SHA` = the main-branch commit Lever 1 branched from (baseline binary); `LEVER_SHA` = the pushed Task 8 HEAD. Both must be reachable from origin.

- [ ] **Step 1: Provision + run the workload.** One session, one box (never parallel with session B). The workload runs both checkouts in one provisioning:

```bash
mise run metal -- d2.c1.medium --yes -- '
  set -euo pipefail
  command -v perf >/dev/null || sudo apt-get install -y linux-perf || true
  command -v perf >/dev/null || echo "DEVIATION: perf unavailable (record in amendment)"
  MODEL=$(bash scripts/fetch-qwen-gguf.sh)
  export QHW_OUT=target/quiet-hw
  bash scripts/quiet-hw/preflight.sh
  echo "=== BASELINE (BASE_SHA) ==="
  git fetch origin <BASE_SHA> 2>/dev/null || git fetch --unshallow || true
  git checkout <BASE_SHA>
  bash scripts/quiet-hw/gate-bench-protocol.sh "$MODEL"
  bash scripts/quiet-hw/gate-decode-attr.sh "$MODEL"
  echo "=== LEVER (LEVER_SHA) ==="
  git checkout <LEVER_SHA>
  bash scripts/quiet-hw/gate-decode-kernel-ubench.sh
  bash scripts/quiet-hw/gate-bench-protocol.sh "$MODEL"
  bash scripts/quiet-hw/gate-attn-split.sh "$MODEL"
'
```

(Substitute the two SHAs literally. `gate-decode-attr.sh` gives the baseline t=1/best-t op-table profiles → `S`; `gate-attn-split.sh` gives the post-Lever-1 dispatch-split profile → Gate 2's `c`; the µbench gives `r` on this box.)

- [ ] **Step 2: On ANY failure:** `mise run metal-gc` and confirm zero servers before retrying. Transient devpod post-create panics: retry once (known quirk).

- [ ] **Step 3: Record session A** in the spec §Amendments (dated `Session A — d2.c1.medium`): machine block, perf deviation if any, verbatim script outputs from `target/metal/<...>/workload.log`, then the human-computed numbers per the spec: `S` (baseline best-t attention share), `r` (µbench, pos 511/639 mean), `baseline_tg`, **the headroom target `tg_target = baseline_tg × (1 + S × r)`**, lever protocol tg/pp, µbench admissibility on this box, dispatch-split drain fraction (Gate 2's `c`), and `S_residual` (post-Lever-1). Protocol data points also go verbatim into the M4a spec §Amendments (standing rule).

- [ ] **Step 4: Commit + push**

```bash
git add docs/superpowers/specs/
git commit -m "specs: M4b.15 session A (16c) — µbench, profiles, headroom target, Gate 2 inputs"
git push
```

---

### Task 10: Quiet-hw session B — 8c `s2.c2.medium` (E-2388G)

Same as Task 9 with `s2.c2.medium`, recorded as `Session B — s2.c2.medium`. Remember: no parallel provisions; on a 406 check `mise run metal-catalog` stock and pass `--location`; `mise run metal-gc` after any failure.

- [ ] **Step 1: Provision + run** (Task 9 Step 1 workload, `s2.c2.medium`)
- [ ] **Step 2: Record session B** (Task 9 Step 3 content, per-box numbers)
- [ ] **Step 3: Commit + push** (message: `specs: M4b.15 session B (8c) — µbench, profiles, headroom target, Gate 2 inputs`)

---

### Task 11: Gate 2 verdict, closing verdict, PR

**Files:**
- Modify: `docs/superpowers/specs/2026-07-17-m4b15-decode-attention-kernel-design.md` (§Amendments only)

- [ ] **Step 1: Gate 2 verdict.** Compute once from sessions A+B, arithmetic shown, per the spec: `P2 = S_residual × c` per machine; thresholds M4b.11 verbatim (≥5% both → authorize; <3% both → STOP; split → judgment call with the argument recorded — the spec pre-commits what the judgment must weigh). Record as a dated `Gate 2 verdict` amendment.

- [ ] **Step 2: If Gate 2 AUTHORIZED:** STOP this plan after recording the verdict. Lever 2 (KV-position-split kernel) requires its own implementation plan — return to superpowers:writing-plans against the spec's §Lever 2 numerics pre-commitments (tolerance re-derivation before any claim, new pool entry point, ABI/cache-key assessment, follow-up session pair). Do not improvise it from this plan.

- [ ] **Step 3: If Gate 2 STOP (or split-to-no): closing verdict.** Append the exit-criteria walk (spec §Exit criteria, all five items, YES/NO each with pointers to the amendments): µbench + admissibility on record (local + both boxes); every gate verdict recorded once; invariants held (link the green differential/rig runs, confirm zero tolerance diffs: `git log -p --  crates/inferno-graph/src/tolerance.rs` shows no M4b.15 change); closing tg vs the per-box headroom targets (MET / NOT MET per box); every STOP recorded. v1 ratios go to the M4a spec §Amendments as context. If the 8c residual again blocks tg ≥ 1.0x, the closing verdict MUST state what the residual decode wall is shaped like (per the spec §Risks — the next milestone's scoping input).

- [ ] **Step 4: Final verification + PR**

```bash
mise run test && mise run lint
git push
gh pr create --title "M4b.15: decode attention kernel quality (phase-marginal µbench + gated bit-neutral levers)" \
  --body "Spec + instrument + gate-authorized Lever 1 + session records + verdicts. See docs/superpowers/specs/2026-07-17-m4b15-decode-attention-kernel-design.md §Amendments for the gate arithmetic."
```

Expected: CI green (remember: clippy runs in CI via `mise run lint`; FlakeHub 'path is not valid' failures → rerun the job once before debugging).

---

## Plan Self-Review (performed at write time)

- **Spec coverage:** instrument (Tasks 1–4), local gates (Task 5), Lever 1a/1b with invariants (Tasks 6–7), local data point (Task 8), sessions incl. `perf` requirement + headroom targets (Tasks 9–10), Gate 2 + closing + Lever 2 handoff (Task 11). Spec Task 8's "follow-up session pair for Lever 2" is inside the Lever 2 addendum plan (Task 11 Step 2) by design.
- **Conditionality:** Tasks 6/7 carry explicit SKIP conditions keyed to Task 5's recorded verdicts; Task 11 Step 2 handles the authorized-Lever-2 branch without placeholder implementation.
- **Type consistency:** bench helpers (`Buffers`, `full_local(b, pos, h0, h1)`, `dot_pass_blocked::<NB>`, `av_pass_regacc`) are used with the same signatures in Tasks 1–3; kernel edits in Tasks 6–7 quote the exact current code from `attention.rs` as the old_string anchor.
