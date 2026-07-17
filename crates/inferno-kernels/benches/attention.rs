//! Attention kernel µbench (M4b.3): scalar vs avx2 over the pinned model's
//! head geometry at representative causal horizons, so the SIMD win is
//! visible per-position. Throughput unit: elements (n_heads * head_dim per
//! call). Numbers only meaningful from `mise run bench-kernels` on quiet HW.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use inferno_kernels::{
    KernelIsa, attention_kernel, attention_reference, inferno_attention_f32_avx2,
    inferno_attention_f32_avx2_qblock,
};
use inferno_target::Isa;

fn bench_attention(c: &mut Criterion) {
    let (n_heads, n_kv_heads, head_dim) = (14usize, 2usize, 64usize);
    let kv_dim = n_kv_heads * head_dim;
    let seq_len = 512usize;
    let mut group = c.benchmark_group("attention");
    for &pos in &[15usize, 127, 511] {
        let mut kv = vec![0.1f32; 2 * seq_len * kv_dim];
        let q = vec![0.05f32; n_heads * head_dim];
        let mut out = vec![0f32; n_heads * head_dim];
        let mut scores = vec![0f32; seq_len];
        group.throughput(Throughput::Elements((n_heads * head_dim) as u64));
        for (name, f) in [
            ("scalar", attention_reference()),
            ("avx2", attention_kernel(Isa::X86_64v3)),
        ] {
            group.bench_with_input(BenchmarkId::new(name, pos), &pos, |b, &pos| {
                b.iter(|| unsafe {
                    f(
                        out.as_mut_ptr(),
                        q.as_ptr(),
                        kv.as_mut_ptr(),
                        scores.as_mut_ptr(),
                        0,
                        seq_len * kv_dim,
                        pos,
                        kv_dim,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                    )
                });
            });
        }
    }
    group.finish();
}

/// Query-blocked vs per-token AVX2 attention (M4b.14 Task 7). Same
/// `m_block`-row block of the criterion model's shape, computed two ways:
/// `per_token` issues `m_block` back-to-back [`inferno_attention_f32_avx2`]
/// calls (today's decode-loop driving pattern, one call per row at
/// `pos0+r`), `qblock` issues one [`inferno_attention_f32_avx2_qblock`]
/// call over the whole block (each visible K/V vector loaded once and
/// reused across rows). `m_block = 64` is the prefill tile size. `kv`/`q`/
/// `out` are allocated once per `pos0`, pre-filled outside the timed loop,
/// and shared by both variants so they read/write identical memory over
/// identical extents; only the scores scratch differs, each sized to its
/// own kernel's contract (`s` for per-token, `m_block * s` for qblock,
/// where `s = pos0 + m_block` is the KV cache's filled extent). Throughput
/// unit: elements per block (`m_block * n_heads * head_dim`), so ns/iter is
/// directly comparable between the two variants. AVX2+FMA-gated at runtime
/// (mirrors the raw-symbol guard in `gemv.rs`'s `reduce-ceiling` arm) since,
/// unlike `attention_kernel`, these are the raw ISA-specific symbols with no
/// scalar-fallback wrapper.
fn bench_attention_qblock(c: &mut Criterion) {
    if !KernelIsa::Avx2.available() {
        return;
    }
    let (n_heads, n_kv_heads, head_dim) = (14usize, 2usize, 64usize);
    let kv_dim = n_kv_heads * head_dim;
    let m_block = 64usize;
    let q_stride = n_heads * head_dim;
    let out_stride = n_heads * head_dim;
    let mut group = c.benchmark_group("attention_qblock");
    for &pos0 in &[64usize, 256, 512] {
        let s = pos0 + m_block;
        let v_off = s * kv_dim;
        let mut kv = vec![0.1f32; 2 * s * kv_dim];
        let q = vec![0.05f32; m_block * q_stride];
        let mut out = vec![0f32; m_block * out_stride];
        // Per-token contract: scores valid for `pos+1`; `s` covers the
        // worst case (the block's last row, `pos = pos0+m_block-1`).
        let mut scores_pt = vec![0f32; s];
        // qblock contract: scores valid for `m_block * (pos0+m_block)`.
        let mut scores_qb = vec![0f32; m_block * s];
        group.throughput(Throughput::Elements((m_block * n_heads * head_dim) as u64));

        group.bench_with_input(BenchmarkId::new("per_token", pos0), &pos0, |b, &pos0| {
            b.iter(|| {
                for r in 0..m_block {
                    // SAFETY: `out`/`q` windows at `r*{out,q}_stride` are
                    // disjoint, in-bounds `n_heads*head_dim` extents (strides
                    // equal that extent, buffers sized `m_block*stride`);
                    // `kv` holds every position `< s` for both K (`kv_base
                    // 0`) and V (`v_off`); `scores_pt` is valid for
                    // `pos+1 <= s`; this host has AVX2+FMA (checked above).
                    unsafe {
                        inferno_attention_f32_avx2(
                            out.as_mut_ptr().add(r * out_stride),
                            q.as_ptr().add(r * q_stride),
                            kv.as_mut_ptr(),
                            scores_pt.as_mut_ptr(),
                            0,
                            v_off,
                            pos0 + r,
                            kv_dim,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                        )
                    }
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("qblock", pos0), &pos0, |b, &pos0| {
            b.iter(|| {
                // SAFETY: `out`/`q` valid for `(m_block-1)*stride +
                // n_heads*head_dim` (buffers sized `m_block*stride`); `kv`
                // holds every position `< pos0+m_block`; `scores_qb` sized
                // `m_block*s`; `pos0+m_block == s <= seq_len` (the KV cache's
                // filled extent); this host has AVX2+FMA (checked above).
                unsafe {
                    inferno_attention_f32_avx2_qblock(
                        out.as_mut_ptr(),
                        q.as_ptr(),
                        kv.as_mut_ptr(),
                        scores_qb.as_mut_ptr(),
                        0,
                        v_off,
                        pos0,
                        m_block,
                        kv_dim,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        q_stride,
                        out_stride,
                    )
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_attention, bench_attention_qblock);
criterion_main!(benches);
