//! Attention kernel µbench (M4b.3): scalar vs avx2 over the pinned model's
//! head geometry at representative causal horizons, so the SIMD win is
//! visible per-position. Throughput unit: elements (n_heads * head_dim per
//! call). Numbers only meaningful from `mise run bench-kernels` on quiet HW.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use inferno_kernels::{attention_kernel, attention_reference};
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

criterion_group!(benches, bench_attention);
criterion_main!(benches);
