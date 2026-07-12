//! Static shard partitioning: pure math, no threads. The shard map is a
//! deterministic function of `(rows, threads)` — boundaries align to the
//! kernels' 8-row strip so AVX2 strips are never split across threads, and
//! only the final shard may end off-alignment (at `rows` itself). Attention
//! shards whole tokens with align-1 (M4b.8).

/// Shard boundary alignment in rows. Must equal `inferno_kernels::STRIP`
/// (asserted by a test in `tests/par_rig.rs`); duplicated here so the pool
/// has no runtime dependency on the kernels crate.
pub const SHARD_ALIGN: usize = 8;

/// Split `0..rows` into at most `threads` contiguous shards whose internal
/// boundaries are multiples of `align.max(1)`. Strips are distributed as
/// evenly as possible (earlier shards get the remainder strips); the final
/// (possibly partial) strip lands in the last shard. `rows == 0` yields no
/// shards; `threads == 0` is treated as 1. GEMV/GEMM shard with
/// [`SHARD_ALIGN`] (AVX2 strips must not split); attention shards whole
/// tokens with `align = 1` (M4b.8 — 8-alignment would cap a 64-token tile
/// at 8 shards).
pub fn shard_table_aligned(rows: usize, threads: usize, align: usize) -> Vec<(usize, usize)> {
    let align = align.max(1);
    if rows == 0 {
        return Vec::new();
    }
    let strips = rows.div_ceil(align);
    let n = threads.max(1).min(strips);
    let base = strips / n;
    let extra = strips % n;
    let mut out = Vec::with_capacity(n);
    let mut strip = 0usize;
    for i in 0..n {
        let take = base + usize::from(i < extra);
        let start = strip * align;
        strip += take;
        out.push((start, (strip * align).min(rows)));
    }
    out
}

/// Split `0..rows` on [`SHARD_ALIGN`] boundaries — the GEMV/GEMM shard map.
pub fn shard_table(rows: usize, threads: usize) -> Vec<(usize, usize)> {
    shard_table_aligned(rows, threads, SHARD_ALIGN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rows_yields_no_shards() {
        assert!(shard_table(0, 8).is_empty());
    }

    #[test]
    fn exact_split() {
        assert_eq!(shard_table(16, 2), vec![(0, 8), (8, 16)]);
    }

    #[test]
    fn partial_final_strip_goes_to_last_shard() {
        // 20 rows = 3 strips (8, 8, 4); 4 threads clamp to 3 shards.
        assert_eq!(shard_table(20, 4), vec![(0, 8), (8, 16), (16, 20)]);
    }

    #[test]
    fn fewer_strips_than_threads_collapses() {
        assert_eq!(shard_table(7, 12), vec![(0, 7)]);
    }

    #[test]
    fn threads_zero_behaves_as_one() {
        assert_eq!(shard_table(100, 0), vec![(0, 100)]);
    }

    /// Exhaustive structural properties over a grid: shards tile `0..rows`
    /// contiguously, every internal boundary is 8-aligned, shard count is
    /// `min(threads, ceil(rows/8))`, and the map is deterministic.
    #[test]
    fn structural_properties_hold_on_grid() {
        for rows in (0..2048usize).step_by(7) {
            for threads in 1..=16usize {
                let s = shard_table(rows, threads);
                assert_eq!(s, shard_table(rows, threads), "determinism");
                if rows == 0 {
                    assert!(s.is_empty());
                    continue;
                }
                assert_eq!(s.len(), threads.min(rows.div_ceil(SHARD_ALIGN)));
                assert_eq!(s[0].0, 0);
                assert_eq!(s.last().unwrap().1, rows);
                for w in s.windows(2) {
                    assert_eq!(w[0].1, w[1].0, "contiguous");
                    assert_eq!(w[0].1 % SHARD_ALIGN, 0, "aligned boundary");
                }
                for &(a, b) in &s {
                    assert!(a < b, "non-empty shard");
                }
            }
        }
    }

    #[test]
    fn align1_uses_all_threads_on_a_prefill_tile() {
        // The M4b.8 motivating case: a 64-token tile across 12 lanes must
        // yield 12 shards (8-aligned sharding would collapse it to 8).
        let s = shard_table_aligned(64, 12, 1);
        assert_eq!(s.len(), 12);
        assert_eq!(s[0], (0, 6)); // 64 = 12*5 + 4 → first 4 shards get 6
        assert_eq!(s.last().unwrap().1, 64);
    }

    #[test]
    fn align1_threads_exceeding_rows_collapses_to_rows() {
        let s = shard_table_aligned(3, 16, 1);
        assert_eq!(s, vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn shard_table_wrapper_is_align8() {
        for rows in [1, 7, 20, 64, 1000] {
            for threads in 1..=16 {
                assert_eq!(
                    shard_table(rows, threads),
                    shard_table_aligned(rows, threads, SHARD_ALIGN)
                );
            }
        }
    }

    /// The structural grid, generalized over align ∈ {1, 8}: shards tile
    /// `0..rows` contiguously, internal boundaries are align-multiples,
    /// shard count is `min(threads, ceil(rows/align))`, deterministic.
    #[test]
    fn structural_properties_hold_on_grid_for_align1() {
        for rows in (0..2048usize).step_by(7) {
            for threads in 1..=16usize {
                let s = shard_table_aligned(rows, threads, 1);
                assert_eq!(s, shard_table_aligned(rows, threads, 1), "determinism");
                if rows == 0 {
                    assert!(s.is_empty());
                    continue;
                }
                assert_eq!(s.len(), threads.min(rows));
                assert_eq!(s[0].0, 0);
                assert_eq!(s.last().unwrap().1, rows);
                for w in s.windows(2) {
                    assert_eq!(w[0].1, w[1].0, "contiguous");
                }
                for &(a, b) in &s {
                    assert!(a < b, "non-empty shard");
                }
            }
        }
    }
}
