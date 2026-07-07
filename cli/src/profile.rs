//! `--profile` output: per-op cycle totals, wall-clock share, and (for
//! matmul sites) achieved GB/s. Self-measurement only; never a CI gate.

/// Render a profile table. `counts[i]` is slot `i`'s accumulated cycles;
/// `bytes[i]` is the weight bytes touched per matmul slot invocation × the
/// invocation count (0 for non-matmul slots), used for the GB/s column.
/// `secs` is the measured wall-clock for this phase (prefill or decode),
/// used to convert the cycle share into GB/s without knowing the TSC rate.
pub fn render(phase: &str, slots: &[String], counts: &[u64], bytes: &[u64], secs: f64) -> String {
    use std::fmt::Write;
    let total: u64 = counts.iter().sum();
    let mut rows: Vec<usize> = (0..slots.len()).collect();
    rows.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
    let mut s = String::new();
    writeln!(s, "profile [{phase}] {secs:.3}s wall, {total} cyc total").unwrap();
    writeln!(
        s,
        "  {:<28} {:>14} {:>7}  {:>10}",
        "op", "cycles", "share", "GB/s"
    )
    .unwrap();
    for i in rows {
        let share = if total > 0 {
            counts[i] as f64 / total as f64
        } else {
            0.0
        };
        // Time attributed to this op = its cycle share of the phase wall-clock.
        let op_secs = share * secs;
        let gbps = if bytes[i] > 0 && op_secs > 0.0 {
            bytes[i] as f64 / op_secs / 1e9
        } else {
            0.0
        };
        let gbps_col = if gbps > 0.0 {
            format!("{gbps:.1}")
        } else {
            "-".into()
        };
        writeln!(
            s,
            "  {:<28} {:>14} {:>6.1}%  {:>10}",
            slots[i],
            counts[i],
            share * 100.0,
            gbps_col
        )
        .unwrap();
    }
    s
}

#[cfg(test)]
mod tests {
    #[test]
    fn render_sorts_and_computes_share() {
        // rmsnorm is declared *after* matmul in `slots` but has more cycles
        // (300 vs 100). If `render()` didn't sort by descending cycle count,
        // the rmsnorm row would still come second (input order). Asserting
        // it comes first below only passes because the sort actually ran.
        let slots = vec![
            "matmul:blk.*.attn_q.weight".to_string(),
            "rmsnorm".to_string(),
        ];
        let out = super::render("decode", &slots, &[100, 300], &[6_000_000, 0], 0.5);
        assert!(out.contains("matmul:blk.*.attn_q.weight"));

        // The first data row (after the two header lines) must be rmsnorm:
        // proof the higher-cycle op was sorted ahead of matmul.
        let lines: Vec<&str> = out.lines().collect();
        assert!(
            lines[2].contains("rmsnorm"),
            "expected rmsnorm (300 cyc) sorted before matmul (100 cyc), got: {}",
            lines[2]
        );

        assert!(out.contains("25.0%")); // matmul: 100/400
        assert!(out.contains("75.0%")); // rmsnorm: 300/400

        // matmul row shows a GB/s number; rmsnorm shows '-'.
        let mm_line = out.lines().find(|l| l.contains("attn_q")).unwrap();
        assert!(mm_line.contains('.') && !mm_line.contains(" - "));
        let rms_line = out.lines().find(|l| l.contains("rmsnorm")).unwrap();
        assert!(rms_line.trim_end().ends_with('-'));
    }
}
