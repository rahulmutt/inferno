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

/// Render the M4b.12 `pool [decode attention]` dispatch-split section.
/// `op_attention_cyc` is the op profiler's decode attention cycle count,
/// for the sum-identity admissibility line (spec: within 10%). Cycle
/// numbers are printed raw — decode-wall shares and gate arithmetic are
/// controller work in the spec's Amendments, never computed here.
pub fn render_pool(s: &inferno_pool::PoolProfSnapshot, op_attention_cyc: u64) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let total = s.instr_total();
    writeln!(
        out,
        "pool [decode attention] {} calls, {} cyc instrumented",
        s.calls, total
    )
    .unwrap();
    let identity = if op_attention_cyc > 0 {
        total as f64 / op_attention_cyc as f64 * 100.0
    } else {
        0.0
    };
    writeln!(
        out,
        "  sum identity vs op-profiler attention: {identity:.1}% (admissible: 90-110%)"
    )
    .unwrap();
    let share = |c: u64| {
        if total > 0 {
            c as f64 / total as f64 * 100.0
        } else {
            0.0
        }
    };
    writeln!(out, "  {:<16} {:>14} {:>7}", "bucket", "cycles", "share").unwrap();
    for (name, c) in [
        ("publish", s.publish_cyc),
        ("kernel(shard0)", s.kernel0_cyc),
        ("drain", s.drain_cyc),
    ] {
        writeln!(out, "  {:<16} {:>14} {:>6.1}%", name, c, share(c)).unwrap();
    }
    writeln!(
        out,
        "  per-call max-lane sums: wake {} | wake-parked {} ({} calls) | kernel-max {} | alloc-max {}",
        s.wake_max_cyc, s.wake_parked_cyc, s.parked_calls, s.kernel_max_cyc, s.alloc_max_cyc
    )
    .unwrap();
    writeln!(
        out,
        "  {:<6} {:>14} {:>14} {:>14} {:>13}",
        "lane", "wake", "kernel", "alloc", "parked-calls"
    )
    .unwrap();
    for i in 0..s.lane_kernel_cyc.len() {
        writeln!(
            out,
            "  {:<6} {:>14} {:>14} {:>14} {:>13}",
            i,
            s.lane_wake_cyc[i],
            s.lane_kernel_cyc[i],
            s.lane_alloc_cyc[i],
            s.lane_parked_calls[i]
        )
        .unwrap();
    }
    let mut hist = String::from("  per-call cycles histogram:");
    for (b, &n) in s.hist_log2.iter().enumerate() {
        if n > 0 {
            write!(hist, " 2^{b}:{n}").unwrap();
        }
    }
    writeln!(out, "{hist}").unwrap();
    out
}

/// Render the M4b.14 prefill attn scores/softmax/output sub-bracket rows
/// (only compiled in under the `attn-subprofile` feature — see
/// `inferno_kernels::attention::subprofile`). Cycle numbers are printed
/// raw, unindented and anchored at column 0 so
/// `scripts/quiet-hw/gate-prefill-attn-split.sh` can `grep -E
/// '^attn:(scores|softmax|output)'` for them; `attn_share`/ceiling
/// arithmetic is controller work done in the spec's Amendments, never here
/// (same discipline as `render_pool`).
#[cfg(feature = "attn-subprofile")]
pub fn render_attn_subprofile((scores, softmax, output): (u64, u64, u64)) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let total = scores + softmax + output;
    writeln!(out, "attn [prefill sub-brackets] {total} cyc instrumented").unwrap();
    for (name, c) in [
        ("attn:scores", scores),
        ("attn:softmax", softmax),
        ("attn:output", output),
    ] {
        writeln!(out, "{:<13} {:>14}", name, c).unwrap();
    }
    out
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

    #[test]
    #[cfg(feature = "attn-subprofile")]
    fn render_attn_subprofile_prints_greppable_rows() {
        let out = super::render_attn_subprofile((10, 20, 30));
        assert!(out.contains("60 cyc instrumented"), "{out}");
        // Every row must be grep-anchorable at column 0 by the gate script's
        // `grep -E '^attn:(scores|softmax|output)'`.
        let rows: Vec<&str> = out.lines().filter(|l| l.starts_with("attn:")).collect();
        assert_eq!(rows.len(), 3, "{out}");
        assert!(
            rows[0].starts_with("attn:scores") && rows[0].contains("10"),
            "{out}"
        );
        assert!(
            rows[1].starts_with("attn:softmax") && rows[1].contains("20"),
            "{out}"
        );
        assert!(
            rows[2].starts_with("attn:output") && rows[2].contains("30"),
            "{out}"
        );
    }

    #[test]
    fn render_pool_prints_buckets_and_identity() {
        let snap = inferno_pool::PoolProfSnapshot {
            calls: 3,
            publish_cyc: 100,
            kernel0_cyc: 700,
            drain_cyc: 200,
            wake_max_cyc: 90,
            wake_parked_cyc: 60,
            parked_calls: 1,
            kernel_max_cyc: 750,
            alloc_max_cyc: 30,
            lane_wake_cyc: vec![0, 90],
            lane_kernel_cyc: vec![700, 740],
            lane_alloc_cyc: vec![30, 25],
            lane_parked_calls: vec![0, 1],
            hist_log2: {
                let mut h = vec![0u64; 64];
                h[9] = 3;
                h
            },
        };
        let out = super::render_pool(&snap, 1100);
        assert!(out.starts_with("pool [decode attention] 3 calls"), "{out}");
        // instr_total = 1000, op attention = 1100 → 90.9%.
        assert!(out.contains("90.9%"), "{out}");
        assert!(out.contains("publish"), "{out}");
        assert!(out.contains("kernel(shard0)"), "{out}");
        assert!(out.contains("drain"), "{out}");
        assert!(out.contains("wake-parked 60 (1 calls)"), "{out}");
        assert!(out.contains("2^9:3"), "{out}");
        // publish share of instr total: 100/1000.
        assert!(out.contains("10.0%"), "{out}");
    }
}
