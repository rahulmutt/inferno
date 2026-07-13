//! M4b.10 curve 2: aggregate streaming bandwidth vs lane count, measured by
//! driving the REAL Q8_0 GEMV kernel through the REAL pool over a weight
//! image larger than any target box's L3 — decode's actual access pattern.
//!
//! Prints the curve and the derived knee P (the smallest lane count reaching
//! 95% of peak bandwidth). Paired with `gate-decode-cap`'s knee, this is what
//! makes the M4b.10 decision rule falsifiable: rule 2 fires only if this
//! curve predicts that knee.
//!
//! Usage: cargo run --release -p inferno-pool --example bw_curve -- <max_lanes>

use inferno_formats::{DType, quant};
use inferno_kernels::{KernelIsa, act, q8_0};
use inferno_pool::{GemvFn, Pool, bandwidth_curve, knee_at_fraction};

/// 32768 rows x 4096 k in Q8_0 packs to ~143 MiB — comfortably past the
/// largest L3 in the M4b.10 machine matrix (Platinum 8352Y, 48 MiB), so
/// every lane streams from DRAM.
const ROWS: usize = 32768;
const K: usize = 4096;
const REPS: usize = 5;
const KNEE_FRACTION: f64 = 0.95;

/// Deterministic pseudo-random f32s in [-1, 1) — the same xorshift the
/// kernels' rig and `par_rig.rs` use, so no dependency is added.
fn pseudo(mut seed: u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn main() {
    let max_lanes: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .max(1);

    // Host ISA, exactly as inferno-plan selects it (weights.rs:52): the real
    // kernel, not the scalar reference — a scalar curve would be
    // compute-bound and measure nothing about memory.
    let isa = if KernelIsa::Avx2.available() {
        KernelIsa::Avx2
    } else {
        KernelIsa::Scalar
    };
    let kernel: GemvFn = match isa {
        KernelIsa::Avx2 => inferno_kernels::inferno_gemv_q8_0_rs8_avx2,
        KernelIsa::Scalar => inferno_kernels::inferno_gemv_q8_0_rs8_scalar,
    };
    // NOTE (deviation from the task-4 brief, flagged in task-4-report.md):
    // the brief's `kernels_for(&DType::Q8_0, isa)` does not type-check
    // against the current registry — `kernels_for` takes
    // `inferno_target::Isa` (the target-profile ISA level), not
    // `inferno_kernels::KernelIsa`, and `inferno-target` is not a
    // dependency of `inferno-pool` (adding it would be a manifest change
    // the brief says to stop and report on). `q8_0::pack_q8_0_rs8` /
    // `packed_len_q8_0_rs8` are ISA-invariant (same fn for both KernelSet
    // variants in the registry), and `act::quantize_row_q8a` already takes
    // `KernelIsa` directly, so this reaches the identical AVX2/scalar
    // symbols the brief intended without adding a dependency.
    let wbytes = quant::pack(&DType::Q8_0, &pseudo(0xfeed_beef, ROWS * K)).expect("pack Q8_0");
    let w = q8_0::pack_q8_0_rs8(&wbytes, ROWS, K).expect("pack rs8");
    let xq =
        act::quantize_row_q8a(isa, &pseudo(0x9e37_79b9_7f4a_7c15, K)).expect("quantize activation");
    let stream_bytes = q8_0::packed_len_q8_0_rs8(ROWS, K);
    let mut y = vec![f32::NAN; ROWS];

    let pool = Pool::new(max_lanes);
    let lanes: Vec<usize> = (1..=max_lanes).collect();

    // SAFETY: w/xq built by this function for exactly (ROWS, K); y has ROWS
    // f32s; `kernel` is the Q8_0 GEMV symbol for the detected ISA.
    let curve = unsafe {
        bandwidth_curve(
            &pool,
            &lanes,
            REPS,
            stream_bytes,
            kernel,
            y.as_mut_ptr(),
            xq.as_ptr(),
            w.as_ptr(),
            K,
            ROWS,
        )
    };

    let base = curve.first().map(|&(_, r)| r).unwrap_or(1.0);
    println!(
        "shape: {ROWS} rows x {K} k, Q8_0, {isa:?} | weight image {:.1} MiB | reps={REPS} (median)",
        stream_bytes as f64 / (1024.0 * 1024.0)
    );
    println!();
    println!("| lanes | GB/s | speedup vs 1 lane |");
    println!("|---|---|---|");
    for &(l, gbps) in &curve {
        println!("| {l} | {gbps:.2} | {:.2}x |", gbps / base);
    }
    println!();
    println!(
        "P (smallest lanes at >= {:.0}% of peak): {}",
        KNEE_FRACTION * 100.0,
        knee_at_fraction(&curve, KNEE_FRACTION)
    );
    println!("gate input (human verdict to the M4b.10 spec): does P match the");
    println!("decode knee from gate-decode-cap on this same box?");
}
