# M4b.10 — Decode-Cap Formula Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace inferno's decode thread-cap formula — which has missed its exit criterion on quiet bare metal three times — with one selected by a decision rule pre-registered before the data is taken.

**Architecture:** This is a **data-gated** milestone. Tasks 1–4 build the measurement surface (a bandwidth-saturation probe in `inferno-pool`, a new quiet-hw gate wrapping it, and a decode-cap sweep that can grid-coarsen and NUMA-pin). Task 5 takes three quiet-hardware sessions. Task 6 applies the pre-registered rule. Only **Task 7** touches the shipped formula, and it deliberately carries **three complete alternative implementations** — the rule picks exactly one. No task before Task 6 may assume which.

**Tech Stack:** Rust (workspace crates `inferno-pool`, `inferno-core`), bash (`scripts/quiet-hw/`), mise tasks, PhoenixNAP bare metal via `mise run metal`.

**Spec:** [`docs/superpowers/specs/2026-07-12-m4b10-decode-cap-formula-design.md`](../specs/2026-07-12-m4b10-decode-cap-formula-design.md) (committed `58f4fbe`).

## Global Constraints

Copied verbatim from the spec. Every task's requirements implicitly include these.

- **Decode only.** Prefill (`inferno_par_gemm`, `inferno_par_attention`, `inferno_par_token_loop`) is untouched.
- **Pool-side only.** No codegen edit, **no `HOST_ABI_VERSION` bump**, **no recompile** — an existing cached `model.so` must benefit immediately.
- **No tolerance edits.** `cargo test -p inferno-codegen --test differential` and `cargo test -p inferno-core --test artifact` must pass with existing bounds (AGENTS.md standing rule).
- **The cap is bit-invisible and must stay so.** `shard_table` computes each output row entirely within one lane. The existing `par_gemv` cap-invariance test is the standing guard.
- **`INFERNO_DECODE_THREADS` survives in every outcome** and keeps precedence over the formula.
- **The formula choice stays OPEN until Task 6.** Do not implement a formula in Tasks 1–5. Do not "helpfully" pick one early.
- **Never edit a recorded data point.** Sweep output is pasted verbatim into spec Amendments.
- **Workflows are mise tasks:** `mise run test` / `lint` / `metal`. Don't hand-roll cargo invocations in docs or CI.
- Scripts never write to `docs/` — verdicts are pasted in by a human (`docs/runbooks/quiet-hw-verification.md`).

## File Structure

| File | Responsibility |
|---|---|
| `crates/inferno-pool/src/probe.rs` (create) | Bandwidth-saturation measurement: time `par_gemv` at each lane count; derive the saturation knee. Pure-pool, no new dependency. |
| `crates/inferno-pool/src/lib.rs` (modify) | Export `probe`. |
| `crates/inferno-pool/examples/bw_curve.rs` (create) | Drives `probe` with the **real Q8_0 GEMV kernel** over a >L3 synthetic matrix; prints the curve. Uses the existing `inferno-kernels` dev-dependency. |
| `scripts/quiet-hw/lib.sh` (modify) | Add `cap_grid` (bound sweep size above 16 cores) and `numa_wrap` (socket pinning). |
| `scripts/quiet-hw/lib-selftest.sh` (modify) | Offline tests for both new helpers. |
| `scripts/quiet-hw/gate-decode-cap.sh` (modify) | Use `cap_grid` + `numa_wrap`. |
| `scripts/quiet-hw/gate-bw-curve.sh` (create) | Wrap the example as gate 6. |
| `scripts/quiet-hw/verify.sh` (modify) | Run the new gate; add it to the summary table. |
| `docs/runbooks/quiet-hw-verification.md` (modify) | Gate-table row → verdict destination. |
| `crates/inferno-core/src/lib.rs` (modify, **Task 7 only**) | The shipped `decode_cap` formula. |
| `AGENTS.md` (modify, Task 8) | The decode-cap constraint bullet currently states the old default. |

---

### Task 1: quiet-hw shell helpers (`cap_grid`, `numa_wrap`)

The 32-core socket-pinned box would otherwise sweep 32 cap values × 3 reps × 128 tokens. `cap_grid` keeps it fine-grained where the knee lives (≤16) and steps by 4 above. `numa_wrap` is what makes the socket-pinned session honest.

**Files:**
- Modify: `scripts/quiet-hw/lib.sh` (append after `phys_cores`, ends line 98)
- Test: `scripts/quiet-hw/lib-selftest.sh` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `cap_grid <max>` → space-separated ascending cap list, always including `1` and `<max>`. `numa_wrap` → echoes the `numactl` prefix words when `QHW_NUMA_NODE` is set, nothing otherwise.

- [ ] **Step 1: Write the failing selftest**

Append to `scripts/quiet-hw/lib-selftest.sh` (before any final summary line; match the file's existing `expect`/`fail` helper style):

```bash
# cap_grid — fine-grained to 16, step 4 above; always includes 1 and max.
expect "cap_grid 8"  "$(cap_grid 8)"  "1 2 3 4 5 6 7 8"
expect "cap_grid 16" "$(cap_grid 16)" "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
expect "cap_grid 32" "$(cap_grid 32)" "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 20 24 28 32"
expect "cap_grid 1"  "$(cap_grid 1)"  "1"
# 18 is not a multiple of 4 above 16 — max must still appear, exactly once.
expect "cap_grid 18" "$(cap_grid 18)" "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18"

# numa_wrap — empty unless QHW_NUMA_NODE is set.
expect "numa_wrap unset" "$(numa_wrap)" ""
expect "numa_wrap set"   "$(QHW_NUMA_NODE=0 numa_wrap)" "numactl --cpunodebind=0 --membind=0"
expect "numa_wrap node1" "$(QHW_NUMA_NODE=1 numa_wrap)" "numactl --cpunodebind=1 --membind=1"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bash scripts/quiet-hw/lib-selftest.sh`
Expected: FAIL — `cap_grid: command not found` (the helper does not exist yet).

- [ ] **Step 3: Implement the helpers**

Append to `scripts/quiet-hw/lib.sh`:

```bash
# cap_grid <max> — decode-cap sweep values: every cap up to 16, then step 4.
# Bounds session time on many-core boxes (M4b.10) while keeping full
# resolution where every recorded knee has landed (8..16). `max` always
# appears, exactly once.
cap_grid() {
  local max="$1" i out=""
  for i in $(seq 1 "$max"); do
    if [ "$i" -le 16 ] || [ $((i % 4)) -eq 0 ] || [ "$i" -eq "$max" ]; then
      out="$out $i"
    fi
  done
  echo "${out# }"
}

# numa_wrap — the numactl prefix pinning CPUs *and* memory to QHW_NUMA_NODE,
# or nothing when unset. Used to take a NUMA-free single-socket point on a
# dual-socket box (M4b.10: d2.c5.large is 2x32c).
numa_wrap() {
  [ -n "${QHW_NUMA_NODE:-}" ] || return 0
  echo "numactl --cpunodebind=${QHW_NUMA_NODE} --membind=${QHW_NUMA_NODE}"
}
```

- [ ] **Step 4: Run the selftest to verify it passes**

Run: `bash scripts/quiet-hw/lib-selftest.sh`
Expected: PASS — all `cap_grid` / `numa_wrap` lines OK, no regressions in existing checks.

- [ ] **Step 5: Commit**

```bash
git add scripts/quiet-hw/lib.sh scripts/quiet-hw/lib-selftest.sh
git commit -m "quiet-hw: add cap_grid + numa_wrap helpers (M4b.10)"
```

---

### Task 2: `gate-decode-cap.sh` — coarse grid + NUMA pinning

**Files:**
- Modify: `scripts/quiet-hw/gate-decode-cap.sh` (lines 17–22 and the `one_run` body, lines 29–39)

**Interfaces:**
- Consumes: `cap_grid`, `numa_wrap` (Task 1).
- Produces: unchanged stdout table contract (`| cap | decode tok/s | per-rep |`, then `knee (best fixed cap):` and the `default clamp(...)` line). The verify.sh driver and the runbook depend on this shape — do not change it.

- [ ] **Step 1: Replace the sweep-bounds block**

In `scripts/quiet-hw/gate-decode-cap.sh`, replace lines 17–22:

```bash
PHYS=$(phys_cores)
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  CAPS="1 2"; REPS=1; MAXTOK=8
else
  CAPS=$(seq 1 "$PHYS" | tr '\n' ' '); REPS=3; MAXTOK=128
fi
```

with:

```bash
PHYS=$(phys_cores)
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  CAPS="1 2"; REPS=1; MAXTOK=8
else
  CAPS=$(cap_grid "$PHYS"); REPS=3; MAXTOK=128
fi
```

- [ ] **Step 2: Pin runs to the NUMA node**

In the same file, replace the `one_run` body (lines 29–39):

```bash
one_run() { # <cap: number|default|t1> -> decode tok/s on stdout
  local threads=0 envset=()
  case "$1" in
    default) ;;                       # heuristic path: env unset
    t1)      threads=1 ;;             # t=1 decode-unchanged row
    *)       envset=(INFERNO_DECODE_THREADS="$1") ;;
  esac
  env "${envset[@]}" cargo run --release -q -p inferno -- run "$MODEL" \
    -p "$PROMPT" --max-tokens "$MAXTOK" --threads "$threads" 2>&1 \
    | tee -a "$OUT/decode-cap-runs.log" | decode_toks -
}
```

with:

```bash
one_run() { # <cap: number|default|t1> -> decode tok/s on stdout
  local threads=0 envset=()
  case "$1" in
    default) ;;                       # heuristic path: env unset
    t1)      threads=1 ;;             # t=1 decode-unchanged row
    *)       envset=(INFERNO_DECODE_THREADS="$1") ;;
  esac
  # numa_wrap is empty unless QHW_NUMA_NODE is set; unquoted on purpose so it
  # expands to zero words in the common case.
  env "${envset[@]}" $(numa_wrap) cargo run --release -q -p inferno -- run "$MODEL" \
    -p "$PROMPT" --max-tokens "$MAXTOK" --threads "$threads" 2>&1 \
    | tee -a "$OUT/decode-cap-runs.log" | decode_toks -
}
```

- [ ] **Step 3: Record the pinning in the header**

Replace line 26 (the `sweep:` echo):

```bash
echo "sweep: caps={$CAPS} + default + t1 | reps=$REPS (interleaved rounds) | max-tokens=$MAXTOK"
```

with:

```bash
echo "sweep: caps={$CAPS} + default + t1 | reps=$REPS (interleaved rounds) | max-tokens=$MAXTOK"
[ -n "${QHW_NUMA_NODE:-}" ] && echo "numa: pinned to node ${QHW_NUMA_NODE} (cpubind+membind); phys_cores=$PHYS"
```

Provenance matters: a pinned sweep and an unpinned one are not the same data point, and the recorded output must say which it is.

- [ ] **Step 4: Smoke the gate**

Run: `QHW_SMOKE=1 QHW_OUT=$(mktemp -d) devenv shell -- bash scripts/quiet-hw/gate-decode-cap.sh <model.gguf>`
Expected: the `### SMOKE — NON-RECORDABLE ###` stamp, a 2-row cap table (caps 1, 2), and `SMOKE: evaluation skipped`. No `numa:` line (env unset).

- [ ] **Step 5: Commit**

```bash
git add scripts/quiet-hw/gate-decode-cap.sh
git commit -m "quiet-hw: gate-decode-cap grid-coarsens above 16 caps and honors QHW_NUMA_NODE (M4b.10)"
```

---

### Task 3: `inferno-pool` bandwidth probe

The physical model M4b.5 names is `total_DRAM_bandwidth / per_core_streaming_bandwidth`. This module **measures** it rather than looking it up (spec §"Why we cannot look the answer up"). It is pool-side and generic over the kernel, so it adds **no new dependency**: the caller supplies the `GemvFn` and buffers.

If the rule selects the probe (rule 2), this same module becomes the runtime probe — which is why the logic lives here rather than in the example.

**Files:**
- Create: `crates/inferno-pool/src/probe.rs`
- Modify: `crates/inferno-pool/src/lib.rs:10-15` (module + re-export)

**Interfaces:**
- Consumes: `Pool` (`set_decode_threads`, `decode_threads`, `par_gemv`) from `crates/inferno-pool/src/pool.rs`; `GemvFn = unsafe extern "C" fn(*mut f32, *const u8, *const u8, usize, usize, usize)`.
- Produces:
  - `pub fn knee_at_fraction(curve: &[(usize, f64)], frac: f64) -> usize`
  - `pub unsafe fn bandwidth_curve(pool: &Pool, lanes: &[usize], reps: usize, stream_bytes: usize, kernel: GemvFn, y: *mut f32, xq: *const u8, w: *const u8, k: usize, rows: usize) -> Vec<(usize, f64)>`

- [ ] **Step 1: Write the failing tests**

Create `crates/inferno-pool/src/probe.rs` containing **only** the test module for now:

```rust
//! Bandwidth-saturation probe (M4b.10). Times `par_gemv` at a range of lane
//! counts and derives the lane count at which aggregate streaming bandwidth
//! saturates — the physically motivated decode cap
//! (`total_DRAM_bandwidth / per_core_streaming_bandwidth`).
//!
//! Generic over the kernel on purpose: the caller supplies the `GemvFn` and
//! the packed buffers, so this crate gains no dependency on `inferno-kernels`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pool;

    #[test]
    fn knee_is_the_first_lane_count_reaching_the_fraction_of_peak() {
        // Saturates at 2 lanes: peak 21.0, 95% of peak = 19.95, and 2 lanes
        // already delivers 20.0.
        let curve = [(1, 10.0), (2, 20.0), (4, 21.0), (8, 21.0)];
        assert_eq!(knee_at_fraction(&curve, 0.95), 2);
    }

    #[test]
    fn a_curve_that_never_saturates_knees_at_the_top() {
        let curve = [(1, 10.0), (2, 20.0), (4, 40.0)];
        assert_eq!(knee_at_fraction(&curve, 0.95), 4);
    }

    #[test]
    fn a_non_monotonic_tail_does_not_move_the_knee_below_the_peak_fraction() {
        // 8 lanes regresses; the knee is still where 95% of peak is first hit.
        let curve = [(1, 10.0), (2, 20.0), (4, 21.0), (8, 18.0)];
        assert_eq!(knee_at_fraction(&curve, 0.95), 2);
    }

    #[test]
    fn degenerate_curves_knee_at_one_lane() {
        assert_eq!(knee_at_fraction(&[], 0.95), 1);
        assert_eq!(knee_at_fraction(&[(3, 12.0)], 0.95), 3);
    }

    /// A stub kernel: writes each row so the dispatcher's work is real but
    /// trivially fast. The GB/s values are meaningless here — this test is
    /// about the curve's *shape contract*, not its numbers.
    unsafe extern "C" fn stub_gemv(
        y: *mut f32,
        _xq: *const u8,
        _w: *const u8,
        _k: usize,
        row_start: usize,
        row_end: usize,
    ) {
        for r in row_start..row_end {
            // SAFETY: the dispatcher only ever passes rows within `y`'s length.
            unsafe { *y.add(r) = r as f32 };
        }
    }

    #[test]
    fn bandwidth_curve_returns_one_entry_per_lane_and_restores_the_cap() {
        let pool = Pool::new(4);
        pool.set_decode_threads(3);
        let mut y = vec![0f32; 64];
        let xq = [0u8; 8];
        let w = [0u8; 8];
        let lanes = [1usize, 2, 4];

        // SAFETY: stub_gemv only writes y[row_start..row_end]; rows == y.len().
        let curve = unsafe {
            bandwidth_curve(
                &pool,
                &lanes,
                2,
                1 << 20,
                stub_gemv,
                y.as_mut_ptr(),
                xq.as_ptr(),
                w.as_ptr(),
                8,
                64,
            )
        };

        assert_eq!(curve.len(), 3);
        assert_eq!(
            curve.iter().map(|&(l, _)| l).collect::<Vec<_>>(),
            vec![1, 2, 4]
        );
        assert!(
            curve.iter().all(|&(_, gbps)| gbps > 0.0),
            "every lane count must record a positive rate: {curve:?}"
        );
        assert_eq!(
            pool.decode_threads(),
            3,
            "the probe must restore the caller's decode cap"
        );
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p inferno-pool --lib probe`
Expected: FAIL to compile — `cannot find function knee_at_fraction`, `cannot find function bandwidth_curve`, and `probe` is not declared as a module.

- [ ] **Step 3: Implement the probe**

Prepend to `crates/inferno-pool/src/probe.rs`, above the `#[cfg(test)]` module (keep the `//!` header at the top of the file):

```rust
use crate::pool::{GemvFn, Pool};
use std::time::Instant;

/// The smallest lane count in `curve` reaching `frac` of the curve's peak
/// rate — the saturation knee. An empty curve knees at 1 lane.
///
/// Deliberately reads the *first* lane count at or above the threshold, not
/// the argmax: past saturation the curve is flat-to-noisy, and the cheapest
/// lane count on the plateau is the one we want.
pub fn knee_at_fraction(curve: &[(usize, f64)], frac: f64) -> usize {
    let peak = curve.iter().map(|&(_, r)| r).fold(f64::MIN, f64::max);
    if curve.is_empty() {
        return 1;
    }
    let target = peak * frac;
    curve
        .iter()
        .find(|&&(_, r)| r >= target)
        .map(|&(lanes, _)| lanes)
        .unwrap_or_else(|| curve.last().map(|&(l, _)| l).unwrap_or(1))
}

/// Time `reps` full-range `par_gemv` dispatches at each lane count in
/// `lanes`, returning `(lanes, GB/s)` per entry. `stream_bytes` is the number
/// of bytes the kernel streams per dispatch (the packed weight image), which
/// is what makes the rate a *bandwidth* rather than a throughput.
///
/// Takes the median of `reps` timings per lane count, so one descheduled
/// iteration cannot move the curve. Saves and restores the pool's decode cap.
///
/// # Safety
/// Same contract as [`Pool::par_gemv`] for `(kernel, y, xq, w, k, rows)`:
/// `y` valid for `rows` f32 writes, `xq`/`w` valid packed buffers built for
/// this exact `k` and `rows`, and `kernel` a valid GEMV-ABI pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe fn bandwidth_curve(
    pool: &Pool,
    lanes: &[usize],
    reps: usize,
    stream_bytes: usize,
    kernel: GemvFn,
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    rows: usize,
) -> Vec<(usize, f64)> {
    let restore = pool.decode_threads();
    let mut out = Vec::with_capacity(lanes.len());

    for &n in lanes {
        pool.set_decode_threads(n);

        // Warm the lanes and the weight image into whatever caches will hold
        // it, so the first timed rep is not paying for a cold pool.
        // SAFETY: forwarding the caller's contract unchanged.
        unsafe { pool.par_gemv(kernel, y, xq, w, k, rows) };

        let mut secs: Vec<f64> = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            // SAFETY: forwarding the caller's contract unchanged.
            unsafe { pool.par_gemv(kernel, y, xq, w, k, rows) };
            secs.push(t0.elapsed().as_secs_f64());
        }
        secs.sort_by(f64::total_cmp);
        let med = secs[secs.len() / 2].max(f64::EPSILON);
        out.push((n, stream_bytes as f64 / med / 1e9));
    }

    pool.set_decode_threads(restore);
    out
}
```

Declare the module and re-export it. In `crates/inferno-pool/src/lib.rs`, change lines 9–15 from:

```rust
pub mod error;
pub mod pool;
pub mod shard;

pub use error::PoolError;
pub use pool::{AttnFn, AttnJob, GemmFn, GemvFn, Pool, TokenBodyFn};
pub use shard::{SHARD_ALIGN, shard_table, shard_table_aligned};
```

to:

```rust
pub mod error;
pub mod pool;
pub mod probe;
pub mod shard;

pub use error::PoolError;
pub use pool::{AttnFn, AttnJob, GemmFn, GemvFn, Pool, TokenBodyFn};
pub use probe::{bandwidth_curve, knee_at_fraction};
pub use shard::{SHARD_ALIGN, shard_table, shard_table_aligned};
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p inferno-pool --lib probe`
Expected: PASS — 5 tests.

Then confirm nothing else moved: `mise run test` → green.

- [ ] **Step 5: Lint**

Run: `mise run lint`
Expected: clean. (`mise run test` skips clippy; CI's `lint` runs `clippy -D warnings`. The `#[allow(clippy::too_many_arguments)]` above is deliberate — the arg list mirrors `par_gemv`'s.)

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-pool/src/probe.rs crates/inferno-pool/src/lib.rs
git commit -m "pool: bandwidth-saturation probe (curve + knee) for M4b.10"
```

---

### Task 4: `bw_curve` example + `gate-bw-curve.sh` + wiring

Drives the probe with the **real Q8_0 GEMV** over a synthetic weight matrix sized past L3 on every target box (the 8352Y's 48 MiB L3 is the largest), so the curve measures DRAM streaming — decode's actual access pattern.

`inferno-pool` already carries `inferno-kernels` and `inferno-formats` as dev-dependencies (`crates/inferno-pool/Cargo.toml`), which examples may use. No manifest change is needed.

**Files:**
- Create: `crates/inferno-pool/examples/bw_curve.rs`
- Create: `scripts/quiet-hw/gate-bw-curve.sh`
- Modify: `scripts/quiet-hw/verify.sh:70-74` (gate list) and `:85-87` (summary loop) and `:95` (exit loop)
- Modify: `docs/runbooks/quiet-hw-verification.md` (gate table)

**Interfaces:**
- Consumes: `inferno_pool::{Pool, bandwidth_curve, knee_at_fraction}` (Task 3); `inferno_kernels::{kernels_for, reference_kernels, KernelIsa, inferno_gemv_q8_0_rs8_avx2, inferno_gemv_q8_0_rs8_scalar}`; `inferno_formats::{DType, quant}`.
- Produces: stdout table `| lanes | GB/s | speedup |` plus a `P (95% of peak):` line. The gate script and the human verdict read these.

- [ ] **Step 1: Write the example**

Create `crates/inferno-pool/examples/bw_curve.rs`:

```rust
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
use inferno_kernels::{KernelIsa, kernels_for, reference_kernels};
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
    let ks = kernels_for(&DType::Q8_0, isa)
        .or_else(|| reference_kernels(&DType::Q8_0))
        .expect("Q8_0 kernel set");

    let wbytes = quant::pack(&DType::Q8_0, &pseudo(0xfeed_beef, ROWS * K)).expect("pack Q8_0");
    let w = ks.pack(&wbytes, ROWS, K).expect("pack rs8");
    let xq = ks
        .quantize_row(&pseudo(0x9e37_79b9_7f4a_7c15, K))
        .expect("quantize activation");
    let stream_bytes = ks.packed_len(ROWS, K);
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
```

- [ ] **Step 2: Run it locally to verify it produces a curve**

Run: `cargo run --release -p inferno-pool --example bw_curve -- 4`
Expected: a `shape:` line naming `Avx2` (on any x86 dev box), a 4-row table with monotonically-ish increasing GB/s, and a `P (...)` line with a lane count in `1..=4`.

The devpod is an 8-CPU cgroup-quota'd shared box, so these numbers are **not** recordable — this step only proves the example runs.

- [ ] **Step 3: Write the gate script**

Create `scripts/quiet-hw/gate-bw-curve.sh`:

```bash
#!/usr/bin/env bash
# M4b.10 gate 6 — curve 2: aggregate streaming bandwidth vs lane count, from
# the real Q8_0 GEMV through the real pool. Paired with gate-decode-cap's
# knee on the same box, this is the falsifiability test for the M4b.10
# decision rule: rule 2 (ship a runtime bandwidth probe) fires only if this
# curve's P predicts the measured decode knee. Verdict destination: the
# M4b.10 spec §Amendments
# (docs/superpowers/specs/2026-07-12-m4b10-decode-cap-formula-design.md).
# Usage: gate-bw-curve.sh   (env: QHW_OUT QHW_SMOKE QHW_NUMA_NODE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

PHYS=$(phys_cores)
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  LANES=2
else
  LANES="$PHYS"
fi

smoke_header "gate-bw-curve (M4b.10 bandwidth saturation)"
machine_block
[ -n "${QHW_NUMA_NODE:-}" ] && echo "numa: pinned to node ${QHW_NUMA_NODE} (cpubind+membind); phys_cores=$PHYS"
echo

# numa_wrap is empty unless QHW_NUMA_NODE is set; unquoted on purpose so it
# expands to zero words in the common case.
$(numa_wrap) cargo run --release -q -p inferno-pool --example bw_curve -- "$LANES"

if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo
  echo "SMOKE: evaluation skipped"
fi
```

Make it executable and match the sibling gates' mode:

```bash
chmod +x scripts/quiet-hw/gate-bw-curve.sh
```

- [ ] **Step 4: Smoke it**

Run: `QHW_SMOKE=1 devenv shell -- bash scripts/quiet-hw/gate-bw-curve.sh`
Expected: the `### SMOKE — NON-RECORDABLE ###` stamp, a `machine:` line, a 2-lane table, a `P (...)` line, then `SMOKE: evaluation skipped`.

- [ ] **Step 5: Wire it into the orchestrator**

In `scripts/quiet-hw/verify.sh`, add the gate after `decode-cap` (line 71), so it runs on the same box in the same session — the pairing is the whole point:

```bash
run_gate prefill-scaling bash "$HERE/gate-prefill-scaling.sh" "$MODEL"
run_gate decode-cap      bash "$HERE/gate-decode-cap.sh" "$MODEL"
run_gate bw-curve        bash "$HERE/gate-bw-curve.sh"
run_gate pf-dist         bash "$HERE/gate-pf-dist.sh"
run_gate bench-protocol  bash "$HERE/gate-bench-protocol.sh" "$MODEL"
run_gate intel-ab        bash "$HERE/gate-intel-ab.sh" "${AB_ARGS[@]}"
```

In the summary loop (line 86), add `bw-curve`:

```bash
  for g in prefill-scaling decode-cap bw-curve pf-dist bench-protocol intel-ab; do
```

And in the final exit loop (line 95), add it to the must-pass set:

```bash
for g in prefill-scaling decode-cap bw-curve pf-dist bench-protocol; do
```

- [ ] **Step 6: Add the runbook row**

In `docs/runbooks/quiet-hw-verification.md`, in the results table (the one whose rows read `` `gate-decode-cap.out` | [M4b.5 spec](...) §Amendments | ... ``), add after the `gate-decode-cap.out` row:

```markdown
| `gate-bw-curve.out` | [M4b.10 spec](../superpowers/specs/2026-07-12-m4b10-decode-cap-formula-design.md) §Amendments | bandwidth curve; P (95%-of-peak lane count); does P predict the decode knee? |
```

- [ ] **Step 7: Full smoke pass**

Run: `QHW_SMOKE=1 devenv shell -- bash scripts/quiet-hw/verify.sh <model.gguf> --smoke`
Expected: the summary table lists `bw-curve` with status `PASS`.

- [ ] **Step 8: Commit**

```bash
git add crates/inferno-pool/examples/bw_curve.rs scripts/quiet-hw/gate-bw-curve.sh \
        scripts/quiet-hw/verify.sh docs/runbooks/quiet-hw-verification.md
git commit -m "quiet-hw: gate-bw-curve — bandwidth saturation vs lane count (M4b.10 curve 2)"
```

---

### Task 5: Three quiet-hardware sessions

**This task spends real money** (`mise run metal`, PhoenixNAP hourly). Operator-driven, never CI. After any interrupted session run `mise run metal-gc` — EXIT traps don't survive killed terminals (AGENTS.md).

No code changes. The deliverable is three recorded sweeps.

**Files:**
- None modified. Raw output is collected under `target/quiet-hw/<timestamp>/`.

**Interfaces:**
- Consumes: `gate-decode-cap.sh` (Task 2), `gate-bw-curve.sh` (Task 4).
- Produces: for each machine, a `gate-decode-cap.out` (knee + per-cap medians) and a `gate-bw-curve.out` (bandwidth curve + P), which Task 6 consumes.

- [ ] **Step 1: Session A — `d2.c1.medium` (16c, the box with three prior knee sessions)**

```bash
mise run metal -- d2.c1.medium --yes -- \
  'bash scripts/quiet-hw/verify.sh models/qwen2.5-0.5b-instruct-q8_0.gguf'
```

Required: `PREFLIGHT FIT`. This box already has three recorded knee sessions but **no bandwidth curve** — the curve is why we re-run it, and the rule (spec §"One session per machine is authoritative") consumes *this* session, not the older ones.

Sanity-check before recording: the knee and the `default` regret must land inside the recorded spread (knee 12–13; default −9.8% to −11.8%). If they don't, that is a measurement problem to resolve **before** applying the rule — not a datum to average away.

- [ ] **Step 2: Session B — `s2.c2.medium` (Xeon E-2388G, 8c)**

```bash
mise run metal -- s2.c2.medium --yes -- \
  'bash scripts/quiet-hw/verify.sh models/qwen2.5-0.5b-instruct-q8_0.gguf'
```

Note this box is AVX-512-capable but 8-core; `phys_cores` → 8, so `cap_grid` yields the full `1..8`.

- [ ] **Step 3: Session C — `d2.c5.large` (Platinum 8352Y), pinned to one socket**

The box is **dual-socket** (2×32c). Pinning is what keeps NUMA out of scope, as it has been since M4b.1:

```bash
mise run metal -- d2.c5.large --yes -- \
  'QHW_NUMA_NODE=0 bash scripts/quiet-hw/verify.sh models/qwen2.5-0.5b-instruct-q8_0.gguf'
```

Verify in the output that **both** `gate-decode-cap.out` and `gate-bw-curve.out` carry the `numa: pinned to node 0` line and report `phys_cores=32`. A session missing that line is unpinned and **must not be recorded as the pinned point**.

`cap_grid 32` → `1..16` then `20 24 28 32`, i.e. 20 cap values, keeping the sweep to ~20×3 runs.

- [ ] **Step 4: Verify the ISA table did not drift**

Each provision verifies `scripts/metal/cpu-features.json` against `/proc/cpuinfo`. On drift, **fix the table in a commit — never override** (AGENTS.md; `docs/runbooks/metal.md`). The catalog has been wrong before: `d2.c1.medium` was catalogued as a 5315Y and delivers a 6336Y (commit `f72d67c`).

Record what each box actually delivered — including its true core count — in the Task 6 amendment.

- [ ] **Step 5: Confirm no stray servers**

Run: `mise run metal-gc`
Expected: no `inferno-metal` servers listed.

- [ ] **Step 6: Commit the raw sweeps into the spec (no verdict yet)**

Paste each session's `gate-decode-cap.out` and `gate-bw-curve.out` **verbatim** into the M4b.10 spec's Amendments as a dated "sweeps recorded" entry. Never edit a recorded data point.

```bash
git add docs/superpowers/specs/2026-07-12-m4b10-decode-cap-formula-design.md
git commit -m "specs: M4b.10 — record the three quiet-hw decode-cap + bandwidth sweeps"
```

---

### Task 6: Apply the pre-registered decision rule

No code. This task **reads** the spec's rule and the Task 5 data, and produces a written verdict naming exactly one formula. Doing this before Task 7 — and writing it down before touching `decode_cap` — is what keeps the choice honest.

**Files:**
- Modify: `docs/superpowers/specs/2026-07-12-m4b10-decode-cap-formula-design.md` (Amendments: the verdict)

**Interfaces:**
- Consumes: Task 5's six recorded outputs.
- Produces: the selected formula — one of **U** (remove the cap), **K_k** (static constant `k`), or **P** (runtime probe) — which Task 7 implements.

- [ ] **Step 1: Compute regret per machine**

For each machine `M`: from `gate-decode-cap.out`, take `T_best(M)` (the knee row) and compute, for each candidate cap `c`:

```
regret(c, M) = (T_best(M) − T(c, M)) / T_best(M)
```

Candidates: **U** = `active`; **K_k** = `clamp(round(k · active), 2, active)` for `k ∈ {⅓, ½, ⅔, ¾, 1}`; **P** = the `P (...)` line from `gate-bw-curve.out` on the same box.

Regret is computed **within a session**, from per-rep ratios in the same interleaved round — never ratios of medians (`lib.sh` discipline; it cancels these boxes' bimodal turbo behavior).

- [ ] **Step 2: Apply the rule, in order**

Verbatim from the spec:

1. If **regret(U) ≤ 5% on all three machines** → **remove the cap**.
2. Else if **P is validated** (regret(P) ≤ 5% on all three) **and** P beats the best static `K_k` on worst-case regret by **≥3pp** → **ship the runtime bandwidth probe**.
3. Else → **ship the static `K_k` with the lowest worst-case regret**. If it ties with U within 2pp, prefer **U**.
4. If **regret(U) > 15% on any machine**, a genuine cliff exists and a cap is mandatory; rules 2 and 3 decide which.

Do not renegotiate the thresholds against the data. That is the entire point of pre-registering them.

- [ ] **Step 3: Record the model verdict**

State explicitly, in the amendment: **did the bandwidth curve predict the decode knee?** Compare `P` against `best_fixed` per box.

If it did not, record the physical model as **refuted** — approach P is retired permanently, and that is a real finding, not a null result. The spec requires this be written down either way (§Verification protocol item 6).

- [ ] **Step 4: Commit the verdict**

```bash
git add docs/superpowers/specs/2026-07-12-m4b10-decode-cap-formula-design.md
git commit -m "specs: M4b.10 verdict — decision rule applied, formula selected"
```

---

### Task 7: Ship the selected formula

**Implement exactly ONE of the three branches below** — the one Task 6 selected. Delete nothing from this plan; just skip the branches that did not win.

**Files:**
- Modify: `crates/inferno-core/src/lib.rs:38-53` (the `decode_cap` fn) and `:179-201` (its test module)

**Interfaces:**
- Consumes: `inferno_pool::set_global_decode_threads` (already called at `crates/inferno-core/src/lib.rs:112`).
- Produces: `fn decode_cap(active: usize, override_env: Option<&str>) -> usize` — signature **unchanged** in branches U and K; branch P widens it (see below).

**Invariants for every branch:** the `INFERNO_DECODE_THREADS` override keeps precedence and must still reject garbage / `0` / empty by falling through to the formula; `active == 1` must yield `1` (the `bench-compiled` t=1 nightly depends on it); the pool re-clamps to `[1, capacity]` regardless.

#### Branch U — remove the cap (rule 1, or rule 3's tie-break)

- [ ] **U-Step 1: Rewrite the tests**

Replace the `decode_cap_tests` module at `crates/inferno-core/src/lib.rs:179-201`:

```rust
#[cfg(test)]
mod decode_cap_tests {
    use super::decode_cap;

    #[test]
    fn default_is_uncapped() {
        // M4b.10: three quiet-hw machine classes showed no bandwidth cliff —
        // uncapped is within 5% of the best fixed cap everywhere, while the
        // old active/3 heuristic gave up 10-12%. The cap prevented nothing.
        assert_eq!(decode_cap(12, None), 12);
        assert_eq!(decode_cap(32, None), 32);
        assert_eq!(decode_cap(2, None), 2);
        assert_eq!(decode_cap(1, None), 1);
    }

    #[test]
    fn env_override_wins() {
        assert_eq!(decode_cap(12, Some("1")), 1);
        assert_eq!(decode_cap(12, Some("8")), 8);
        assert_eq!(decode_cap(12, Some(" 6 ")), 6); // trimmed
    }

    #[test]
    fn garbage_override_falls_through_to_the_default() {
        assert_eq!(decode_cap(12, Some("garbage")), 12);
        assert_eq!(decode_cap(12, Some("0")), 12); // 0 rejected
        assert_eq!(decode_cap(12, Some("")), 12);
    }
}
```

- [ ] **U-Step 2: Run to verify they fail**

Run: `cargo test -p inferno-core --lib decode_cap`
Expected: FAIL — `assertion failed: left == right`, `left: 4, right: 12` (the old `active/3`).

- [ ] **U-Step 3: Implement**

Replace `crates/inferno-core/src/lib.rs:38-53`:

```rust
/// Resolve the decode-phase thread cap. An explicit `INFERNO_DECODE_THREADS`
/// override wins when it parses to a positive integer (the pool re-clamps it
/// to `[1, capacity]`); otherwise decode runs **uncapped**, at `active`.
///
/// M4b.10 retired the `clamp(active/3, 2, active)` bandwidth-knee heuristic:
/// across three quiet-hardware machine classes, uncapped landed within 5% of
/// the best fixed cap, while the heuristic gave up 10-12%. The high-thread
/// cliff the cap was built against was an artifact of the cgroup-quota'd
/// devpod that first measured it — on quiet bare metal it does not exist.
/// `active` is the engine's resolved thread count (physical cores by default).
fn decode_cap(active: usize, override_env: Option<&str>) -> usize {
    override_env
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(active)
}
```

- [ ] **U-Step 4: Run to verify they pass**

Run: `cargo test -p inferno-core --lib decode_cap`
Expected: PASS — 3 tests.

#### Branch K — refit the constant (rule 3)

Substitute the winning `k` for `K_NUM`/`K_DEN` below (e.g. `k = ¾` → `3`/`4`).

- [ ] **K-Step 1: Rewrite the tests**

Replace `crates/inferno-core/src/lib.rs:179-201` (shown for `k = ¾`; adjust the expected values to the selected `k`):

```rust
#[cfg(test)]
mod decode_cap_tests {
    use super::decode_cap;

    #[test]
    fn default_is_the_refit_knee_fraction() {
        // M4b.10: k refit against three quiet-hw machine classes. The old
        // active/3 undershot the knee by 10-12% on every one of them.
        assert_eq!(decode_cap(16, None), 12); // round(16 * 3/4)
        assert_eq!(decode_cap(12, None), 9);
        assert_eq!(decode_cap(32, None), 24);
        assert_eq!(decode_cap(2, None), 2); // floor of 2
        assert_eq!(decode_cap(1, None), 1); // never exceeds active
    }

    #[test]
    fn env_override_wins() {
        assert_eq!(decode_cap(12, Some("1")), 1);
        assert_eq!(decode_cap(12, Some("8")), 8);
        assert_eq!(decode_cap(12, Some(" 6 ")), 6); // trimmed
    }

    #[test]
    fn garbage_override_falls_through_to_the_default() {
        assert_eq!(decode_cap(12, Some("garbage")), 9);
        assert_eq!(decode_cap(12, Some("0")), 9); // 0 rejected
        assert_eq!(decode_cap(12, Some("")), 9);
    }
}
```

- [ ] **K-Step 2: Run to verify they fail**

Run: `cargo test -p inferno-core --lib decode_cap`
Expected: FAIL — `left: 4, right: 9` (the old `active/3`).

- [ ] **K-Step 3: Implement**

Replace `crates/inferno-core/src/lib.rs:38-53`:

```rust
/// Numerator/denominator of the decode-cap knee fraction, refit in M4b.10
/// against three quiet-hardware machine classes (the M4b.5 `active/3`
/// hypothesis undershot the measured knee by 10-12% on every one).
const DECODE_KNEE_NUM: usize = 3;
const DECODE_KNEE_DEN: usize = 4;

/// Resolve the decode-phase thread cap. An explicit `INFERNO_DECODE_THREADS`
/// override wins when it parses to a positive integer (the pool re-clamps it
/// to `[1, capacity]`); otherwise the refit bandwidth-knee heuristic
/// `clamp(round(active * 3/4), 2, active)` — written `.max(2).min(active)` so
/// `active == 1` yields `1` instead of a cap above the pool. `active` is the
/// engine's resolved thread count (physical cores by default).
fn decode_cap(active: usize, override_env: Option<&str>) -> usize {
    if let Some(v) = override_env
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v >= 1)
    {
        return v;
    }
    // Rounding half-up without floats: (a*num + den/2) / den.
    let knee = (active * DECODE_KNEE_NUM + DECODE_KNEE_DEN / 2) / DECODE_KNEE_DEN;
    knee.max(2).min(active)
}
```

- [ ] **K-Step 4: Run to verify they pass**

Run: `cargo test -p inferno-core --lib decode_cap`
Expected: PASS — 3 tests.

#### Branch P — ship the runtime bandwidth probe (rule 2)

The probe from Task 3 already exists and is tested. This branch wires it into pool init. Because a probe result is *measured*, `decode_cap` gains an explicit probe argument rather than reaching for global state — that keeps it a pure function and keeps its tests fast.

- [ ] **P-Step 1: Rewrite the tests**

Replace `crates/inferno-core/src/lib.rs:179-201`:

```rust
#[cfg(test)]
mod decode_cap_tests {
    use super::decode_cap;

    #[test]
    fn probe_result_is_the_default() {
        // M4b.10 rule 2: the measured bandwidth curve predicted the decode
        // knee on all three machine classes, so the cap is measured per-box
        // rather than guessed from a constant.
        assert_eq!(decode_cap(16, None, Some(13)), 13);
        assert_eq!(decode_cap(32, None, Some(24)), 24);
    }

    #[test]
    fn a_probe_above_active_is_clamped() {
        assert_eq!(decode_cap(8, None, Some(99)), 8);
    }

    #[test]
    fn a_failed_probe_falls_back_to_uncapped() {
        // Never panic, never guess: a probe that could not run leaves decode
        // uncapped, which the same sweeps showed is within 5% of the knee.
        assert_eq!(decode_cap(12, None, None), 12);
        assert_eq!(decode_cap(1, None, None), 1);
    }

    #[test]
    fn env_override_wins_and_skips_the_probe() {
        assert_eq!(decode_cap(12, Some("8"), Some(4)), 8);
        assert_eq!(decode_cap(12, Some(" 6 "), None), 6); // trimmed
    }

    #[test]
    fn garbage_override_falls_through_to_the_probe() {
        assert_eq!(decode_cap(12, Some("garbage"), Some(9)), 9);
        assert_eq!(decode_cap(12, Some("0"), Some(9)), 9); // 0 rejected
        assert_eq!(decode_cap(12, Some(""), Some(9)), 9);
    }
}
```

- [ ] **P-Step 2: Run to verify they fail**

Run: `cargo test -p inferno-core --lib decode_cap`
Expected: FAIL to compile — `this function takes 2 arguments but 3 arguments were supplied`.

- [ ] **P-Step 3: Implement the resolver**

Replace `crates/inferno-core/src/lib.rs:38-53`:

```rust
/// Resolve the decode-phase thread cap. Precedence: an explicit
/// `INFERNO_DECODE_THREADS` override (when it parses to a positive integer),
/// then the measured bandwidth knee from `probed`, then uncapped.
///
/// M4b.10 retired the `clamp(active/3, 2, active)` constant: across three
/// quiet-hardware machine classes the measured bandwidth curve predicted the
/// decode knee, so the cap is *measured* on the host rather than guessed from
/// a fitted constant that was wrong on every box we rented. A probe that
/// could not run falls back to uncapped — the same sweeps put uncapped within
/// 5% of the knee, so the fallback is safe, never a guess.
///
/// The pool re-clamps to `[1, capacity]` regardless; the clamp here keeps the
/// resolved value honest for logging and tests.
fn decode_cap(active: usize, override_env: Option<&str>, probed: Option<usize>) -> usize {
    if let Some(v) = override_env
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&v| v >= 1)
    {
        return v;
    }
    probed.map_or(active, |p| p.clamp(1, active))
}
```

- [ ] **P-Step 4: Run to verify they pass**

Run: `cargo test -p inferno-core --lib decode_cap`
Expected: PASS — 5 tests.

- [ ] **P-Step 5: Wire the probe into pool init**

At `crates/inferno-core/src/lib.rs:110-112` the call site currently reads:

```rust
        // `INFERNO_DECODE_THREADS` overrides the heuristic.
        let env = std::env::var("INFERNO_DECODE_THREADS").ok();
        inferno_pool::set_global_decode_threads(decode_cap(self.threads, env.as_deref()));
```

Replace with:

```rust
        // `INFERNO_DECODE_THREADS` overrides the probe; a probe that cannot
        // run leaves decode uncapped (M4b.10).
        let env = std::env::var("INFERNO_DECODE_THREADS").ok();
        let probed = if env.is_some() {
            None // the override wins anyway — don't pay for a probe we'd discard
        } else {
            self.probe_decode_knee()
        };
        inferno_pool::set_global_decode_threads(decode_cap(
            self.threads,
            env.as_deref(),
            probed,
        ));
```

Then add the probe driver as a private method on the same `impl` block. It reuses the exact shape and constants the `bw_curve` example validated, so the shipped probe and the measured curve are the same measurement:

```rust
    /// Measure this host's decode knee: the smallest lane count reaching 95%
    /// of peak streaming bandwidth on the real Q8_0 GEMV. Returns `None` if
    /// the kernel set or the packed buffers cannot be built, in which case
    /// the caller leaves decode uncapped (M4b.10 rule 2).
    ///
    /// Bounded on purpose: 4096 rows x 4096 k is ~18 MiB, enough to leave L2
    /// on every supported target while keeping the probe in the low
    /// milliseconds, and the pool is already warm from `init_global`.
    fn probe_decode_knee(&self) -> Option<usize> {
        use inferno_formats::DType;
        use inferno_kernels::{KernelIsa, kernels_for, reference_kernels};

        const ROWS: usize = 4096;
        const K: usize = 4096;
        const REPS: usize = 3;
        const KNEE_FRACTION: f64 = 0.95;

        let isa = if KernelIsa::Avx2.available() {
            KernelIsa::Avx2
        } else {
            KernelIsa::Scalar
        };
        let kernel: inferno_pool::GemvFn = match isa {
            KernelIsa::Avx2 => inferno_kernels::inferno_gemv_q8_0_rs8_avx2,
            KernelIsa::Scalar => inferno_kernels::inferno_gemv_q8_0_rs8_scalar,
        };
        let ks = kernels_for(&DType::Q8_0, isa).or_else(|| reference_kernels(&DType::Q8_0))?;

        let vals: Vec<f32> = (0..ROWS * K).map(|i| (i % 17) as f32 / 17.0 - 0.5).collect();
        let wbytes = inferno_formats::quant::pack(&DType::Q8_0, &vals).ok()?;
        let w = ks.pack(&wbytes, ROWS, K).ok()?;
        let xq = ks
            .quantize_row(&vals[..K].iter().copied().collect::<Vec<f32>>())
            .ok()?;
        let stream_bytes = ks.packed_len(ROWS, K);
        let mut y = vec![f32::NAN; ROWS];

        let pool = inferno_pool::global()?;
        let lanes: Vec<usize> = (1..=self.threads).collect();
        // SAFETY: w/xq built here for exactly (ROWS, K); y has ROWS f32s;
        // `kernel` is the Q8_0 GEMV symbol for the detected ISA.
        let curve = unsafe {
            inferno_pool::bandwidth_curve(
                pool,
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
        Some(inferno_pool::knee_at_fraction(&curve, KNEE_FRACTION))
    }
```

This needs a `&'static Pool` accessor, which `inferno-pool` does not yet expose (only `set_global_*` helpers). Add it to `crates/inferno-pool/src/lib.rs`, after `set_global_decode_threads` (line 68):

```rust
/// The process-global pool, or `None` if [`init_global`] has not run.
/// Exposed for the M4b.10 decode-knee probe, which needs to dispatch through
/// the real pool to measure real lane scaling.
pub fn global() -> Option<&'static Pool> {
    GLOBAL.get()
}
```

`inferno-core` already lists both `inferno-kernels` and `inferno-formats` in its `[dependencies]` (verified in `crates/inferno-core/Cargo.toml`), so **no manifest change is needed**.

- [ ] **P-Step 6: Verify the probe does not perturb the t=1 path**

Run: `cargo test -p inferno-core --lib`
Expected: PASS.

Run: `mise run bench-compiled`
Expected: green. At `active == 1` the lane sweep is `[1]`, so the knee is 1 — the cap resolves to 1 exactly as before, and the pinned t=1 nightly is unaffected by construction.

#### Every branch: verify the invariants and commit

- [ ] **Step 8: The cap is still bit-invisible**

Run: `cargo test -p inferno-pool --test par_rig`
Expected: PASS — the cap-invariance sweep (`1..=capacity`) is the standing guard that the cap only regroups rows and never changes output bits.

- [ ] **Step 9: The differentials are green with no tolerance edits**

Run:
```bash
cargo test -p inferno-codegen --test differential
cargo test -p inferno-core --test artifact
```
Expected: PASS, both. If either fails, **stop** — do not touch `logits_abs_tol` or any other tolerance (AGENTS.md standing rule). A red differential here means the change was not cap-only.

- [ ] **Step 10: Confirm no ABI bump and no recompile**

Run: `git diff --stat HEAD -- crates/inferno-codegen`
Expected: **empty**. The spec forbids a codegen edit or a `HOST_ABI_VERSION` bump; a cached `model.so` must benefit from the new cap immediately.

- [ ] **Step 11: Full suite + lint**

Run: `mise run test && mise run lint`
Expected: both green.

- [ ] **Step 12: Commit**

```bash
git add crates/inferno-core/src/lib.rs crates/inferno-pool/src/lib.rs
git commit -m "core: ship the M4b.10 decode-cap formula (<U|K|P> per the pre-registered rule)"
```

---

### Task 8: Close the ledgers

The verdict lives in **M4b.5's** spec — that is the ledger carrying three recorded misses and the DEFERRED exit-criterion leg 2, open since 2026-07-08. This task closes it.

**Files:**
- Modify: `docs/superpowers/specs/2026-07-08-m4b5-phase-aware-decode-threading-design.md` (Amendments: leg-2 verdict)
- Modify: `docs/superpowers/specs/2026-07-12-m4b10-decode-cap-formula-design.md` (Amendments: shipped result)
- Modify: `docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md` (Amendments: re-recorded headline)
- Modify: `AGENTS.md:76-91` (the decode-threading constraint bullet)
- Modify: `mise.toml:67` (the `bench` task's "record data points in ..." pointer)

**Interfaces:**
- Consumes: Task 7's shipped formula; a final quiet-hw session.

- [ ] **Step 1: Re-record the headline on quiet hardware**

```bash
mise run metal -- d2.c1.medium --yes -- \
  'bash scripts/quiet-hw/verify.sh models/qwen2.5-0.5b-instruct-q8_0.gguf'
```

This is the same protocol as every prior session, now with the shipped cap. Paste `gate-decode-cap.out` and `gate-bench-protocol.out` verbatim.

- [ ] **Step 2: Record the M4b.5 leg-2 verdict**

Append an amendment to the M4b.5 spec stating the exit criterion's two legs:

- **Leg 1 (correctness):** already PROVEN (2026-07-08).
- **Leg 2 (performance):** the shipped default's **worst-case regret across all three machine classes**, against the ≤5% criterion → **MET** or **NOT MET**.

If MET, state plainly that **M4b.5's exit criterion is now satisfied and leg 2 is closed**, and cross-link the M4b.10 spec.

- [ ] **Step 3: Record the M4a headline honestly**

Append the new `gate-bench-protocol` block to the M4a spec's Amendments. The expected move is **tg 0.84x → ~0.94x**.

If tg is still below 1x, record the v1 win criterion as **NOT MET**. Do not loosen the gate; it stays owned by the M4a spec. Note in the same amendment that decode attention is still serial (M4b.2's fork), which is the next lever — and that it is now measurable against a de-throttled baseline, which was the point of this milestone.

- [ ] **Step 4: Update `AGENTS.md`**

The decode-threading bullet (`AGENTS.md:76-91`) currently reads, in part:

> Default `clamp(active/3, 2, active)`, override with `INFERNO_DECODE_THREADS=N`.

Rewrite that sentence to state the shipped formula. Keep the surrounding invariants verbatim — they are still true and load-bearing:

> The cap is bit-neutral (`shard_table` keeps each row on one lane); never treat a cap change as a numeric change.

- [ ] **Step 5: Update the stale `mise.toml` pointer**

`mise.toml:67`'s `bench` task description points recorders at the M4b.6 spec. Repoint it at the current milestone spec so the next operator records data in the right ledger.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-07-08-m4b5-phase-aware-decode-threading-design.md \
        docs/superpowers/specs/2026-07-12-m4b10-decode-cap-formula-design.md \
        docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md \
        AGENTS.md mise.toml
git commit -m "specs: M4b.10 closes M4b.5's exit-criterion leg 2; re-record the M4a headline"
```

- [ ] **Step 7: Open the PR**

```bash
gh pr create --title "M4b.10: decode-cap formula revision" --body "$(cat <<'EOF'
Closes M4b.5's exit-criterion leg 2, DEFERRED since 2026-07-08.

The shipped `clamp(active/3, 2, active)` decode cap missed on quiet bare
metal three times (-9.8% / -11.2% / -11.8% vs the best fixed cap). This
replaces it with a formula selected by a decision rule pre-registered
before any data was taken, against sweeps on three machine classes (16c,
8c, and a socket-pinned 32c), each pairing the decode-knee sweep with a
new bandwidth-saturation curve.

Pool-side only: no codegen edit, no HOST_ABI_VERSION bump, no recompile.
The cap remains bit-invisible (par_rig cap-invariance sweep green); no
tolerance touched.
EOF
)"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: the bandwidth curve and its knee → Task 3; the `bw_curve` example and gate → Task 4; `gate-decode-cap`'s coarse grid and NUMA pinning → Tasks 1–2; the three machine sessions → Task 5; the pre-registered rule and the model verdict (§Verification protocol item 6) → Task 6; the three formula shapes → Task 7's three branches; the exit criterion, the M4a re-record, and closing M4b.5's leg 2 → Task 8. The spec's invariants (bit-invisibility, no tolerance edits, no ABI bump, `INFERNO_DECODE_THREADS` survives) are Global Constraints and are re-verified in Task 7 Steps 8–10.

**Placeholders.** None. Task 7's branches are all fully written; `K_NUM`/`K_DEN` is the one value the rule supplies, and the branch shows a worked `k = ¾` so the shape is unambiguous.

**Type consistency.** `knee_at_fraction(&[(usize, f64)], f64) -> usize` and `bandwidth_curve(...) -> Vec<(usize, f64)>` are defined in Task 3 and consumed with matching signatures in Task 4's example and Task 7's Branch P. `GemvFn` matches `crates/inferno-pool/src/pool.rs:20` exactly. `decode_cap`'s signature is unchanged in branches U and K and explicitly widened to three arguments in branch P, where the call site at `crates/inferno-core/src/lib.rs:112` is updated in the same step. `inferno_pool::global()` is new and is added in the one branch that needs it.

**Dependencies.** Verified, not assumed: `inferno-core` already lists `inferno-kernels` and `inferno-formats` in `[dependencies]`, and `inferno-pool` already carries `inferno-kernels` + `inferno-formats` as dev-dependencies (which examples may use). No branch of this plan adds a manifest entry. The `expect <label> <got> <want>` helper used in Task 1 matches `lib-selftest.sh`'s existing signature.
