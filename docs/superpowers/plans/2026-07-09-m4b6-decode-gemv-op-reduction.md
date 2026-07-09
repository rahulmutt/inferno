# M4b.6 Decode GEMV Reduce/Combine Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the true wall-time share of the per-block `hsum8_i32` + f32 combine in the Q8_0 AVX2 GEMV via bench-only ceiling arms, project the maximum end-to-end decode win, and record the spec's PROCEED/STOP gate decision.

**Architecture:** Two cost-model copies of the AVX2 full-strip GEMV land in the criterion bench (never the library): arm A stubs both the transpose-reduce and the f32 combine; arm B keeps the reduce and stubs only the combine. Same-box baseline-vs-arm time ratios, weighted by `inferno run --profile` decode cycle shares, produce the projected decode-wall reduction the gate judges.

**Tech Stack:** Rust, criterion benches (`crates/inferno-kernels/benches/gemv.rs`), AVX2 intrinsics (`std::arch::x86_64`), `inferno` CLI `--profile`.

**Spec:** `docs/superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md`. This plan covers the spec's **Task 1 (diagnostic + gate)** only. The gated restructure's candidate is picked by this diagnostic's attribution, so it gets its own plan after the gate amendment is recorded (spec §Design; M4b.4 precedent).

## Global Constraints

- **Ratio-only signal:** same-box baseline-vs-arm time ratios; absolute GB/s recorded but untrusted on the shared devpod (spec §Scope Decisions "Measurement discipline").
- **Arms are bench-only:** they live in `benches/gemv.rs`, produce wrong numbers by design, and must never ship in the library (spec §Task 1).
- **No numeric surface is touched in this phase:** no kernel, tolerance, ABI, or interpreter change. `git diff` on `crates/inferno-kernels/src/`, `crates/inferno-graph/src/tolerance.rs`, `crates/inferno-codegen/src/lib.rs` stays empty through this whole plan.
- **Gate thresholds (verbatim from spec):** projected decode-wall reduction **≥5% → proceed**; **3–5% → controller judgment call**, recorded as a spec amendment either way; **<3% → STOP**.
- **Recording:** the gate decision, per-shape table, and projection land in the spec's `## Amendments` section; never edit a recorded data point.
- **Before every commit:** `mise run lint` (rustfmt + clippy `-D warnings`) and `mise run test` must be green. CI's lint is stricter than the test task — do not skip it.
- **Benches run inside `devenv shell`** (`devenv shell -- cargo bench ...`); the devpod is directional-only, which is exactly what the gate's ratio reading assumes.

---

### Task 1: Ceiling arms in the GEMV bench

**Files:**
- Modify: `crates/inferno-kernels/benches/gemv.rs` (module at end of file; registration inside `bench_dtype`, after the stream-read arm at lines 106–130)

**Interfaces:**
- Consumes: `inferno_kernels::STRIP` (pub, = 8), `inferno_kernels::registry::kernels_for`, `KernelSet::{pack, quantize_row}` (already imported in this file); the rs8 packed layout (`GROUP_BYTES = 288`, group = 8 f32 `d` + 8×32 qs) and q8a activation layout (`Q8A_BLOCK_BYTES = 36`, block = f32 `d` + 32 i8) mirrored as local consts, exactly as `q8_0.rs:10-19` defines them.
- Produces: criterion benchmark ids `gemv/Q8_0/reduce-ceiling/{rows}x{k}` and `gemv/Q8_0/combine-stub/{rows}x{k}` for every Q8_0 shape — Task 2 measures these against `gemv/Q8_0/inferno-avx2/{rows}x{k}` and `gemv/Q8_0/stream-read/{rows}x{k}`.

There is no unit test for a cost-model arm (its outputs are wrong by design); its test cycle is criterion's `--test` mode (each bench runs once, catching panics/segfaults) plus Task 2's ordering sanity checks.

- [ ] **Step 1: Add the ceiling-arm module at the end of `benches/gemv.rs`**

Append after the final line (`criterion_main!(gemv, gemm);` stays last — insert this module immediately *before* the `fn benches(...)` block at line 314):

```rust
/// M4b.6 Task 1 diagnostic arms (spec 2026-07-09-m4b6): copies of the AVX2
/// full-strip GEMV body with the per-block reduce/combine progressively
/// stubbed. Wrong results by design — these are cost models, not kernels,
/// and must never ship in the library. Layout consts mirror `q8_0.rs`.
///
/// Conservative by construction: keeping each lane's dot product live costs
/// one `vpaddd` per lane per block (arm A) or one per block (arm B), so the
/// measured baseline-vs-arm delta *understates* the true reduce/combine cost
/// slightly — the gate can under-claim headroom, never over-claim it.
#[cfg(target_arch = "x86_64")]
mod reduce_ceiling {
    use std::arch::x86_64::__m256i;

    use inferno_kernels::STRIP;

    const WBLOCK: usize = 32; // weight elements per block (q8_0.rs:10)
    const GROUP_BYTES: usize = 288; // 8 f32 d + 8×32 qs (q8_0.rs:12)
    const Q8A_BLOCK_BYTES: usize = 36; // f32 d + 32 i8 (act.rs:15)
    const PF_DIST: usize = 4; // mirrors q8_0.rs:19 so arms model the shipped kernel

    /// Copied verbatim from `q8_0.rs::hsum8_i32` (pub(crate) there, so the
    /// bench keeps its own copy): transpose-reduce 8 lane-parallel i32
    /// accumulators into one vector whose lane `i` holds v[i]'s horizontal sum.
    #[target_feature(enable = "avx2")]
    fn hsum8_i32(v: [__m256i; 8]) -> __m256i {
        use std::arch::x86_64::*;
        let s01 = _mm256_hadd_epi32(v[0], v[1]);
        let s23 = _mm256_hadd_epi32(v[2], v[3]);
        let s45 = _mm256_hadd_epi32(v[4], v[5]);
        let s67 = _mm256_hadd_epi32(v[6], v[7]);
        let s0123 = _mm256_hadd_epi32(s01, s23);
        let s4567 = _mm256_hadd_epi32(s45, s67);
        let lo = _mm256_permute2x128_si256::<0x20>(s0123, s4567);
        let hi = _mm256_permute2x128_si256::<0x31>(s0123, s4567);
        _mm256_add_epi32(lo, hi)
    }

    /// Arm A — "reduce+combine → 0". Full weight/activation streaming, the
    /// prefetch hint, and the complete int8 dot (sign×2/maddubs/madd per lane)
    /// are intact; `hsum8_i32` and the f32 combine (dx load, cvt, dw load,
    /// mul, fmadd) are replaced by one `vpaddd` per lane into a sink that is
    /// stored once per strip. Lower bound on kernel time if the per-block
    /// reduce/combine were free.
    ///
    /// # Safety
    /// As `inferno_gemv_q8_0_rs8_avx2` (`y` writable for `rows` f32; `x` a
    /// q8a buffer for this `k`; `w` an rs8 pack for exactly this `k`/`rows`,
    /// 32-byte aligned; `k` a positive multiple of 32), plus: whole strips
    /// only (`rows % STRIP == 0`) and AVX2+FMA present.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemv_no_reduce_no_combine(
        y: *mut f32,
        x: *const u8,
        w: *const u8,
        k: usize,
        rows: usize,
    ) {
        use std::arch::x86_64::*;
        let nb = k / WBLOCK;
        let ones = _mm256_set1_epi16(1);
        for strip in 0..rows / STRIP {
            let mut sink = _mm256_setzero_si256();
            for b in 0..nb {
                let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                // Same pure-hint prefetch as the shipped kernel (q8_0.rs:160-163).
                let pf_addr = w
                    .wrapping_add((strip * nb + b + PF_DIST) * GROUP_BYTES)
                    .cast();
                _mm_prefetch::<_MM_HINT_T0>(pf_addr);
                let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
                let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                let qs = unsafe { g.add(32) };
                for lane in 0..STRIP {
                    let wv = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                    let aw = _mm256_sign_epi8(wv, wv);
                    let sx = _mm256_sign_epi8(xv, wv);
                    let p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                    // Cheapest possible liveness: without this add the whole
                    // lane's dot chain is dead and the optimizer deletes it.
                    sink = _mm256_add_epi32(sink, p);
                }
            }
            // One garbage store per strip keeps `sink` observable.
            unsafe { _mm256_storeu_ps(y.add(strip * STRIP), _mm256_castsi256_ps(sink)) };
        }
    }

    /// Arm B — "combine → 0, reduce kept". Identical to arm A except the
    /// per-block `hsum8_i32` transpose-reduce still runs; only the f32
    /// combine is stubbed (one `vpaddd` of the reduced vector into the sink).
    /// Attribution: (baseline − B) ≈ combine cost, (B − A) ≈ reduce cost.
    ///
    /// # Safety
    /// As [`gemv_no_reduce_no_combine`].
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemv_no_combine(y: *mut f32, x: *const u8, w: *const u8, k: usize, rows: usize) {
        use std::arch::x86_64::*;
        let nb = k / WBLOCK;
        let ones = _mm256_set1_epi16(1);
        for strip in 0..rows / STRIP {
            let mut sink = _mm256_setzero_si256();
            for b in 0..nb {
                let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                let pf_addr = w
                    .wrapping_add((strip * nb + b + PF_DIST) * GROUP_BYTES)
                    .cast();
                _mm_prefetch::<_MM_HINT_T0>(pf_addr);
                let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
                let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                let qs = unsafe { g.add(32) };
                let mut p = [_mm256_setzero_si256(); STRIP];
                for (lane, pl) in p.iter_mut().enumerate() {
                    let wv = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                    let aw = _mm256_sign_epi8(wv, wv);
                    let sx = _mm256_sign_epi8(xv, wv);
                    *pl = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                }
                // Reduce kept, combine stubbed: cvt/dw/dx/mul/fmadd → one add.
                sink = _mm256_add_epi32(sink, hsum8_i32(p));
            }
            unsafe { _mm256_storeu_ps(y.add(strip * STRIP), _mm256_castsi256_ps(sink)) };
        }
    }
}
```

- [ ] **Step 2: Register the arms in `bench_dtype`**

Inside the `for &(rows, k) in shapes` loop, immediately after the stream-read arm's closing brace (currently line 130, before the `#[cfg(feature = "ggml-compare")]` line):

```rust
        // M4b.6 Task 1: reduce/combine ceiling arms (cost models, wrong
        // numbers by design — see the reduce_ceiling module docs). Q8_0 only;
        // every SHAPES_Q8_0 rows value is a multiple of STRIP, asserted here
        // so a future shape can't silently hit the arms' whole-strip limit.
        #[cfg(target_arch = "x86_64")]
        if matches!(dtype, DType::Q8_0) && std::arch::is_x86_feature_detected!("avx2") {
            assert_eq!(rows % inferno_kernels::STRIP, 0, "ceiling arms need whole strips");
            let set = kernels_for(&dtype, Isa::X86_64v3).unwrap();
            let w = set.pack(&file, rows, k).unwrap();
            let xq = set.quantize_row(&x).unwrap();
            let mut y = vec![0f32; rows];
            group.throughput(Throughput::Bytes(w.len() as u64));
            group.bench_function(
                BenchmarkId::new("reduce-ceiling", format!("{rows}x{k}")),
                |b| {
                    b.iter(|| {
                        // SAFETY: y/xq/w built above for exactly this rows/k;
                        // rows % STRIP == 0 asserted; AVX2 runtime-detected.
                        unsafe {
                            reduce_ceiling::gemv_no_reduce_no_combine(
                                y.as_mut_ptr(),
                                xq.as_slice().as_ptr(),
                                w.as_slice().as_ptr(),
                                k,
                                rows,
                            );
                        }
                        std::hint::black_box(y[0]);
                    })
                },
            );
            group.bench_function(BenchmarkId::new("combine-stub", format!("{rows}x{k}")), |b| {
                b.iter(|| {
                    // SAFETY: as the reduce-ceiling arm above.
                    unsafe {
                        reduce_ceiling::gemv_no_combine(
                            y.as_mut_ptr(),
                            xq.as_slice().as_ptr(),
                            w.as_slice().as_ptr(),
                            k,
                            rows,
                        );
                    }
                    std::hint::black_box(y[0]);
                })
            });
        }
```

Note: `xq.as_slice()` works whether `quantize_row` returns `AlignedBuf` or `Vec<u8>`; the stream-read arm already uses `w.as_slice()` the same way.

- [ ] **Step 3: Compile-check the bench**

Run: `cargo bench -p inferno-kernels --bench gemv --no-run`
Expected: compiles cleanly (finishes with `Finished` + an executable path; no warnings — clippy runs in Step 5).

- [ ] **Step 4: Smoke-run every arm once in criterion test mode**

Run: `cargo bench -p inferno-kernels --bench gemv -- --test 'gemv/Q8_0'`
Expected: every `gemv/Q8_0/...` bench (including `reduce-ceiling` and `combine-stub` for all six shapes) prints `Testing ... Success`; no panic, no segfault. This is the arms' correctness bar — they compute garbage on purpose, so "runs without fault on every shape" is the whole test.

- [ ] **Step 5: Lint and full test suite**

Run: `mise run lint && mise run test`
Expected: rustfmt/clippy clean (no new `#[allow]`), all workspace tests pass (the arms touch no library code, so the count matches main).

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-kernels/benches/gemv.rs
git commit -m "bench(kernels): reduce/combine ceiling arms for M4b.6 decode GEMV diagnostic"
```

---

### Task 2: Measure baseline vs arms (three interleaved reps)

**Files:**
- Create: `/tmp/claude-1000/-workspace/5d5a8b95-3b48-4176-9630-04913a993610/scratchpad/m4b6-task2-table.md` (scratch working table; the durable record lands in the spec in Task 4)

**Interfaces:**
- Consumes: the four benchmark ids per Q8_0 shape from Task 1 (`inferno-avx2`, `stream-read`, `reduce-ceiling`, `combine-stub`).
- Produces: a per-shape table of median times and derived ratios — columns `shape | t_base | t_B (combine-stub) | t_A (reduce-ceiling) | t_stream | headroom_A = 1 − t_A/t_base | combine share = (t_base − t_B)/t_base | reduce share = (t_B − t_A)/t_base` — with one row per shape per rep. Task 3 consumes `headroom_A` per shape; Task 4 records the table.

- [ ] **Step 1: Run the Q8_0 GEMV group, rep 1**

Run: `devenv shell -- cargo bench -p inferno-kernels --bench gemv -- 'gemv/Q8_0' 2>&1 | tee /tmp/claude-1000/-workspace/5d5a8b95-3b48-4176-9630-04913a993610/scratchpad/m4b6-rep1.out`
Expected: criterion output blocks like

```
gemv/Q8_0/reduce-ceiling/151936x896
                        time:   [XX.X ms XX.X ms XX.X ms]
                        thrpt:  [XX.XX GiB/s ...]
```

for all four arms × six shapes. Takes several minutes (sample_size 20, six shapes).

- [ ] **Step 2: Reps 2 and 3, back-to-back**

Run the identical command twice more, teeing to `m4b6-rep2.out` and `m4b6-rep3.out`. Back-to-back full-group runs are the interleaving unit: each rep yields one within-run ratio sample per shape, and within-run comparisons are the trustworthy signal on this box (M4b.4 protocol).

- [ ] **Step 3: Extract medians and compute the ratio table**

For each rep and each shape, take the **middle** value of each `time: [low mid high]` triple. Compute per rep:
- `headroom_A = 1 − t_reduce-ceiling / t_inferno-avx2`
- `combine_share = (t_inferno-avx2 − t_combine-stub) / t_inferno-avx2`
- `reduce_share = (t_combine-stub − t_reduce-ceiling) / t_inferno-avx2`

Write all three reps' rows into `m4b6-task2-table.md`, then a summary row per shape with the **median headroom_A across reps** and the min–max range. Report ratios only; absolute GiB/s columns are recorded in the table but flagged untrusted.

- [ ] **Step 4: Sanity-check the arms (gate the data, not just the code)**

Verify, on every rep, for the DRAM-bound shape `151936x896`:
- `t_stream-read < t_reduce-ceiling < t_inferno-avx2` — the arm cannot beat pure streaming (if it does, the dot got deleted: check the disassembly for missing `vpmaddubsw` before trusting anything) and cannot be slower than the full kernel (if it is, the sink spilled: look for `vmovdqa` stack traffic in the loop).
- `t_reduce-ceiling ≤ t_combine-stub ≤ t_inferno-avx2` on every shape (A stubs strictly more than B).

If any ordering is violated on ≥2 reps, STOP this task and fix the arm (the disassembly check: `objdump -d` the bench executable from Task 1 Step 3's path, find `gemv_no_reduce_no_combine`, confirm the loop still contains `vpmaddubsw`/`vpmaddwd` and no stack stores). Do not record data from a broken arm.

- [ ] **Step 5: Commit the scratch table reference**

No repo commit — bench outputs are scratch until Task 4 records the amendment. Confirm `git status` shows a clean tree.

---

### Task 3: Profile-weighted decode-win projection

**Files:**
- Modify: `/tmp/claude-1000/-workspace/5d5a8b95-3b48-4176-9630-04913a993610/scratchpad/m4b6-task2-table.md` (append the projection section)

**Interfaces:**
- Consumes: per-shape median `headroom_A` from Task 2; the `--profile` decode table from the CLI (`profile [decode] ...` block: rows of `op | cycles | share | GB/s`, ops named like `matmul:blk.*.attn_q.weight`).
- Produces: `projected_decode_win = Σ_slot share_slot × headroom_A(shape(slot))` — a single percentage Task 4 judges against the gate.

- [ ] **Step 1: Run the pinned decode profile at t=1**

Run:

```bash
devenv shell -- cargo run --release -p inferno -- run \
  /home/dev/.cache/inferno-tests/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "The quick brown fox jumps over the lazy dog." \
  --max-tokens 64 --threads 1 --profile \
  2>&1 | tee /tmp/claude-1000/-workspace/5d5a8b95-3b48-4176-9630-04913a993610/scratchpad/m4b6-profile-t1.out
```

Expected: two tables, `profile [prefill]` and `profile [decode]`, each row `op | cycles | share% | GB/s`. Use the **decode** table only. `--threads 1` matches the kernel arms' single-thread basis; the shares are cycle-structural, so devpod noise matters far less here than for absolute GB/s.

- [ ] **Step 2: Map decode matmul slots to bench shapes**

Use exactly this mapping (Qwen2.5-0.5B geometry):

| profile slot (matmul:) | shape (rows×k) | headroom source |
|---|---|---|
| `blk.*.attn_q.weight`, `blk.*.attn_output.weight` | 896×896 | measured |
| `blk.*.attn_k.weight`, `blk.*.attn_v.weight` | 128×896 | **approximation:** use the 896×896 ratio (shape not in the bench set; small share) — flag it in the table |
| `blk.*.ffn_gate.weight`, `blk.*.ffn_up.weight` | 4864×896 | measured |
| `blk.*.ffn_down.weight` | 896×4864 | measured |
| `output.weight` (lm_head) | 151936×896 | measured |
| every non-matmul op (rmsnorm, rope, attention, swiglu, add, quantize, bias) | — | headroom 0 |

- [ ] **Step 3: Compute and record the projection**

`projected_decode_win = Σ share_slot × headroom_A(shape(slot))`, shares as fractions of decode cycles from the profile table. Show the arithmetic line-by-line in `m4b6-task2-table.md` (slot, share, headroom, product), then the sum as a percentage. Also compute the same sum with `combine_share` and with `reduce_share` in place of `headroom_A` — that attribution split is what picks the candidate if the gate passes (combine-dominated → candidate 2/3; reduce-dominated → candidate 1 is in play).

- [ ] **Step 4: Confirm clean tree**

Run: `git status`
Expected: clean — nothing from Tasks 2–3 belongs in the repo yet.

---

### Task 4: Gate decision, spec amendment, doc hygiene

**Files:**
- Modify: `docs/superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md` (append to `## Amendments`)
- Modify: `mise.toml:66` (bench task description still points at the M4b.4 spec)

**Interfaces:**
- Consumes: the Task 2 table, Task 3 projection and attribution split.
- Produces: the recorded gate decision that either closes the milestone (STOP) or authorizes writing the restructure plan (PROCEED), plus the candidate the attribution points at.

- [ ] **Step 1: Apply the gate rule**

From Global Constraints, verbatim: `projected_decode_win ≥ 5%` → PROCEED; `3–5%` → controller judgment call (present the number, the attribution split, and the candidate it implies to the user and ask; bit-neutral candidate 1 deserves a lower effective bar than contract-changing candidates 2–3); `< 3%` → STOP.

- [ ] **Step 2: Write the amendment**

Append to the spec's `## Amendments` using this skeleton (fill every bracket from the scratch files; never edit it after commit):

```markdown
### 2026-07-09 — Task 1 diagnostic: reduce/combine ceiling — [PROCEED / STOP / judgment: ...]

- **Commit:** [Task 1 commit hash] (`bench(kernels): reduce/combine ceiling arms ...`).
- **Command:** `devenv shell -- cargo bench -p inferno-kernels --bench gemv -- 'gemv/Q8_0'`, 3 back-to-back reps; full outputs in scratch (`m4b6-rep{1,2,3}.out`).
- **Environment caveat:** shared 24-core devpod (AMD Ryzen 9 3900, 12C/24T); ratio-only reading per standing protocol; absolute GiB/s recorded but untrusted.

#### Per-shape baseline vs arms (median across 3 reps; range in parens)

| shape | t_base | t_combine-stub | t_reduce-ceiling | t_stream-read | headroom_A | combine share | reduce share |
|---|---|---|---|---|---|---|---|
| [six rows] |

Sanity orderings (Task 2 Step 4) held on [N/3] reps: [detail].

#### Profile weighting (t=1 decode, pinned qwen2.5-0.5b Q8_0, 64 steps)

| slot | decode share | shape | headroom_A | product |
|---|---|---|---|---|
| [rows; flag the 128×896 ≈ 896×896 approximation] |

**projected_decode_win = [X.X]%** (combine-only: [X.X]%, reduce-only: [X.X]%).

#### Gate decision

[≥5% / 3–5% / <3%] → **[PROCEED to the restructure plan, candidate N per the
attribution / STOP — the decode GEMV inner loop is exhausted as a lever; the
tg win effort moves to the quiet-hardware verification pass (natural M4b.7)]**.
```

- [ ] **Step 3: Repoint the bench-task description (doc hygiene, in scope per M4b.4 precedent)**

In `mise.toml` line 66, change the parenthetical `(M4b.4: docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md)` to `(M4b.6: docs/superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md)`. No other edit.

- [ ] **Step 4: Lint, test, commit**

Run: `mise run lint && mise run test`
Expected: green (docs + one string change only).

```bash
git add docs/superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md mise.toml
git commit -m "docs(specs): record M4b.6 Task 1 diagnostic + gate decision; repoint bench task to M4b.6"
```

- [ ] **Step 5: Hand off per the gate**

- **STOP:** the milestone is complete (spec §Measurement & Exit Criterion, STOP branch). Report the projection and close.
- **PROCEED / judgment-PROCEED:** invoke the writing-plans skill to author the restructure plan for the candidate the attribution picked (spec §Task 2+, including the lockstep contract-change discipline if candidates 2–3 won). Do not start restructure code without that plan.
