# M4b.4 — Decode GEMV Memory-Level Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the effective memory bandwidth of the decode-path single-row GEMV (`inferno_gemv_*_rs8_avx2`) from ~16 GB/s toward prefill's ~40 GB/s by increasing memory-level parallelism, without changing a single output bit.

**Architecture:** The AVX2 full-strip GEMV path streams contiguous packed weights but issues too few in-flight cache-line loads (one token, no reuse) to hide DRAM latency. We add software prefetch of upcoming weight groups and — only if prefetch alone falls short — process multiple strips concurrently for more outstanding load streams. Both transforms leave the per-block f32 `fmadd` accumulation order untouched, so bit-identity holds by construction and is guarded by the existing `inferno-kernels` proptest rig.

**Tech Stack:** Rust, x86-64 AVX2/FMA intrinsics (`std::arch::x86_64`), criterion benches (`inferno-kernels/benches/gemv.rs`), the devenv-pinned ggml CPU backend for side-by-side comparison, `inferno bench` (vs llama.cpp) for the end-to-end leg.

## Global Constraints

Every task implicitly includes these. Values copied verbatim from the spec (`docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md`) and `AGENTS.md`:

- **Only the AVX2 GEMV bodies change.** Scalar GEMV stays untouched — it is the independent reference oracle.
- **Bit-identity is the hard invariant:** scalar ≡ AVX2 (exact) and `gemm(m=1)` ≡ `gemv` (exact). Prefetch is a hint; strip-interleave only reorders row iteration. Never reorder or re-fuse the per-block f32 combine.
- **The existing rig proptests are the correctness gate** — `q8_0_isa_variants_bitwise_equal`, `q8_0_gemm_m1_equals_gemv`, `q8_0_gemv_matches_oracle`, `q8_0_range_partition_bitwise` (and the `f32_*` / `q4_k_*` families). They must stay green with no edits.
- **No `logits_abs_tol` / `gemv_rel_tol` loosening.** No new tolerance constant, no `observed_error_*` sweep — this change has no numeric surface.
- **No `HOST_ABI_VERSION` bump.** Codegen is unchanged (GEMV was already a symbol call); cached `model.so` artifacts stay valid.
- **Kernel perf numbers come only from `mise run bench-kernels`** inside the devenv shell on quiet hardware; CI runners are noise. Record every data point in the M4b.4 spec's `## Amendments` section, and never edit a recorded point.
- **`inferno bench` (vs llama.cpp) is a manual protocol**, never a CI gate: devenv shell, release build, quiet hardware; record each report in the M4b.4 spec's `## Amendments`.
- **`inferno-kernels` and `inferno-core` are the only crates allowed `unsafe`;** clippy runs deny-warnings (`mise run lint`) — keep it clean, no new `#[allow]` beyond what the file already carries.
- **`inferno-kernels` is x86-64-only** (a `compile_error!` guards other targets); all edits sit behind the existing `#[cfg(target_arch = "x86_64")]` + `#[target_feature(enable = "avx2,fma")]`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `crates/inferno-kernels/benches/gemv.rs` | GEMV throughput benches on real Qwen shapes vs ggml | Add a pure weight-streaming ceiling bench (diagnostic classification) |
| `crates/inferno-kernels/src/q8_0.rs` | Q8_0 rs8 GEMV/GEMM kernels | Prefetch (+ optional interleave) in `inferno_gemv_q8_0_rs8_avx2` full-strip path |
| `crates/inferno-kernels/src/q4_k.rs` | Q4_K rs8 GEMV/GEMM kernels | Mirror the winning transform in `inferno_gemv_q4_k_rs8_avx2` |
| `crates/inferno-kernels/src/f32k.rs` | F32 rs8 GEMV/GEMM kernels | Mirror the winning transform in `inferno_gemv_f32_rs8_avx2` |
| `mise.toml` | Task runner | Repoint the `bench` task description to the M4b.4 spec |
| `docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md` | The spec | Record all bench/profile data points in `## Amendments` |

Q8_0 is implemented and tuned first (Tasks 1–3) because it is the bench model's dtype and carries the decode profile; q4_k and f32 mirror the frozen transform afterward (Tasks 4–5).

---

## Task 1: Diagnostic — streaming ceiling + baseline classification

Establishes whether decode GEMV is memory-latency/MLP-bound (prefetch will help) or compute-bound (pivot to op-reduction). Adds a pure weight-streaming bench as the machine's achievable-read-bandwidth ceiling, captures the current GEMV baseline, and records the classification. **No kernel change in this task.**

**Files:**
- Modify: `crates/inferno-kernels/benches/gemv.rs` (add a streaming ceiling bench inside `bench_dtype`)
- Record: `docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md` (`## Amendments`)

**Interfaces:**
- Consumes: existing `bench_dtype(c, dtype, shapes)`, `gen_weights`, `sets_for`, `KernelSet::pack`.
- Produces: a `stream-read` bench arm in the `gemv/{dtype}` criterion group; a recorded classification decision (memory- vs compute-bound) plus the baseline GEMV GB/s per Qwen shape.

- [ ] **Step 1: Add a weight-streaming ceiling bench**

In `crates/inferno-kernels/benches/gemv.rs`, inside `bench_dtype`, after the `for (name, set) in sets_for(&dtype)` loop and before the `#[cfg(feature = "ggml-compare")]` line, add a streaming baseline over the packed weights for the AVX2-packed image. Insert:

```rust
        // Pure weight-streaming ceiling: the max read bandwidth this machine
        // sustains over the packed weight image, with no dot-product compute.
        // GEMV GB/s far below this ⇒ memory-latency/MLP-bound (prefetch helps);
        // GEMV GB/s near this ⇒ compute-bound (pivot to op-reduction). Uses the
        // AVX2 pack so the byte count matches the inferno-avx2 GEMV arm.
        if let Some(set) = kernels_for(&dtype, Isa::X86_64v3) {
            let w = set.pack(&file, rows, k).unwrap();
            group.throughput(Throughput::Bytes(w.len() as u64));
            group.bench_function(BenchmarkId::new("stream-read", format!("{rows}x{k}")), |b| {
                b.iter(|| {
                    let mut acc = 0u64;
                    // 8-wide u64 reduction: enough ILP to expose read bandwidth,
                    // no cross-lane dependency that would serialize on latency.
                    for chunk in w.as_slice().chunks_exact(64) {
                        for w8 in chunk.chunks_exact(8) {
                            acc = acc.wrapping_add(u64::from_le_bytes(w8.try_into().unwrap()));
                        }
                    }
                    std::hint::black_box(acc)
                })
            });
        }
```

`AlignedBuf::as_slice()` already exists (`crates/inferno-kernels/src/buf.rs:35`), so no buf.rs change is needed.

- [ ] **Step 2: Verify the bench compiles and lists the new arm**

Run: `cargo bench -p inferno-kernels --no-run 2>&1 | tail -5`
Expected: builds clean (no run). Then, on quiet hardware inside the devenv shell:
Run: `mise run bench-kernels 2>&1 | grep -E "stream-read|inferno-avx2" | head`
Expected: both `stream-read/<shape>` and `inferno-avx2/<shape>` arms report throughput for each Q8_0 shape.

- [ ] **Step 3: Capture baseline + classify (devenv shell, quiet hardware)**

Run: `mise run bench-kernels` and read the `gemv/Q8_0` group. For each Qwen decode shape (`896x896`, `4864x896`, `896x4864`, `151936x896`), record `inferno-avx2` GB/s (baseline, expect ~16) and `stream-read` GB/s (ceiling). Classify:
- baseline ≲ 0.6 × ceiling → **memory-latency/MLP-bound** → proceed to Task 2 (prefetch).
- baseline ≳ 0.85 × ceiling → **compute-bound** → STOP the prefetch path; open the approach-B op-reduction follow-up per the spec's Risks section and record that decision.

- [ ] **Step 4: Commit**

```bash
git add crates/inferno-kernels/benches/gemv.rs
git commit -m "bench(kernels): weight-streaming ceiling arm for decode GEMV diagnostic"
```

- [ ] **Step 5: Record the diagnostic in the spec**

Append a dated subsection to `## Amendments` in the M4b.4 spec with the per-shape baseline vs ceiling table and the one-line classification verdict (e.g. "memory-latency-bound: avg baseline 16.0 GB/s vs 41 GB/s ceiling → prefetch path confirmed"). Commit:

```bash
git add docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md
git commit -m "docs(specs): M4b.4 Task 1 diagnostic — decode GEMV is memory-bound"
```

---

## Task 2: Software prefetch in the Q8_0 AVX2 GEMV

Adds `_mm_prefetch` of upcoming weight groups to the full-strip fast path — the sole decode hot path — and tunes the prefetch distance. Bit-neutral: output is byte-identical to before.

**Files:**
- Modify: `crates/inferno-kernels/src/q8_0.rs` (`inferno_gemv_q8_0_rs8_avx2`, full-strip fast path ~`q8_0.rs:144-174`; add a module `const PF_DIST`)
- Test: `crates/inferno-kernels/tests/rig.rs` (existing proptests — run, do not edit)

**Interfaces:**
- Consumes: existing `inferno_gemv_q8_0_rs8_avx2` signature and the rs8 packed layout (`GROUP_BYTES = 288`, contiguous per strip).
- Produces: same symbol, same bits, higher throughput. `PF_DIST` (weight groups prefetched ahead) becomes the tuned constant reused by Tasks 4–5.

- [ ] **Step 1: Run the bit-identity proptests to confirm the green baseline**

Run: `cargo nextest run -p inferno-kernels -E 'test(q8_0)'`
Expected: PASS (all `q8_0_*` rig proptests green) — this is the safety net the change must preserve.

- [ ] **Step 2: Add the prefetch distance constant**

In `crates/inferno-kernels/src/q8_0.rs`, near the other consts at the top (after `const GROUP_BYTES: usize = 288;`), add:

```rust
/// Weight groups to software-prefetch ahead in the AVX2 GEMV (M4b.4). A
/// strip's `nb` groups are contiguous (`nb × GROUP_BYTES`), so prefetching
/// `PF_DIST` groups ahead reaches cleanly across the block loop and into the
/// next strip. Tuned by the Task 2 sweep; a pure hint, so it never affects
/// output bits.
const PF_DIST: usize = 4;
```

- [ ] **Step 3: Insert the prefetch into the full-strip fast path**

In `inferno_gemv_q8_0_rs8_avx2`, the full-strip branch begins with `if lane0 == 0 && r + STRIP <= row_end {` then `for b in 0..nb {` whose first line is `let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };`. Immediately after that `let g = ...;` line, add:

```rust
                // Prefetch a future weight group into L1 to overlap DRAM latency
                // with this block's int8 dot. Prefetching past the buffer end is
                // a hardware no-op (never faults, never forms a Rust reference),
                // so no bounds guard is needed. Pure hint — output is unchanged.
                unsafe {
                    _mm_prefetch::<_MM_HINT_T0>(
                        w.add((strip * nb + b + PF_DIST) * GROUP_BYTES).cast(),
                    );
                }
```

`_mm_prefetch` and `_MM_HINT_T0` come from `std::arch::x86_64::*`, already imported at the top of the function (`use std::arch::x86_64::*;`). Do **not** add prefetch to the partial head/tail per-row path (not the hot path) or to the scalar kernel.

- [ ] **Step 4: Verify bit-identity still holds**

Run: `cargo nextest run -p inferno-kernels -E 'test(q8_0)'`
Expected: PASS — `q8_0_isa_variants_bitwise_equal`, `q8_0_gemm_m1_equals_gemv`, `q8_0_gemv_matches_oracle`, `q8_0_range_partition_bitwise` all green. If any fail, the prefetch touched control flow it should not have — revert and re-inspect.

- [ ] **Step 5: Lint clean**

Run: `mise run lint`
Expected: PASS (rustfmt clean, clippy deny-warnings clean).

- [ ] **Step 6: Sweep the prefetch distance (devenv shell, quiet hardware)**

For `PF_DIST ∈ {2, 4, 8}`: edit the constant, run `mise run bench-kernels`, and read the `gemv/Q8_0 inferno-avx2` GB/s for the Qwen shapes. Keep the value with the best average GB/s (ties → smallest). Set `PF_DIST` to the winner.

- [ ] **Step 7: Record the result and decide on Task 3**

Compare the winning `inferno-avx2` GB/s to Task 1's numbers and the Leg-1 bar (a meaningful lift off ~16 toward the ~40/ceiling; working target ≥25 GB/s or ≥1.5× baseline):
- **Bar met** → skip Task 3; go to Task 4.
- **Bar missed** → keep the prefetch and proceed to Task 3 (add interleaving).

Append the sweep table + decision to the spec `## Amendments`.

- [ ] **Step 8: Commit**

```bash
git add crates/inferno-kernels/src/q8_0.rs docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md
git commit -m "feat(kernels): prefetch weight groups in q8_0 avx2 GEMV (decode MLP)"
```

---

## Task 3 (CONDITIONAL — only if Task 2 missed the Leg-1 bar): two-strip interleave in the Q8_0 AVX2 GEMV

Processes two full strips concurrently so twice the independent weight-load streams are in flight, hiding more DRAM latency. Row iteration reorders across strips; each row's per-block `fmadd` chain is byte-for-byte unchanged, so output is identical. **Skip this entire task if Task 2 already hit the bar.**

**Files:**
- Modify: `crates/inferno-kernels/src/q8_0.rs` (`inferno_gemv_q8_0_rs8_avx2`, add a two-strip loop before the existing single-strip fast path)
- Test: `crates/inferno-kernels/tests/rig.rs` (existing proptests — run, do not edit)

**Interfaces:**
- Consumes: `PF_DIST`, `hsum8_i32`, the rs8 layout.
- Produces: same symbol/bits; the two-strip loop handles rows in pairs of strips, the existing single-strip fast path handles the odd remaining strip, the per-row path handles the tail.

- [ ] **Step 1: Confirm green baseline**

Run: `cargo nextest run -p inferno-kernels -E 'test(q8_0)'`
Expected: PASS.

- [ ] **Step 2: Add the two-strip interleave loop**

In `inferno_gemv_q8_0_rs8_avx2`, immediately after `let mut r = row_start;` and before the `while r < row_end {` loop, insert a two-strip loop. It reuses the exact per-strip block math (sign-trick → `hsum8_i32` → f32 `fmadd`) for two independent accumulators:

```rust
    // Two full strips at a time: 2× the outstanding weight-load streams to
    // hide DRAM latency. Each strip keeps its own acc and runs the identical
    // per-block math as the single-strip path, so bits are unchanged.
    while r + 2 * STRIP <= row_end {
        let s0 = r / STRIP;
        let s1 = s0 + 1;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        for b in 0..nb {
            let g0 = unsafe { w.add((s0 * nb + b) * GROUP_BYTES) };
            let g1 = unsafe { w.add((s1 * nb + b) * GROUP_BYTES) };
            unsafe {
                _mm_prefetch::<_MM_HINT_T0>(w.add((s0 * nb + b + PF_DIST) * GROUP_BYTES).cast());
                _mm_prefetch::<_MM_HINT_T0>(w.add((s1 * nb + b + PF_DIST) * GROUP_BYTES).cast());
            }
            let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
            let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
            let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
            let dxv = _mm256_set1_ps(dx);
            for (g, acc) in [(g0, &mut acc0), (g1, &mut acc1)] {
                let qs = unsafe { g.add(32) };
                let mut p = [_mm256_setzero_si256(); STRIP];
                for (lane, pl) in p.iter_mut().enumerate() {
                    let wv = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                    let aw = _mm256_sign_epi8(wv, wv);
                    let sx = _mm256_sign_epi8(xv, wv);
                    *pl = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                }
                let isum = _mm256_cvtepi32_ps(hsum8_i32(p));
                let dw = unsafe { _mm256_load_ps(g.cast()) };
                let dwdx = _mm256_mul_ps(dw, dxv);
                *acc = _mm256_fmadd_ps(dwdx, isum, *acc);
            }
        }
        unsafe { _mm256_storeu_ps(y.add(r), acc0) };
        unsafe { _mm256_storeu_ps(y.add(r + STRIP), acc1) };
        r += 2 * STRIP;
    }
```

The per-strip body is a copy of the existing single-strip fast path's inner block (kept for the odd remaining strip), so the two paths stay obviously equivalent. `WBLOCK` and `Q8A_BLOCK_BYTES` are already in scope.

- [ ] **Step 3: Verify bit-identity**

Run: `cargo nextest run -p inferno-kernels -E 'test(q8_0)'`
Expected: PASS. `q8_0_range_partition_bitwise` specifically exercises non-strip-aligned row ranges and mixed strip counts — it must stay green, proving the two-strip/one-strip/tail hand-off is exact.

- [ ] **Step 4: Lint clean**

Run: `mise run lint`
Expected: PASS.

- [ ] **Step 5: Bench + record (devenv shell, quiet hardware)**

Run: `mise run bench-kernels` → read `gemv/Q8_0 inferno-avx2`. Record the GB/s vs Task 2. Keep the interleave only if it improves GB/s; if it regresses (register pressure), revert this task and stay with prefetch-only. Append the result to the spec `## Amendments`.

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-kernels/src/q8_0.rs docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md
git commit -m "feat(kernels): two-strip interleave in q8_0 avx2 GEMV (decode MLP)"
```

---

## Task 4: Mirror the winning transform into the Q4_K AVX2 GEMV

Applies the same prefetch (and, if adopted, the interleave shape) to `inferno_gemv_q4_k_rs8_avx2` so the three rs8 kernels stay structurally parallel.

**Files:**
- Modify: `crates/inferno-kernels/src/q4_k.rs` (`inferno_gemv_q4_k_rs8_avx2`, full-strip fast path ~`q4_k.rs:145`; add a module `const PF_DIST`)
- Test: `crates/inferno-kernels/tests/rig.rs` (existing `q4_k_*` proptests — run, do not edit)

**Interfaces:**
- Consumes: the q4_k rs8 layout (`GROUP_BYTES` for q4_k, `nsb = k / WBLOCK` super-blocks, contiguous per strip).
- Produces: same `inferno_gemv_q4_k_rs8_avx2` symbol/bits, higher throughput.

- [ ] **Step 1: Confirm green baseline**

Run: `cargo nextest run -p inferno-kernels -E 'test(q4_k)'`
Expected: PASS.

- [ ] **Step 2: Add the prefetch constant**

In `crates/inferno-kernels/src/q4_k.rs`, near the top consts, add the same doc'd constant (q4_k's groups are also contiguous per strip; the index uses `nsb`):

```rust
/// Weight groups to software-prefetch ahead in the AVX2 GEMV (M4b.4). See the
/// q8_0 kernel for the rationale; q4_k's `nsb` super-block groups are likewise
/// contiguous per strip. Pure hint — never affects output bits.
const PF_DIST: usize = 4;
```

Use the value Task 2 settled on (adjust here only if Step 5's q4_k sweep prefers a different one).

- [ ] **Step 3: Insert the prefetch into the full-strip fast path**

In `inferno_gemv_q4_k_rs8_avx2`, the full-strip branch's block loop is `for sb in 0..nsb {` whose first line is `let g = unsafe { w.add((strip * nsb + sb) * GROUP_BYTES) };`. Immediately after it, add:

```rust
                // Prefetch a future super-block group; hardware no-op past the
                // buffer end. Pure hint — output unchanged. (See q8_0 kernel.)
                unsafe {
                    _mm_prefetch::<_MM_HINT_T0>(
                        w.add((strip * nsb + sb + PF_DIST) * GROUP_BYTES).cast(),
                    );
                }
```

If Task 3's interleave was adopted for q8_0 and the q4_k bench (Step 5) shows it helps here too, mirror the two-strip loop analogously (two accumulators over the q4_k per-strip body). Otherwise prefetch-only.

- [ ] **Step 4: Verify bit-identity**

Run: `cargo nextest run -p inferno-kernels -E 'test(q4_k)'`
Expected: PASS (`q4_k_isa_variants_bitwise_equal`, `q4_k_gemm_m1_equals_gemv`, `q4_k_gemv_matches_oracle`, `q4_k_range_partition_bitwise`).

- [ ] **Step 5: Lint, bench, record**

Run: `mise run lint` → PASS. Then (devenv shell) `mise run bench-kernels` → read `gemv/Q4_K inferno-avx2`; record GB/s before/after in the spec `## Amendments`.

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-kernels/src/q4_k.rs docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md
git commit -m "feat(kernels): prefetch weight groups in q4_k avx2 GEMV (decode MLP)"
```

---

## Task 5: Mirror the winning transform into the F32 AVX2 GEMV

Applies prefetch to `inferno_gemv_f32_rs8_avx2`. The f32 layout streams one aligned 32-byte column vector per `k`, so the prefetch reaches ahead by columns rather than groups.

**Files:**
- Modify: `crates/inferno-kernels/src/f32k.rs` (`inferno_gemv_f32_rs8_avx2`, strip loop `f32k.rs:104-115`; add a module `const PF_DIST_F32`)
- Test: `crates/inferno-kernels/tests/rig.rs` (existing `f32_*` proptests — run, do not edit)

**Interfaces:**
- Consumes: the f32 rs8 layout — per strip, `k` columns of 8 contiguous f32 at `base.add(c * STRIP)`, one aligned 32-byte vector each.
- Produces: same `inferno_gemv_f32_rs8_avx2` symbol/bits, higher throughput.

- [ ] **Step 1: Confirm green baseline**

Run: `cargo nextest run -p inferno-kernels -E 'test(f32)'`
Expected: PASS.

- [ ] **Step 2: Add the prefetch constant**

In `crates/inferno-kernels/src/f32k.rs`, near the top, add (distance is in **columns** here — each column is one 32-byte vector, so a larger distance covers the same bytes as a q8_0 group):

```rust
/// Columns to software-prefetch ahead in the AVX2 GEMV (M4b.4). The f32 rs8
/// layout stores one aligned 32-byte vector per column; `PF_DIST_F32` columns
/// ahead ≈ one cache line beyond the current fetch. Pure hint — never affects
/// output bits.
const PF_DIST_F32: usize = 16;
```

- [ ] **Step 3: Insert the prefetch into the strip loop**

In `inferno_gemv_f32_rs8_avx2`, the full-strip loop is `while r + STRIP <= row_end {` with inner `for c in 0..k {` whose first line is `let wv = unsafe { _mm256_load_ps(base.add(c * STRIP)) };`. Immediately before that load, add:

```rust
            // Prefetch a future column vector; hardware no-op past the buffer
            // end. Pure hint — output unchanged. (See q8_0 kernel.)
            unsafe {
                _mm_prefetch::<_MM_HINT_T0>(base.add((c + PF_DIST_F32) * STRIP).cast());
            }
```

`_mm_prefetch` / `_MM_HINT_T0` are available via the function's `use std::arch::x86_64::*;`.

- [ ] **Step 4: Verify bit-identity**

Run: `cargo nextest run -p inferno-kernels -E 'test(f32)'`
Expected: PASS (`f32_isa_variants_bitwise_equal`, `f32_gemm_m1_equals_gemv`, `f32_gemv_matches_oracle`).

- [ ] **Step 5: Lint, bench, record**

Run: `mise run lint` → PASS. Then (devenv shell) `mise run bench-kernels` → read `gemv/F32 inferno-avx2`; record GB/s before/after in the spec `## Amendments`.

- [ ] **Step 6: Full kernel suite green**

Run: `cargo nextest run -p inferno-kernels`
Expected: PASS (all rig proptests across all three dtypes).

- [ ] **Step 7: Commit**

```bash
git add crates/inferno-kernels/src/f32k.rs docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md
git commit -m "feat(kernels): prefetch column vectors in f32 avx2 GEMV (decode MLP)"
```

---

## Task 6: End-to-end exit criterion + doc hygiene

Confirms the change lands at the model level: the compiled-vs-interpreter differential stays green (no numeric drift), the decode `--profile` shows the GEMV ops' GB/s risen, and a recorded `inferno bench` t=1 point shows tg improved with pp not regressed. Also repoints the stale `bench` task description.

**Files:**
- Modify: `mise.toml` (the `[tasks.bench]` description string)
- Verify (no edit): `crates/inferno-codegen/tests/differential.rs`, `crates/inferno-core/tests/artifact.rs`
- Record: `docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md` (`## Amendments`)

**Interfaces:**
- Consumes: the tuned kernels from Tasks 2–5; the standard M4a bench protocol.
- Produces: a recorded t=1 bench + decode profile data point; a corrected task description.

- [ ] **Step 1: Compiled-vs-interpreter differential stays green (no loosening)**

Run: `cargo nextest run -p inferno-codegen -E 'test(differential)' && cargo nextest run -p inferno-core -E 'test(artifact)'`
Expected: PASS with `logits_abs_tol` / `gemv_rel_tol` untouched — the kernels emit identical bits, so nothing drifts. (These are the compiled-path correctness gates named in `AGENTS.md`.)

- [ ] **Step 2: Repoint the stale bench task description**

In `mise.toml`, the `[tasks.bench]` `description` points at the M4b.1 spec for recording data points. Update that path to the current milestone spec:

Change `docs/superpowers/specs/2026-07-06-m4b1-threading-design.md` → `docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md` inside the `[tasks.bench]` description string only. Verify:

Run: `mise tasks | grep -A0 bench` (or `grep -n "m4b4" mise.toml`)
Expected: the `bench` task description now references the M4b.4 spec.

- [ ] **Step 3: Fresh decode profile (devenv shell, quiet hardware, freshly built release host)**

Rebuild and capture a t=1 decode profile on the pinned Q8_0 model (a fresh build so `dlopen` resolves the new kernel; no `HOST_ABI_VERSION` bump needed):

Run:
```bash
cargo build --release -p inferno
cargo run --release -p inferno -- run \
  --model <path>/qwen2.5-0.5b-instruct-q8_0.gguf \
  --prompt "<the standard profile prompt>" --max-tokens 64 --threads 1 --profile
```
Expected: a `profile [decode]` table. Confirm the `matmul:*` GEMV ops' `GB/s` column has risen from ~16 toward the Task-2/3 kernel figure, and attention share is unchanged (~26–27%).

- [ ] **Step 4: Recorded t=1 bench vs llama.cpp**

Run (devenv shell, quiet hardware):
```bash
mise run bench -- <path>/qwen2.5-0.5b-instruct-q8_0.gguf --threads 1 --reps 5
```
Expected: a report with pp/tg tok/s for inferno and llama.cpp. Confirm **tg improved and its low end at/above parity (≥1.0×)** and **pp not regressed** vs M4b.3's recorded 1.26–1.34× pp / 0.88–1.09× tg.

- [ ] **Step 5: Record both data points in the spec**

Append a dated `## Amendments` subsection with: the full decode `--profile` table (GEMV GB/s before/after), the complete `inferno bench` JSON/table (never edited afterward), and a Verdict paragraph stating both exit-criterion legs (Leg 1 kernel GB/s from Tasks 2–5; Leg 2 end-to-end tg/pp here) against their bars.

- [ ] **Step 6: Commit**

```bash
git add mise.toml docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md
git commit -m "docs(bench): record M4b.4 decode GEMV t=1 data point + repoint bench task"
```

---

## Self-Review

**Spec coverage:**
- Diagnostic-first (memory- vs compute-bound, prefetch/interleave sweep) → Task 1 + Task 2 Step 6 + Task 3 Step 5.
- Prefetch + optional multi-strip interleave in AVX2 GEMV → Tasks 2 (prefetch), 3 (interleave, conditional).
- Q8_0 primary, q4_k + f32 mirrored → Tasks 2–3 (q8_0), 4 (q4_k), 5 (f32).
- Scalar unchanged / bit-identity invariants → enforced every task via the existing rig proptests (Task 2 Step 4, Task 3 Step 3, Task 4 Step 4, Task 5 Steps 4 & 6).
- No tolerance loosening, no ABI bump, differential green → Task 6 Step 1.
- Two-legged exit criterion (kernel GB/s + t=1 tg/pp, wide margin) → Leg 1 across Tasks 2–5 bench steps; Leg 2 Task 6 Steps 3–5.
- Compute-bound fallback recorded, not silent → Task 1 Step 3.
- Doc hygiene (repoint `bench` task) → Task 6 Step 2.
- All data points in the spec Amendments, never edited → Tasks 1, 2, 3, 4, 5, 6 recording steps.

**Placeholder scan:** No TBD/TODO. `<path>` and `<the standard profile prompt>` in Task 6 are runtime inputs of the manual bench protocol (the model file and prompt are supplied by the operator per the M4a protocol), not unfilled plan content. `PF_DIST` starts at a concrete `4` and is tuned by an explicit sweep.

**Type/name consistency:** `PF_DIST` (q8_0, q4_k) and `PF_DIST_F32` (f32) are declared before use in their own modules; `_mm_prefetch::<_MM_HINT_T0>`, `hsum8_i32`, `WBLOCK`, `Q8A_BLOCK_BYTES`, `GROUP_BYTES`, `STRIP`, `nsb` all match the existing source. Bench arm names (`stream-read`, `inferno-avx2`) match `benches/gemv.rs`. Test filter expressions (`test(q8_0)`, `test(q4_k)`, `test(f32)`, `test(differential)`, `test(artifact)`) match the rig proptest names.
