# M4b.6 Reduce Restructure (Candidate 1: Unpack/Add Tree) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 6× `vphaddd` transpose-reduce in `hsum8_i32` with a `vpunpck`/`vpaddd` tree (spec §Task 2+ candidate 1, bit-neutral), gated on a measured same-box kernel win.

**Architecture:** One bench-only candidate arm (a copy of the AVX2 full-strip GEMV with only the reduce tree swapped, bitwise-checked against the library kernel) is measured against the shipped baseline first; only if the interleaved A/B shows a noise-robust win does the swap land in the shared `hsum8_i32` helper, where GEMV, GEMM, and Q4_K all inherit it. No contract change: integer wrapping adds are associative, so any reduction structure is bit-identical — tolerances, ABI version, scalar reference, and interpreter are all untouched.

**Tech Stack:** Rust + AVX2 intrinsics (`core::arch::x86_64`), criterion benches, proptest, mise tasks (`test`, `lint`, `bench-kernels`, `bench`), devenv shell.

**Spec:** `docs/superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md` §Task 2+, entered via the recorded Task 1 PROCEED gate (projected 14.9% decode-wall, reduce share 10.3pp / combine 4.7pp). This plan is the "restructure plan" that gate amendment calls for.

## Global Constraints

- **Bit-neutrality is the whole deal:** candidate 1 changes only the association of exact wrapping i32 adds. Every lock test must stay green untouched: `q8_0_isa_variants_bitwise_equal`, `q8_0_gemm_m1_equals_gemv`, `q8_0_gemv_matches_oracle`, `q8_0_range_partition_bitwise`, the pool's `*_thread_count_is_bit_invisible` / `q8_0_decode_cap_is_bit_invisible`, and the Q4_K equivalents (Q4_K shares `hsum8_i32`).
- **No tolerance edits:** `git diff -- crates/inferno-graph/src/tolerance.rs` must be empty at every commit in this plan.
- **No `HOST_ABI_VERSION` bump:** kernel outputs are bitwise unchanged (spec §Scope Decisions, bit-neutral branch).
- **Interpreter untouched.**
- **Measurement discipline (standing M4b):** same-box interleaved A/B, per-rep **ratios only**; absolute times/GiB/s recorded but untrusted on the shared devpod (AMD Ryzen 9 3900, Zen 2, 12C/24T); formal perf verdict deferred to quiet hardware; never edit a recorded data point.
- **Benches run inside the devenv shell** (`devenv shell -- ...`).
- **Before any task is called done:** whole-workspace `mise run test` green. **Before push:** `mise run lint` (CI runs clippy with `-D warnings`; `mise run test` does not).
- Scratch outputs go under `/tmp/claude-1000/-workspace/6ac72279-de24-4b07-9a69-90b815a08c0a/scratchpad/` (session scratchpad), never into the repo.

## File Structure

| File | Role in this plan |
|---|---|
| `crates/inferno-kernels/benches/gemv.rs` | Task 1 adds a `reduce_unpack` candidate-arm module + registration; Task 4 (or 5-B) removes it |
| `crates/inferno-kernels/src/q8_0.rs` | Task 3 adds a `#[cfg(test)]` oracle proptest for `hsum8_i32`, then swaps its body |
| `docs/superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md` | Task 5 appends the restructure amendment (ship or no-ship) |
| `docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md` | Task 4 records the directional end-to-end tg data point per the M4a protocol |

No other file changes. `q4_k.rs`, the GEMM twins, and the scalar reference are untouched — Q4_K and GEMM inherit the swap through the shared `pub(crate) fn hsum8_i32`.

---

## Decision Record — µop analysis (spec-mandated; resolves the Zen 2 caveat)

The spec's recorded caveat: "on Zen 2 the µop count is a near-wash (~14 shuffle + 7 add either way); this candidate only wins if the measured bottleneck is `vphaddd`'s port placement, and it helps Intel more than AMD." The plan's µop table must decide before code is written. Data from uops.info (fetched 2026-07-09), Zen 2 unless noted:

| Instruction (ymm) | µops | lat | rthroughput | ports |
|---|---|---|---|---|
| `VPHADDD` | **3** | 2 | **2.0** | (Zen 3 lists `1*FP0123 + 1*FP1 + 1*FP12`; Zen 2 measures identically) |
| `VPUNPCK{L,H}DQ` | 1 | 1 | 0.5 | `FP12` |
| `VPUNPCK{L,H}QDQ` | 1 | 1 | 1.0 | `FP12` |
| `VPERM2I128` | 1 | 3 | 1.0 | `FP2` |
| `VPADDD` | 1 | 1 | 0.33 | `FP013` |
| `VPMADDUBSW` / `VPMADDWD` | 1 | 4 | 1.0 | **`FP0` only** |
| `VPSIGNB` | 1 | 1 | 0.5 | `FP03` |
| `VCVTDQ2PS` | 1 | 3 | 1.0 | `FP3` only |

**Per-block tree comparison** (both trees: 8×`__m256i` → 1):

| | instructions | µops | issue cost (model) | dep-chain latency |
|---|---|---|---|---|
| current hadd tree | 6 hadd + 2 perm + 1 add = 9 | 21 | 6 hadd × rt 2.0 ≈ **12 cyc**, plus 6 of the 18 hadd µops are `FP0123`-class → contend with the `FP0`-only dot (16 madd µops/block) | 2+2+3+1 = 8 cyc |
| candidate 1 unpack tree | 12 unpck + 2 perm + 7 add = 21 | 21 | 8 dq-unpck × 0.5 + 4 qdq-unpck × 1.0 + 2 perm ≈ **8–10 cyc** on `FP1/FP2`; the 7 adds land in `FP013` slack; **zero** forced-`FP0` µops | 1+1+1+1+3+1 = 8 cyc |

**Verdict: PROCEED with candidate 1.** The caveat was right that the µop *count* is a wash (21 vs 21) and wrong about the direction of the port effect: `VPHADDD`'s rt 2.0 vs the dword unpacks' rt 0.5 gives the unpack tree roughly half the issue cost on Zen 2's two shuffle-capable pipes, and it stops stealing slots from the `FP0`-bound integer dot. On Intel SKL-class cores it is the wash the caveat predicted (all shuffle µops on p5 either way: 14 vs 14) — the win is AMD-side, i.e. on exactly the box we measure on. Latency is unchanged, so the win, if real, shows up as throughput on the L2/L3-resident mid shapes where Task 1 measured the ~15% reduce share; expected capture is a mid-single-digit kernel-level win, which the profile weighting turns into roughly a 2–5% decode-wall projection.

**Candidates 2–3 are rejected, not deferred:**

- **Candidate 2 (block-pair combine merge):** attacks only the combine, whose measured share is 4.7pp; halving its chain depth caps the win at ~2.4pp — under the milestone's own 3% STOP bar — while costing a numeric-contract change (tolerance re-derivation, ABI bump, scalar redefinition). Dominated.
- **Candidate 3 (lane-deferred f32 accumulation):** per block it replaces the 21-µop reduce + ~4-µop combine with 8× `VCVTDQ2PS` (rt 1.0 on `FP3` *only*, colliding with the 16 `VPSIGNB` µops on `FP03`), 8 per-lane scale broadcasts (`FP12` shuffles), and 8 FMAs (`FP01`) ≈ 26 µops with strictly worse port placement — the spec's anticipated "naive lane-deferred form adds per-lane scale broadcasts" caveat, now confirmed with numbers. Also the largest contract change. Rejected on Zen 2.

If candidate 1's measured A/B (Task 2) fails its ship gate, the correct exit is the no-ship amendment (Task 5-B) — *not* an escalation to candidates 2–3, which this table already rules out on the available hardware.

**Why the new tree is bit-identical** (load-bearing for the whole plan): `hsum8_i32` computes, per output lane, a sum of eight i32 lane values. Rust/LLVM `_mm256_add_epi32` and `_mm256_hadd_epi32` are wrapping two's-complement adds, and wrapping addition is associative and commutative, so *any* reduction structure yields identical bits — including on inputs that overflow (they can't here anyway: `maddubs` products are bounded by the pack-time `-128 → -127` clamp, but the proof doesn't need that). The function's doc comment already records "the reduction structure is unconstrained by the numeric contract"; this plan exercises exactly that freedom. The intermediate after round 2 of the new tree is byte-for-byte the same layout the hadd tree produces (`[v0lo v1lo v2lo v3lo | v0hi v1hi v2hi v3hi]`), so round 3 (cross-128 recombine) is literally unchanged code.

---

### Task 1: `reduce-unpack` candidate arm in the GEMV bench

Adds a bench-only copy of the AVX2 full-strip GEMV whose only delta is the unpack/add tree, plus a registration block that first asserts the arm's output is **bitwise equal** to the library kernel, then times it. Unlike the `reduce_ceiling` cost models, this arm computes correct numbers — the assert is the arm's test cycle. No library change in this task.

**Files:**
- Modify: `crates/inferno-kernels/benches/gemv.rs` (new module after `mod reduce_ceiling` ends at ~line 491; registration inside the existing Q8_0 x86_64 block, after the `combine-stub` registration at ~line 183)

**Interfaces:**
- Consumes: the existing bench scaffolding — `set` (`kernels_for(&dtype, Isa::X86_64v3)` KernelSet), `w`/`xq`/`y` buffers, the `rows % STRIP == 0` assert already in that block.
- Produces: criterion id `gemv/Q8_0/reduce-unpack/{rows}x{k}` for every `SHAPES_Q8_0` shape; `reduce_unpack::gemv(y, x, w, k, rows)` (unsafe, whole strips, AVX2+FMA) and `reduce_unpack::hsum8_i32_unpack([__m256i; 8]) -> __m256i` — Task 3 copies the latter's body into the library.

- [ ] **Step 1: Add the `reduce_unpack` module**

In `crates/inferno-kernels/benches/gemv.rs`, immediately after the closing brace of `mod reduce_ceiling` (~line 491), insert:

```rust
/// M4b.6 restructure candidate 1 (spec §Task 2+): the shipped AVX2 full-strip
/// GEMV with exactly one delta — `hsum8_i32` swapped for an unpack/add
/// transpose-reduce. Unlike the `reduce_ceiling` cost models this arm computes
/// CORRECT numbers; the bench setup asserts bitwise equality against the
/// library kernel before timing it. Whole strips only (`rows % STRIP == 0`).
/// Deleted once the A/B is decided — the winning tree ships in `q8_0.rs`, and
/// a losing tree is recorded in the spec Amendments and removed.
#[cfg(target_arch = "x86_64")]
mod reduce_unpack {
    use std::arch::x86_64::__m256i;

    use inferno_kernels::STRIP;

    const WBLOCK: usize = 32; // weight elements per block (q8_0.rs:10)
    const GROUP_BYTES: usize = 288; // 8 f32 d + 8×32 qs (q8_0.rs:12)
    const Q8A_BLOCK_BYTES: usize = 36; // f32 d + 32 i8 (act.rs:15)
    const PF_DIST: usize = 4; // mirrors q8_0.rs:19 so the A/B isolates the tree

    /// Candidate-1 transpose-reduce: same contract as `q8_0.rs::hsum8_i32`
    /// (lane `i` of the result = horizontal sum of `v[i]`), built from
    /// `vpunpck`/`vpaddd` instead of `vphaddd`. Wrapping i32 adds are
    /// associative, so this is bit-identical to the hadd tree by construction.
    #[target_feature(enable = "avx2")]
    fn hsum8_i32_unpack(v: [__m256i; 8]) -> __m256i {
        use std::arch::x86_64::*;
        // Round 1 — 32-bit interleave + add. Per 128-bit half, lo/hi unpack of
        // (a, b) give [a0 b0 a1 b1] and [a2 b2 a3 b3]; their sum holds each
        // input's even/odd 2-element partials interleaved: [a02 b02 a13 b13].
        let s01 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(v[0], v[1]),
            _mm256_unpackhi_epi32(v[0], v[1]),
        );
        let s23 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(v[2], v[3]),
            _mm256_unpackhi_epi32(v[2], v[3]),
        );
        let s45 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(v[4], v[5]),
            _mm256_unpackhi_epi32(v[4], v[5]),
        );
        let s67 = _mm256_add_epi32(
            _mm256_unpacklo_epi32(v[6], v[7]),
            _mm256_unpackhi_epi32(v[6], v[7]),
        );
        // Round 2 — 64-bit interleave + add: per half [x02 y02] + [x13 y13]
        // leaves [v0half v1half v2half v3half], i.e. the same
        // [v0lo v1lo v2lo v3lo | v0hi v1hi v2hi v3hi] layout the hadd tree's
        // second round produces.
        let s0123 = _mm256_add_epi32(
            _mm256_unpacklo_epi64(s01, s23),
            _mm256_unpackhi_epi64(s01, s23),
        );
        let s4567 = _mm256_add_epi32(
            _mm256_unpacklo_epi64(s45, s67),
            _mm256_unpackhi_epi64(s45, s67),
        );
        // Round 3 — unchanged cross-128 recombine: lane i = full sum of v[i].
        let lo = _mm256_permute2x128_si256::<0x20>(s0123, s4567);
        let hi = _mm256_permute2x128_si256::<0x31>(s0123, s4567);
        _mm256_add_epi32(lo, hi)
    }

    /// The shipped full-strip fast path (`q8_0.rs::inferno_gemv_q8_0_rs8_avx2`,
    /// whole-strip branch) verbatim — prefetch, sign-trick dot, and the f32
    /// combine included — with only the reduce tree swapped.
    ///
    /// # Safety
    /// As `inferno_gemv_q8_0_rs8_avx2` (`y` writable for `rows` f32; `x` a q8a
    /// buffer for this `k`; `w` an rs8 pack for exactly this `k`/`rows`,
    /// 32-byte aligned; `k` a positive multiple of 32), plus: whole strips
    /// only (`rows % STRIP == 0`) and AVX2+FMA present.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemv(y: *mut f32, x: *const u8, w: *const u8, k: usize, rows: usize) {
        use std::arch::x86_64::*;
        let nb = k / WBLOCK;
        let ones = _mm256_set1_epi16(1);
        for strip in 0..rows / STRIP {
            let mut acc = _mm256_setzero_ps();
            for b in 0..nb {
                let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                // Same pure-hint prefetch as the shipped kernel (q8_0.rs:160-163).
                let pf_addr = w
                    .wrapping_add((strip * nb + b + PF_DIST) * GROUP_BYTES)
                    .cast();
                _mm_prefetch::<_MM_HINT_T0>(pf_addr);
                let xb = unsafe { x.add(b * Q8A_BLOCK_BYTES) };
                let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                let qs = unsafe { g.add(32) };
                let mut p = [_mm256_setzero_si256(); STRIP];
                for (lane, pl) in p.iter_mut().enumerate() {
                    let wv = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                    let aw = _mm256_sign_epi8(wv, wv);
                    let sx = _mm256_sign_epi8(xv, wv);
                    *pl = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                }
                // The one delta vs the shipped kernel: unpack tree, not hadd.
                let isum = _mm256_cvtepi32_ps(hsum8_i32_unpack(p));
                let dw = unsafe { _mm256_load_ps(g.cast()) };
                let dwdx = _mm256_mul_ps(dw, _mm256_set1_ps(dx));
                acc = _mm256_fmadd_ps(dwdx, isum, acc);
            }
            unsafe { _mm256_storeu_ps(y.add(strip * STRIP), acc) };
        }
    }
}
```

- [ ] **Step 2: Register the arm with a bitwise pre-check**

Inside `bench_dtype`, in the existing `#[cfg(target_arch = "x86_64")] if matches!(dtype, DType::Q8_0) ...` block, immediately after the `combine-stub` `group.bench_function(...)` call (after ~line 183, before the block's closing brace), insert:

```rust
            // M4b.6 restructure candidate 1: the same kernel with the
            // unpack/add reduce tree. Correct numbers — prove bitwise equality
            // against the library kernel once per process, then time it.
            {
                let mut y_ref = vec![f32::NAN; rows];
                set.gemv(&mut y_ref, &xq, &w, rows, k, 0, rows).unwrap();
                // SAFETY: buffers built above for exactly this rows/k;
                // rows % STRIP == 0 asserted; AVX2 runtime-detected.
                unsafe {
                    reduce_unpack::gemv(
                        y.as_mut_ptr(),
                        xq.as_slice().as_ptr(),
                        w.as_slice().as_ptr(),
                        k,
                        rows,
                    );
                }
                for (i, (a, b)) in y_ref.iter().zip(&y).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "reduce-unpack arm diverges from library kernel at row {i} ({rows}x{k})"
                    );
                }
            }
            group.bench_function(
                BenchmarkId::new("reduce-unpack", format!("{rows}x{k}")),
                |b| {
                    b.iter(|| {
                        // SAFETY: as the bitwise pre-check above.
                        unsafe {
                            reduce_unpack::gemv(
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
```

Note: `y` is the `let mut y = vec![0f32; rows];` already declared in that block for the ceiling arms; only `y_ref` is new.

- [ ] **Step 3: Validate — every arm runs once, asserts included**

Run: `devenv shell -- cargo bench -p inferno-kernels --bench gemv -- 'gemv/Q8_0' --test`

Expected: exits 0; every `gemv/Q8_0/*` bench (including all six `reduce-unpack/{shape}` entries) prints a test-mode pass line; no assert fires. If the bitwise assert fires, the tree is mis-derived — fix the arm (compare intermediates against the Round-1/2 comments) before any measurement; do not weaken the assert.

- [ ] **Step 4: Workspace still green, then commit**

Run: `mise run test && mise run lint`
Expected: both exit 0 (the bench arm compiles under clippy's `-D warnings` in lint).

```bash
git add crates/inferno-kernels/benches/gemv.rs
git commit -m "bench(kernels): reduce-unpack candidate arm for M4b.6 restructure A/B"
```

---

### Task 2: Interleaved A/B measurement + ship gate

Measures baseline vs candidate on the same box, same binary, same run — criterion executes `inferno-avx2` and `reduce-unpack` back-to-back per shape, and whole-run repetitions give per-rep ratios. Decides SHIP / NO-SHIP. **No repo changes in this task** — outputs are scratch until the amendment (Task 5).

**Files:**
- Create: `/tmp/claude-1000/-workspace/6ac72279-de24-4b07-9a69-90b815a08c0a/scratchpad/m4b6r-table.md` (working table; durable record lands in the spec in Task 5)

**Interfaces:**
- Consumes: criterion ids `gemv/Q8_0/inferno-avx2/{rows}x{k}` and `gemv/Q8_0/reduce-unpack/{rows}x{k}` from Task 1.
- Produces: per-shape per-rep wins `w_r = 1 − t_unpack,r / t_base,r`, their medians, a SHIP/NO-SHIP verdict, and `projected_decode_win` — consumed by Tasks 3–5.

- [ ] **Step 1: Run 3 interleaved reps**

```bash
SCRATCH=/tmp/claude-1000/-workspace/6ac72279-de24-4b07-9a69-90b815a08c0a/scratchpad
for i in 1 2 3; do
  devenv shell -- cargo bench -p inferno-kernels --bench gemv -- \
    'gemv/Q8_0/(inferno-avx2|reduce-unpack)/' 2>&1 | tee "$SCRATCH/m4b6r-rep$i.out"
done
```

Expected: each rep prints `time: [low mid high]` lines for both arms on all six `SHAPES_Q8_0` shapes (896x896, 4864x896, 896x4864, 151936x896, 4096x4096, 14336x4096). The regex keeps the run to just the two arms so a rep is short and the A/B pairs sit close in time.

- [ ] **Step 2: Build the per-shape table**

For each shape and rep, take the **middle** criterion estimate for both arms and compute `w_r = 1 − t_unpack,r / t_base,r`. Write `$SCRATCH/m4b6r-table.md` with one row per shape: `shape | t_base med | t_unpack med | w_1 | w_2 | w_3 | median w | min–max w`. The signal is the median of per-rep ratios, never the ratio of medians (Task 1 amendment discipline).

- [ ] **Step 3: Apply the ship gate**

Using the decode-wall shares fixed in the Task 1 amendment (never re-derived here):

```text
projected_decode_win = 0.270·w(151936x896) + 0.211·w(896x4864)
                     + 0.407·w(4864x896)   + 0.087·w(896x896)
```

(0.407 = up 20.4% + gate 20.3%; 0.087 = q 3.8% + o 3.8% + k/v ≈1.1%, both mapped to 896×896 per the amendment's recorded approximation; non-matmul 2.6% contributes 0.)

**SHIP** iff both:
1. `w_r > 0` in **every** rep on at least 2 of the 3 mid shapes (896x896, 4864x896, 896x4864) — the noise-robustness bar the Task 1 amendment used (`A < base` held 6/6), and
2. no shape's median `w` is below −3% (a consistent regression anywhere kills a "free" win).

**NO-SHIP** otherwise. If the verdict hinges on a shape whose per-rep `w` range straddles 0, run 3 more reps first (`for i in 4 5 6`; precedent: the Task 1 amendment extended 3→6) and re-apply the gate over all reps. Record `projected_decode_win` in the scratch table either way — it goes in the amendment but does not override the two conditions (a real, all-reps kernel win on a bit-neutral one-function change ships even if the projection is small).

- [ ] **Step 4: Confirm clean tree**

Run: `git status --porcelain`
Expected: empty — nothing from this task belongs in the repo. On **SHIP** → Task 3. On **NO-SHIP** → Task 5-B (skip Tasks 3–4).

---

### Task 3 (SHIP-gated): Swap the library `hsum8_i32`

Locks the helper's semantics with a scalar-oracle proptest first, then swaps the body. GEMV, GEMM, and Q4_K (all AVX2 variants call this one `pub(crate)` helper) inherit in one edit; scalar references are untouched, so the existing bitwise lock suite is the proof of bit-neutrality.

**Files:**
- Modify: `crates/inferno-kernels/src/q8_0.rs` (`hsum8_i32` body ~lines 112–127; new `#[cfg(test)] mod tests` at end of file)

**Interfaces:**
- Consumes: `reduce_unpack::hsum8_i32_unpack`'s body from Task 1 (verbatim — it is now bitwise-proven and measured).
- Produces: `pub(crate) fn hsum8_i32(v: [__m256i; 8]) -> __m256i` — same name, signature, contract, and bits; only the instruction mix changes. Callers (`q8_0.rs` GEMV/GEMM fast paths, `q4_k.rs:216,217,429,430`) need no edits.

- [ ] **Step 1: Write the oracle proptest (against the CURRENT tree)**

Append to `crates/inferno-kernels/src/q8_0.rs`:

```rust
#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    /// Loads 8×8 i32s, runs the kernel reduce, stores the result — the
    /// `target_feature` wrapper keeps the intrinsics in an AVX2 context.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn hsum8_via_kernel(vals: &[[i32; 8]; 8]) -> [i32; 8] {
        use std::arch::x86_64::*;
        let v: [__m256i; 8] =
            std::array::from_fn(|i| unsafe { _mm256_loadu_si256(vals[i].as_ptr().cast()) });
        let mut out = [0i32; 8];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), super::hsum8_i32(v)) };
        out
    }

    #[cfg(target_arch = "x86_64")]
    proptest! {
        /// `hsum8_i32` contract: lane `i` = wrapping sum of `v[i]`'s 8 lanes.
        /// Wrapping adds are associative and commutative, so every reduction
        /// structure must match this exactly — full-range i32s included, which
        /// locks the M4b.6 hadd→unpack swap (and any future one) to the bit.
        #[test]
        fn hsum8_i32_matches_scalar_oracle(
            vals in proptest::array::uniform8(proptest::array::uniform8(any::<i32>()))
        ) {
            if !std::arch::is_x86_feature_detected!("avx2") { return Ok(()); }
            // SAFETY: AVX2 just runtime-detected.
            let got = unsafe { hsum8_via_kernel(&vals) };
            for (i, lanes) in vals.iter().enumerate() {
                let want = lanes.iter().fold(0i32, |a, &x| a.wrapping_add(x));
                prop_assert_eq!(got[i], want, "lane {}", i);
            }
        }
    }
}
```

- [ ] **Step 2: Run it — must pass against the hadd tree**

Run: `cargo nextest run -p inferno-kernels hsum8_i32_matches_scalar_oracle`
Expected: PASS (this is a characterization lock, not a red test — it pins the semantics both trees must satisfy).

- [ ] **Step 3: Commit the lock**

```bash
git add crates/inferno-kernels/src/q8_0.rs
git commit -m "test(kernels): scalar-oracle proptest locks hsum8_i32 to the bit"
```

- [ ] **Step 4: Swap the body**

Replace the body of `hsum8_i32` (q8_0.rs ~lines 112–127) and refresh its comments. The function becomes:

```rust
/// Transpose-reduce 8 lane-parallel i32 accumulators into one vector whose
/// lane `i` holds the horizontal sum of `v[i]`'s 8 lanes. Pure integer adds,
/// so — like [`hsum_i32`] — the reduction structure is unconstrained by the
/// numeric contract; it lets a strip emit all 8 rows' block dots at once.
///
/// Built from `vpunpck`/`vpaddd` rather than `vphaddd` (M4b.6): same 21 µops
/// on Zen 2, but dword unpacks issue at 0.5/cycle on two shuffle pipes where
/// `vphaddd` is a 3-µop, 2.0-throughput op whose third µop competes with the
/// `FP0`-bound `maddubs`/`maddwd` dot — see the M4b.6 spec's restructure
/// amendment for the measured A/B.
///
/// Callers must have AVX2 enabled (`target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) fn hsum8_i32(v: [std::arch::x86_64::__m256i; 8]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // Round 1 — 32-bit interleave + add. Per 128-bit half, lo/hi unpack of
    // (a, b) give [a0 b0 a1 b1] and [a2 b2 a3 b3]; their sum holds each
    // input's even/odd 2-element partials interleaved: [a02 b02 a13 b13].
    let s01 = _mm256_add_epi32(
        _mm256_unpacklo_epi32(v[0], v[1]),
        _mm256_unpackhi_epi32(v[0], v[1]),
    );
    let s23 = _mm256_add_epi32(
        _mm256_unpacklo_epi32(v[2], v[3]),
        _mm256_unpackhi_epi32(v[2], v[3]),
    );
    let s45 = _mm256_add_epi32(
        _mm256_unpacklo_epi32(v[4], v[5]),
        _mm256_unpackhi_epi32(v[4], v[5]),
    );
    let s67 = _mm256_add_epi32(
        _mm256_unpacklo_epi32(v[6], v[7]),
        _mm256_unpackhi_epi32(v[6], v[7]),
    );
    // Round 2 — 64-bit interleave + add leaves, per half,
    // [v0half v1half v2half v3half]: the same
    // [v0lo v1lo v2lo v3lo | v0hi v1hi v2hi v3hi] layout as before.
    let s0123 = _mm256_add_epi32(
        _mm256_unpacklo_epi64(s01, s23),
        _mm256_unpackhi_epi64(s01, s23),
    );
    let s4567 = _mm256_add_epi32(
        _mm256_unpacklo_epi64(s45, s67),
        _mm256_unpackhi_epi64(s45, s67),
    );
    // Round 3 — cross-128 recombine so lane i = full sum of v[i].
    let lo = _mm256_permute2x128_si256::<0x20>(s0123, s4567);
    let hi = _mm256_permute2x128_si256::<0x31>(s0123, s4567);
    _mm256_add_epi32(lo, hi)
}
```

- [ ] **Step 5: The full lock suite, workspace-wide**

Run: `cargo nextest run -p inferno-kernels`
Expected: PASS — in particular `hsum8_i32_matches_scalar_oracle`, `q8_0_isa_variants_bitwise_equal`, `q8_0_gemm_m1_equals_gemv`, `q8_0_gemv_matches_oracle`, `q8_0_range_partition_bitwise`, and every `q4_k_*` test (Q4_K's AVX2 path calls the swapped helper at `q4_k.rs:216,217,429,430`).

Then: `mise run test`
Expected: PASS (whole workspace: the codegen differential and core artifact suites see bitwise-identical kernel outputs).

Then: `git diff -- crates/inferno-graph/src/tolerance.rs`
Expected: empty output — the bit-neutral branch's proof obligation from the spec.

- [ ] **Step 6: Lint, then commit**

Run: `mise run lint`
Expected: exit 0.

```bash
git add crates/inferno-kernels/src/q8_0.rs
git commit -m "perf(kernels): unpack/add transpose-reduce in hsum8_i32 (M4b.6 candidate 1)"
```

---

### Task 4 (SHIP-gated): Retire the arm, post-swap sanity, e2e data point

The arm now duplicates the shipped kernel instruction-for-instruction, so it comes out; one short bench run confirms the library kernel picked up the arm's timing; one `mise run bench` gives the spec's Leg-2 directional tg data point.

**Files:**
- Modify: `crates/inferno-kernels/benches/gemv.rs` (delete `mod reduce_unpack` and its registration + bitwise-check block from Task 1)
- Modify: `docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md` (append the bench report to its Amendments, matching the format of the most recent entry there)

**Interfaces:**
- Consumes: Task 2's per-shape medians (the sanity bar); the pinned model via `scripts/fetch-qwen-gguf.sh` (prints the cached GGUF path on stdout).
- Produces: a recorded M4a-protocol data point that Task 5's amendment cross-references.

- [ ] **Step 1: Delete the arm**

Remove from `crates/inferno-kernels/benches/gemv.rs`: the whole `mod reduce_unpack { ... }` module and, in the Q8_0 x86_64 registration block, the bitwise pre-check `{ ... }` block plus the `reduce-unpack` `group.bench_function(...)` call (both inserted in Task 1). The `reduce_ceiling` cost-model arms stay — they model costs, not candidates.

- [ ] **Step 2: Post-swap sanity rep**

Run: `devenv shell -- cargo bench -p inferno-kernels --bench gemv -- 'gemv/Q8_0/inferno-avx2/'`
Expected: `inferno-avx2` mid-shape times land at (noise-band of) Task 2's `t_unpack` medians, not its `t_base` medians — the library kernel now IS the candidate. A reading at `t_base` means the swap didn't land (wrong function edited, stale build); stop and fix.

- [ ] **Step 3: Directional end-to-end tg data point (spec Leg 2)**

```bash
GGUF=$(devenv shell -- scripts/fetch-qwen-gguf.sh | tail -1)
devenv shell -- mise run bench -- "$GGUF"
```

Append the report to the M4a spec's Amendments following the format of the most recent recorded entry (same box caveat, ratio-only reading), noting it as the M4b.6 restructure data point. Directional only — the formal perf verdict stays deferred to quiet hardware (standing M4b discipline).

- [ ] **Step 4: Verify and commit**

Run: `mise run test && mise run lint`
Expected: both exit 0.

```bash
git add crates/inferno-kernels/benches/gemv.rs docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md
git commit -m "bench: drop reduce-unpack arm (shipped in hsum8_i32); record M4b.6 tg data point"
```

---

### Task 5: Spec amendment — close the milestone

The durable record. Two variants; exactly one lands.

**Files:**
- Modify: `docs/superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md` (append to `## Amendments`; never edit the existing Task 1 entry)
- Modify (5-B only): `crates/inferno-kernels/benches/gemv.rs` (delete the arm, as Task 4 Step 1)

**Interfaces:**
- Consumes: Task 2's table and verdict; the µop Decision Record from this plan; (5-A) Task 4's data-point pointer.
- Produces: the milestone's closing record; M4b.7 (quiet-hardware verification pass) inherits the follow-up list.

- [ ] **Step 1: Append the amendment**

**Variant 5-A (SHIPPED):**

```markdown
### 2026-07-XX — Task 2+ restructure: candidate 1 shipped

- **Commits:** [arm commit], [test-lock commit], [swap commit], [arm-removal commit].
- **Candidate selection:** the restructure plan's µop table
  (docs/superpowers/plans/2026-07-09-m4b6-reduce-unpack-restructure.md
  §Decision Record, uops.info Zen 2 data) resolved the recorded Zen 2
  shuffle-port caveat FOR candidate 1: µop count is the anticipated wash
  (21 vs 21) but `vphaddd` is 3 µops at rt 2.0 vs the dword unpacks' 1 µop
  at rt 0.5, and the hadd tree's third µop class contends with the FP0-bound
  int8 dot. The caveat's "helps Intel more than AMD" guess inverted: SKL-class
  is the wash (14 p5 µops either way); the win is Zen-side. Candidates 2–3
  rejected by the same table (combine share caps C2 under the 3% bar; C3 is
  µop- and port-negative on Zen 2), not deferred.
- **Command:** [3 or 6] reps of `cargo bench -p inferno-kernels --bench gemv
  -- 'gemv/Q8_0/(inferno-avx2|reduce-unpack)/'` (devenv shell, shared devpod,
  ratio-only). Outputs in scratch (`m4b6r-rep{1..N}.out`).

| shape | t_base | t_unpack | median w | w range | all-reps w>0? |
|---|---|---|---|---|---|
| [per Task 2 table — six rows] |

- **projected_decode_win = [X]%** (Task 1 amendment's decode shares).
- **Ship gate:** [met: which mid shapes held w>0 in all reps; no median
  regression below −3%].
- **Bit-neutrality:** `hsum8_i32_matches_scalar_oracle` (full-range i32,
  wrapping-add oracle) + the standing bitwise lock suite green;
  `git diff -- crates/inferno-graph/src/tolerance.rs` empty; no
  `HOST_ABI_VERSION` bump. Q4_K and the GEMM twins inherit via the shared
  helper.
- **e2e:** M4b.6 restructure tg data point recorded in the M4a spec's
  Amendments ([date/section]). Formal perf verdict deferred to quiet
  hardware (M4b.7), per §Measurement & Exit Criterion.
```

**Variant 5-B (NO-SHIP):** first delete the arm (exactly Task 4 Step 1) and commit `bench(kernels): drop reduce-unpack arm (no measured win; M4b.6 restructure closed)`; then append:

```markdown
### 2026-07-XX — Task 2+ restructure: candidate 1 measured, not shipped

- **Commits:** [arm commit] (candidate arm, bitwise-checked), [removal commit].
- **Candidate selection:** [same µop-table paragraph as 5-A].
- **Command / table / projection:** [as 5-A — the full per-shape record].
- **Ship gate:** NOT met — [which condition failed, with the per-rep data].
  The measured bottleneck is not `vphaddd` issue throughput; with candidates
  2–3 already µop-negative on Zen 2 (plan §Decision Record), the decode GEMV
  inner loop is exhausted as an op-reduction lever on this hardware. The tg
  win effort moves to the quiet-hardware verification pass (M4b.7), which
  should re-run this A/B once on an Intel box before declaring the lever dead
  cross-vendor (SKL model says wash, not loss).
- No library change shipped; tolerances/ABI untouched by construction.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md
git commit -m "docs(specs): record M4b.6 restructure verdict; close the milestone"
```

(For 5-B, also `git add crates/inferno-kernels/benches/gemv.rs` in the earlier removal commit.)

- [ ] **Step 3: Final gates before push**

Run: `mise run test && mise run lint`
Expected: both exit 0. Then push / open the PR per repo workflow (lefthook runs the blocking tier pre-push; lint is the CI-parity check).
