# M4b.13 — Prefill GEMM Register Tiles + Gated VNNI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the prefill half of the v1 win criterion (pp ≥ 1.0x vs llama.cpp best-of on both quiet-hw boxes) by register-tiling the Q8_0 batched-GEMM kernels (Lever 1, the M4b.2 escalation), then — only if the pre-registered mid-milestone gate authorizes it — adding an AVX-512 VNNI GEMM path (Lever 2).

**Architecture:** This is a **laddered, data-gated** milestone. Tasks 1–3 land Lever 1 on the dev box behind a local µbench gate (no metal spend until the tiled kernel beats the current one). Task 4 builds the attribution gate script. Task 5 runs the mid-milestone quiet-hw sessions and applies the pre-registered ladder rule — **no task after Task 5 may assume its outcome.** Tasks 6–7 (Lever 2: Intel SDE test harness, then the `KernelIsa::Avx512Vnni` kernel + dispatch) run only if the gate authorizes. Task 8 records the closing data and walks the exit criteria (it runs on every path; if Lever 2 was skipped, Task 5's session is the closing data).

**Tech Stack:** Rust intrinsics (`std::arch::x86_64`: AVX2+FMA, then AVX-512 F/BW/VL/VNNI), proptest rig, criterion µbench vs pinned ggml, bash (`scripts/quiet-hw/`), devenv.nix (Intel SDE), mise tasks, PhoenixNAP bare metal via `mise run metal`.

**Spec:** [`docs/superpowers/specs/2026-07-17-m4b13-prefill-gemm-register-tiles-vnni-design.md`](../specs/2026-07-17-m4b13-prefill-gemm-register-tiles-vnni-design.md) (committed `6e05ed0`).

## Global Constraints

Copied from the spec. Every task's requirements implicitly include these.

- **Prefill GEMM (`m > 1`) only.** GEMV and every decode path are untouched; no tg claim is made. Q4_K and F32 keep their existing kernels.
- **Q8_0 only** (the criterion model, `models/qwen2.5-0.5b-instruct-q8_0.gguf`).
- **Standing invariants, no tolerance loosening:** `gemm(m=1)` bit-equals `gemv`; scalar-vs-SIMD bit-identity per ISA (VNNI joins the rig at the same bar); cross-thread and cross-`prefill_tile` bit-identity; compiled-vs-interpreter differential green unchanged.
- **Ladder discipline:** Lever 1's local gate blocks metal spend; Lever 2 ships only if the Task 5 pre-registered rule authorizes it. The ½ VNNI-ceiling factor is fixed in the spec, **not adjustable at gate time**. An all-STOP with the finding is a successful outcome.
- **Exit criterion:** pp vs llama best-of ≥ 1.0x on **both** boxes (d2.c1.medium 16c 6336Y, s2.c2.medium 8c E-2388G), or the recorded STOP finding.
- **Never edit a recorded data point.** Session output is pasted verbatim into spec Amendments. **Scripts never write to `docs/`** — verdicts are computed and pasted by a human.
- **Perf numbers come only from quiet bare metal** (`mise run metal`), except the Task 3 dev-box local gate, which is honestly labeled non-quiet and never judges the exit criterion. No CI perf gates.
- **Workflows are mise tasks:** `mise run test` / `lint` / `bench-kernels` / `metal`. Run `mise run lint` before every push (CI runs clippy `-D warnings`; `mise run test` alone skips it).
- **Metal runbook** (`docs/runbooks/metal.md`): no parallel PNAP provisions; retry one transient devpod post-create panic; on 406 check catalog stock and pass `--location`; after ANY failed session run `mise run metal-gc` and confirm zero servers.

## File Structure

| File | Responsibility |
|---|---|
| `crates/inferno-kernels/tests/rig.rs` (modify, Tasks 1, 7) | MR-boundary/`PREFILL_TILE`-shaped gemm coverage (Task 1); `Avx512Vnni` arms in the `gemv_*`/`gemm_*` helpers (Task 7). |
| `crates/inferno-kernels/src/pf.rs` (modify, Task 2) | `parse_gemm_mr` — compile-time `INFERNO_GEMM_MR` parsing (sibling of `parse_pf_dist`). |
| `crates/inferno-kernels/src/q8_0.rs` (modify, Tasks 2, 7) | `MR` const + register-tiled scalar/AVX2 GEMM (Task 2); `inferno_gemm_q8_0_rs8_avx512vnni` (Task 7). |
| `crates/inferno-kernels/src/lib.rs` (modify, Task 7) | `KernelIsa::Avx512Vnni` variant, `available()`, `all_available()`. |
| `crates/inferno-kernels/src/registry.rs` (modify, Task 7) | `Avx512Vnni` kernel set (q8_0 gemm only; everything else folds to the Avx2 fns); `kernels_for` v4 arm. |
| `crates/inferno-codegen/src/loopir.rs` (modify, Task 7) | `gemv_symbol`/`gemm_symbol`/`attention_symbol` mapping for `Avx512Vnni`. |
| `crates/inferno-codegen/src/lib.rs` (modify, Task 7) | `HOST_ABI_VERSION` `"7"` → `"8"`. |
| `crates/inferno-core/src/artifact.rs` (modify, Task 7) | `ensure_kernels_linked` gains the VNNI gemm symbol. |
| `scripts/quiet-hw/gate-prefill-attr.sh` (create, Task 4) | Fresh split-bracket t=1 prefill profile (the gate's matmul_share input). |
| `scripts/quiet-hw/verify.sh` (modify, Task 4) | Wire the new gate into the pass. |
| `docs/runbooks/quiet-hw-verification.md` (modify, Task 4) | Verdict-destination row for gate-prefill-attr. |
| `devenv.nix`, `devenv.yaml` (modify, Task 6) | `intel-sde` derivation (+ `allowUnfree`), `pkgs.jq`. |
| `scripts/test-vnni.sh` (create, Task 6) | Build rig natively, run it under `sde64 -icl`. |
| `mise.toml` (modify, Task 6) | `test-vnni` task. |
| `.github/workflows/nightly.yml` (modify, Task 6) | Nightly SDE lane (blocking tier stays ≤ 5 min). |
| `AGENTS.md` (modify, Task 8) | VNNI path + `test-vnni` note at closure (only if Lever 2 landed). |
| `docs/superpowers/specs/2026-07-17-m4b13-...-design.md` (modify, Tasks 3, 5, 8) | §Amendments: dev data point, session records, gate verdict, exit walk. |
| `docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md` (modify, Task 8) | §Amendments: closing protocol runs, verbatim (standing convention). |

---

### Task 1: Rig coverage for MR-tile boundaries

The existing q8_0 gemm proptests stop at `m < 6` — they never cross an MR=4 or MR=8 tile boundary with a tail, and never see a `PREFILL_TILE`-shaped panel. Strengthen the rig **before** touching the kernel; these tests must pass on the current code (they assert properties the restructure must preserve, not new behavior).

**Files:**
- Modify: `crates/inferno-kernels/tests/rig.rs`

**Interfaces:**
- Consumes: existing rig helpers `gemv_q8_0(isa, &w, &xq, rows, k, (lo, hi), &mut y)` and `gemm_q8_0(isa, &w, &panel, k, m, rows, (lo, hi), &mut y)` (both already in `rig.rs`).
- Produces: nothing new — stronger properties over the same helpers.

- [ ] **Step 1: Widen the q8_0 gemm proptest `m` ranges**

In `rig.rs`, change the three q8_0 gemm proptests' `m` strategy so tiles and tails are exercised at every MR the sweep will try (2, 4, 8):

- `q8_0_gemm_rows_match_per_token_gemv`: `m in 1usize..6` → `m in 1usize..20`
- `q8_0_gemm_range_partition_bitwise`: `m in 1usize..4` → `m in 1usize..20`

(`q8_0_gemm_m1_equals_gemv` stays m=1 by definition.)

- [ ] **Step 2: Add the deterministic PREFILL_TILE-shaped case**

Append to `rig.rs` (uses the same imports/helpers as the neighboring q8_0 tests):

```rust
/// M4b.13: a PREFILL_TILE-shaped panel (m = 64) crossing many register
/// tiles, with rows spanning full strips plus a partial tail — the shape
/// the tiled fast path sees in production. Every token must bit-equal an
/// independent scalar gemv on that token, on every runnable ISA.
#[test]
fn q8_0_gemm_prefill_tile_matches_per_token_gemv() {
    let (rows, k, m) = (28usize, 128usize, 64usize);
    let vals = pseudo(7, rows * k);
    let w = q8_0::pack_q8_0_rs8(&quant::pack(&DType::Q8_0, &vals).unwrap(), rows, k).unwrap();
    let mut panel = Vec::new();
    let mut per_token = Vec::new();
    for t in 0..m {
        let x = pseudo(0x300 + t as u64, k);
        let xq = act::quantize_row_q8a(KernelIsa::Scalar, &x).unwrap();
        panel.extend_from_slice(&xq);
        let mut yv = vec![f32::NAN; rows];
        gemv_q8_0(KernelIsa::Scalar, &w, &xq, rows, k, (0, rows), &mut yv);
        per_token.push(yv);
    }
    for isa in KernelIsa::all_available() {
        let mut yg = vec![f32::NAN; m * rows];
        gemm_q8_0(isa, &w, &panel, k, m, rows, (0, rows), &mut yg);
        for t in 0..m {
            for r in 0..rows {
                assert_eq!(
                    yg[t * rows + r].to_bits(),
                    per_token[t][r].to_bits(),
                    "t{t} r{r} isa {isa:?}"
                );
            }
        }
    }
}
```

If `KernelIsa` doesn't derive `Debug`, drop the `isa {isa:?}` fragment rather than adding a derive.

- [ ] **Step 3: Run the rig — must pass on the current kernels**

Run: `cargo nextest run -p inferno-kernels --test rig`
Expected: PASS (all tests, including the new ones — the properties already hold).

- [ ] **Step 4: Commit**

```bash
git add crates/inferno-kernels/tests/rig.rs
git commit -m "rig: cover MR-tile boundaries and a PREFILL_TILE-shaped gemm panel (M4b.13 Task 1)"
```

---

### Task 2: Register-tiled scalar + AVX2 GEMM (Lever 1)

Restructure the Q8_0 GEMM fast path: process tokens in `MR`-wide tiles whose accumulators stay in registers across the whole block loop, with each weight group's qs vectors and sign-magnitude split computed **once per group per tile** instead of once per token. The per-`(t, r)` f32 combine still walks blocks `0..nb` in order — output bits are unchanged by construction, and the Task 1 rig proves it.

The scalar sibling mirrors the same tiling (spec: the rig compares like to like). The partial head/tail **row** path in the AVX2 kernel is untouched.

**Files:**
- Modify: `crates/inferno-kernels/src/pf.rs` (add `parse_gemm_mr`)
- Modify: `crates/inferno-kernels/src/q8_0.rs` (MR const; restructure `inferno_gemm_q8_0_rs8_scalar` and `inferno_gemm_q8_0_rs8_avx2`)

**Interfaces:**
- Consumes: `STRIP` (=8), `WBLOCK` (=32), `GROUP_BYTES` (=288), `Q8A_BLOCK_BYTES`, `hsum8_i32`, `hsum_i32` — all existing.
- Produces: same `extern "C"` symbols, same ABI, same bits. New const `MR: usize` (private to `q8_0.rs`) and `pub(crate) const fn parse_gemm_mr(&str) -> usize` in `pf.rs`. Task 7's VNNI kernel reuses `MR` and the same tiling structure.

- [ ] **Step 1: Add `parse_gemm_mr` to `pf.rs`**

Mirror `parse_pf_dist`'s const-fn body exactly (same digit loop), with its own messages and a range assert:

```rust
/// Compile-time parsing for the `INFERNO_GEMM_MR` register-tile width
/// (M4b.13 µbench sweep). 1..=16; pure loop restructuring, so output bits
/// never depend on the value.
pub(crate) const fn parse_gemm_mr(s: &str) -> usize {
    let b = s.as_bytes();
    assert!(!b.is_empty(), "INFERNO_GEMM_MR must be a decimal integer");
    let mut v = 0usize;
    let mut i = 0;
    while i < b.len() {
        assert!(
            b[i] >= b'0' && b[i] <= b'9',
            "INFERNO_GEMM_MR must be a decimal integer"
        );
        v = v * 10 + (b[i] - b'0') as usize;
        i += 1;
    }
    assert!(v >= 1 && v <= 16, "INFERNO_GEMM_MR must be in 1..=16");
    v
}
```

Add a `#[test]` beside `parse_pf_dist`'s existing tests covering `"4"`, `"16"`, and (via `#[should_panic]` if that's the local pattern, else skip) rejection.

- [ ] **Step 2: Add the `MR` const to `q8_0.rs`**

Below `PF_DIST`, following its documentation pattern:

```rust
/// Register-tile token-group width for the batched GEMM fast path (M4b.13):
/// MR tokens share each weight-group load and keep their accumulators
/// register-resident across the block loop. Pure loop restructuring — the
/// per-(t,r) block-order f32 combine is unchanged at any value, so output
/// bits never depend on it. Compile-time override via `INFERNO_GEMM_MR`
/// for the Task 3 µbench sweep; default fixed by that sweep's Amendment.
const MR: usize = match option_env!("INFERNO_GEMM_MR") {
    Some(s) => crate::pf::parse_gemm_mr(s),
    None => 4,
};
```

- [ ] **Step 3: Restructure the scalar GEMM**

Replace the body of `inferno_gemm_q8_0_rs8_scalar` (keep signature, safety doc; extend the doc comment's batching note to mention the MR tiling):

```rust
    let nb = k / WBLOCK;
    let act = Q8A_BLOCK_BYTES * nb; // per-token activation stride
    for r in row_start..row_end {
        let (strip, lane) = (r / STRIP, r % STRIP);
        // MR-token tiles then a shorter tail tile; per (t,r) the block
        // order stays 0..nb → gemv order (bit-identity by construction).
        let mut t0 = 0;
        while t0 < m {
            let mr = MR.min(m - t0);
            let mut acc = [0f32; 16]; // MR ≤ 16 (parse_gemm_mr bound)
            for b in 0..nb {
                let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                let dw = f32::from_le_bytes(unsafe { g.add(lane * 4).cast::<[u8; 4]>().read() });
                let qw = unsafe { g.add(32 + lane * WBLOCK) };
                for (ti, at) in acc.iter_mut().enumerate().take(mr) {
                    let xb = unsafe { xq.add((t0 + ti) * act + b * Q8A_BLOCK_BYTES) };
                    let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                    let qx = unsafe { xb.add(4) };
                    let mut isum = 0i32;
                    for i in 0..WBLOCK {
                        let a = i32::from(unsafe { qw.add(i).cast::<i8>().read() });
                        let bb = i32::from(unsafe { qx.add(i).cast::<i8>().read() });
                        isum += a * bb;
                    }
                    *at = (dw * dx).mul_add(isum as f32, *at);
                }
            }
            for (ti, at) in acc.iter().enumerate().take(mr) {
                unsafe { y.add((t0 + ti) * rows + r).write(*at) };
            }
            t0 += mr;
        }
    }
```

- [ ] **Step 4: Restructure the AVX2 GEMM full-strip fast path**

In `inferno_gemm_q8_0_rs8_avx2`, replace the full-strip branch (the `if lane0 == 0 && r + STRIP <= row_end` block) with the tiled version. The partial-row path below it is untouched.

```rust
        // Full-strip fast path (M4b.13 register tiles): MR tokens per tile,
        // accumulators register-resident across the block loop; each group's
        // weight vectors + sign-magnitude split computed once per tile
        // (previously once per token). Per (t,r) the block-order f32 combine
        // is unchanged → bit-identical to gemv.
        if lane0 == 0 && r + STRIP <= row_end {
            let mut t0 = 0;
            while t0 + MR <= m {
                let mut acc = [_mm256_setzero_ps(); MR];
                for b in 0..nb {
                    let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                    let qs = unsafe { g.add(32) };
                    let dw = unsafe { _mm256_load_ps(g.cast()) };
                    let mut wv = [_mm256_setzero_si256(); STRIP];
                    let mut aw = [_mm256_setzero_si256(); STRIP];
                    for (lane, (wvl, awl)) in wv.iter_mut().zip(&mut aw).enumerate() {
                        *wvl = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                        *awl = _mm256_sign_epi8(*wvl, *wvl);
                    }
                    for (ti, at) in acc.iter_mut().enumerate() {
                        let xb = unsafe { xq.add((t0 + ti) * act + b * Q8A_BLOCK_BYTES) };
                        let dx =
                            f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                        let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                        let mut p = [_mm256_setzero_si256(); STRIP];
                        for (lane, pl) in p.iter_mut().enumerate() {
                            let sx = _mm256_sign_epi8(xv, wv[lane]);
                            *pl = _mm256_madd_epi16(_mm256_maddubs_epi16(aw[lane], sx), ones);
                        }
                        let isum = _mm256_cvtepi32_ps(hsum8_i32(p));
                        let dwdx = _mm256_mul_ps(dw, _mm256_set1_ps(dx));
                        *at = _mm256_fmadd_ps(dwdx, isum, *at);
                    }
                }
                for (ti, at) in acc.iter().enumerate() {
                    unsafe { _mm256_storeu_ps(y.add((t0 + ti) * rows + r), *at) };
                }
                t0 += MR;
            }
            // Token tail (m % MR): per-token, same block body (weight loads
            // per token again — a vanishing share at PREFILL_TILE = 64).
            for t in t0..m {
                let mut at = _mm256_setzero_ps();
                for b in 0..nb {
                    let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                    let qs = unsafe { g.add(32) };
                    let dw = unsafe { _mm256_load_ps(g.cast()) };
                    let xb = unsafe { xq.add(t * act + b * Q8A_BLOCK_BYTES) };
                    let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                    let xv = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                    let mut p = [_mm256_setzero_si256(); STRIP];
                    for (lane, pl) in p.iter_mut().enumerate() {
                        let wv = unsafe { _mm256_load_si256(qs.add(lane * WBLOCK).cast()) };
                        let aw = _mm256_sign_epi8(wv, wv);
                        let sx = _mm256_sign_epi8(xv, wv);
                        *pl = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
                    }
                    let isum = _mm256_cvtepi32_ps(hsum8_i32(p));
                    let dwdx = _mm256_mul_ps(dw, _mm256_set1_ps(dx));
                    at = _mm256_fmadd_ps(dwdx, isum, at);
                }
                unsafe { _mm256_storeu_ps(y.add(t * rows + r), at) };
            }
            r += STRIP;
            continue;
        }
```

Note the tile loop is **inside** the strip loop: a strip's weights (8–44 KB for the profile-blamed shapes) are re-read from L1/L2 once per tile — cheap — while accumulators and the sign-magnitude split live in registers. Do NOT hoist the tile loop above the strip loop (that would re-stream the whole weight matrix `m/MR` times from DRAM).

- [ ] **Step 5: Rig at the default and swept MR values**

Run:
```bash
cargo nextest run -p inferno-kernels --test rig
INFERNO_GEMM_MR=2 cargo nextest run -p inferno-kernels --test rig
INFERNO_GEMM_MR=8 cargo nextest run -p inferno-kernels --test rig
INFERNO_GEMM_MR=1 cargo nextest run -p inferno-kernels --test rig
```
Expected: PASS ×4 (bit-identity holds at every MR — including MR=1, the degenerate old shape).

- [ ] **Step 6: Full blocking tier**

Run: `mise run test`
Expected: PASS (the codegen differential and core artifact suites exercise the gemm path end-to-end; zero tolerance change).

- [ ] **Step 7: Commit**

```bash
git add crates/inferno-kernels/src/pf.rs crates/inferno-kernels/src/q8_0.rs
git commit -m "kernels: register-tiled Q8_0 GEMM (MR-token tiles, register-resident acc) — M4b.13 Lever 1"
```

---

### Task 3: µbench MR sweep + dev-box local gate (blocks metal spend)

Fix `MR` from data and record the Lever-1 local-gate data point. This is the spec's **local iteration gate**: if the tiled kernel does not beat the current one on the blamed shapes, iterate on the tile structure — do not proceed to Task 4/5.

**Files:**
- Modify: `crates/inferno-kernels/src/q8_0.rs` (only if the sweep picks a default ≠ 4)
- Modify: `docs/superpowers/specs/2026-07-17-m4b13-prefill-gemm-register-tiles-vnni-design.md` (§Amendments)

**Interfaces:**
- Consumes: the existing `gemm/Q8_0` criterion group (`benches/gemv.rs`, shapes include 4864×896, 896×4864, 151936×896; `m ∈ {1,16,64}`; `--features ggml-compare` adds the ggml rows).
- Produces: the fixed `MR` default; the recorded before/after µbench + dev t=1 pp data point.

- [ ] **Step 1: Baseline µbench (pre-tiling), from a scratch worktree**

```bash
BASE=$(git rev-parse HEAD~1)   # the commit before Task 2's kernel change
git worktree add /tmp/m4b13-base "$BASE"
(cd /tmp/m4b13-base && cargo bench -p inferno-kernels --features ggml-compare -- 'gemm/Q8_0')
```
Record the `q8_0/avx2` and `ggml` throughput rows for `4864x896/m64`, `896x4864/m64`, `151936x896/m64`. Dev box is not quiet — this is same-machine, same-session A/B, labeled as such. Remove the worktree (`git worktree remove /tmp/m4b13-base`) after Step 3's "before" bench run.

- [ ] **Step 2: Sweep MR on the tiled kernel**

```bash
INFERNO_GEMM_MR=2 cargo bench -p inferno-kernels --features ggml-compare -- 'gemm/Q8_0'
INFERNO_GEMM_MR=4 cargo bench -p inferno-kernels --features ggml-compare -- 'gemm/Q8_0'
INFERNO_GEMM_MR=8 cargo bench -p inferno-kernels --features ggml-compare -- 'gemm/Q8_0'
```
Pick the MR with the best geomean over the three blamed shapes at m=64. If it isn't 4, change the `None => 4` default in `q8_0.rs` and re-run `cargo nextest run -p inferno-kernels --test rig`.

- [ ] **Step 3: Dev-box t=1 before/after (same session)**

```bash
# after (tiled), from the repo root inside devenv shell:
mise run bench -- models/qwen2.5-0.5b-instruct-q8_0.gguf
# before: same command from the pre-Task-2 scratch worktree
```
Record both runs' `inferno (t=1 diag)` pp rows. **Local gate: tiled µbench Gelem/s beats baseline on all three blamed shapes AND t=1 pp improves.** If not met: iterate on Task 2's tile structure (this is the spec's sanctioned loop), re-run this task.

- [ ] **Step 4: Record the data point in the spec §Amendments**

Append to the spec a `### 2026-MM-DD — Lever 1 dev data point (Task 3 local gate)` section: the µbench table (baseline vs MR sweep vs ggml, the three shapes at m=64), the chosen MR, the t=1 pp before/after, and the sentence "Dev box (Zen 2), not quiet hardware — local gate only, never the exit criterion."

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/src/q8_0.rs docs/superpowers/specs/2026-07-17-m4b13-prefill-gemm-register-tiles-vnni-design.md
git commit -m "specs: M4b.13 Lever 1 dev data point + MR default (Task 3 local gate)"
```

---

### Task 4: `gate-prefill-attr.sh` — the mid-milestone attribution gate script

The fresh split-bracket t=1 prefill profile the pre-registered ladder rule consumes. Follows `gate-attn-split.sh`'s conventions exactly (lib.sh, `QHW_OUT`/`QHW_SMOKE`, verbatim table print, VERDICTS ARE HUMAN).

**Files:**
- Create: `scripts/quiet-hw/gate-prefill-attr.sh`
- Modify: `scripts/quiet-hw/verify.sh` (one `run_gate` line)
- Modify: `docs/runbooks/quiet-hw-verification.md` (verdict-destination row)

**Interfaces:**
- Consumes: `scripts/quiet-hw/lib.sh` helpers (`smoke_header`, `machine_block`), `inferno run --profile` (per-op prefill table with the M4b.9 split brackets), `gate-bench-protocol.sh` (already in verify.sh — provides the pp ratios in the same session).
- Produces: `$QHW_OUT/prefill-attr-t1.txt` and the printed t=1 prefill op table.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# M4b.13 mid-milestone attribution gate — the fresh split-bracket t=1
# prefill profile the pre-registered ladder rule consumes (spec §Mid-
# Milestone Gate). Prints the t=1 prefill op table verbatim; the pp ratios
# come from gate-bench-protocol.sh in the same session. VERDICTS ARE
# HUMAN: paste into the M4b.13 spec §Amendments and compute there
# matmul_share (sum of the prefill table's matmul:* rows / prefill total)
# and the ceiling check pp_ratio / (1 - matmul_share * 0.5) >= 1.0, per
# the spec's pre-registered rule (docs/runbooks/quiet-hw-verification.md).
# Usage: gate-prefill-attr.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-prefill-attr.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PROMPT_BYTES=256; MAXTOK=8; else PROMPT_BYTES=2048; MAXTOK=32; fi

smoke_header "gate-prefill-attr (M4b.13: split-bracket t=1 prefill profile)"
machine_block
echo

PROMPT="$(head -c "$PROMPT_BYTES" /dev/urandom | base64 | tr -d '\n')"

echo "--- t=1 prefill profile (split brackets) ---"
cargo run --release -q -p inferno -- run "$MODEL" \
  --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads 1 --profile \
  > "$OUT/prefill-attr-t1.txt" 2>&1
sed -n '/^profile \[/,$p' "$OUT/prefill-attr-t1.txt"
```

`chmod +x scripts/quiet-hw/gate-prefill-attr.sh`.

- [ ] **Step 2: Wire into verify.sh**

Next to the existing `run_gate` lines (after `bench-protocol`):

```bash
run_gate prefill-attr    bash "$HERE/gate-prefill-attr.sh" "$MODEL"
```

- [ ] **Step 3: Runbook row**

Add to the gate table in `docs/runbooks/quiet-hw-verification.md`, matching the existing rows' format: `gate-prefill-attr` → verdict destination "M4b.13 spec §Amendments (matmul_share + ceiling-check arithmetic, human-computed)".

- [ ] **Step 4: Smoke the script locally**

Run: `QHW_SMOKE=1 bash scripts/quiet-hw/gate-prefill-attr.sh models/qwen2.5-0.5b-instruct-q8_0.gguf`
Expected: header + machine block + a prefill op table with the split brackets (`matmul:*`, `attention`, `kv_append`, `quantize` rows). Non-recordable smoke output — do not paste anywhere.

- [ ] **Step 5: Commit**

```bash
git add scripts/quiet-hw/gate-prefill-attr.sh scripts/quiet-hw/verify.sh docs/runbooks/quiet-hw-verification.md
git commit -m "quiet-hw: gate-prefill-attr — split-bracket t=1 prefill profile (M4b.13 gate input)"
```

---

### Task 5: Mid-milestone quiet-hw sessions + the pre-registered gate verdict

**PR/merge checkpoint first:** Tasks 1–4 are the complete Lever-1 change; push the branch, run `mise run lint`, open the PR, get it merged (the sessions bench a commit on `main`, per the standing convention). Then run the sessions — **sequentially, never two provisions in parallel** — and apply the gate rule. **No task after this one may assume the outcome.**

**Files:**
- Modify: `docs/superpowers/specs/2026-07-17-m4b13-prefill-gemm-register-tiles-vnni-design.md` (§Amendments: session records + gate verdict)

**Interfaces:**
- Consumes: `mise run metal` (runbook `docs/runbooks/metal.md`), `gate-bench-protocol.sh`, `gate-prefill-attr.sh`.
- Produces: the gate verdict that enables or skips Tasks 6–7.

- [ ] **Step 1: Session A — d2.c1.medium (6336Y 16c)**

Per `docs/runbooks/metal.md`, provision and run the workload: `gate-bench-protocol.sh` then `gate-prefill-attr.sh` on the criterion model. On any failure: `mise run metal-gc` and confirm zero servers before retrying.

- [ ] **Step 2: Session B — s2.c2.medium (E-2388G 8c)**

Same workload, after Session A is fully deprovisioned.

- [ ] **Step 3: Record both sessions verbatim**

Append to the spec §Amendments: `### 2026-MM-DD — mid-milestone gate sessions (Lever 1 on both boxes)` with each box's `gate-bench-protocol.out` and t=1 prefill op table pasted verbatim.

- [ ] **Step 4: Apply the pre-registered rule, arithmetic shown**

In the same Amendment, compute per box: `matmul_share` = Σ `matmul:*` prefill cycles ÷ prefill total; the measured pp ratio vs llama best-of; then walk the rule **exactly as pre-registered**:

1. pp ≥ 1.0x on both boxes → exit criterion met; Tasks 6–7 SKIPPED; go to Task 8.
2. pp < 1.0x on either box AND `pp_ratio / (1 − matmul_share × 0.5) ≥ 1.0` on every box still under 1.0x → **Lever 2 authorized**; Tasks 6–8 run.
3. Otherwise → **STOP-out**: record the finding (where the residual lives, from the fresh profile), Tasks 6–7 SKIPPED; go to Task 8 and close as diagnostic.

The ½ factor is fixed by the spec — not adjustable here.

- [ ] **Step 5: Commit the verdict**

```bash
git add docs/superpowers/specs/2026-07-17-m4b13-prefill-gemm-register-tiles-vnni-design.md
git commit -m "specs: M4b.13 mid-milestone gate sessions + ladder verdict (arithmetic shown)"
```

---

### Task 6 (GATED — only if Task 5 authorized Lever 2): Intel SDE test harness

The dev box is Zen 2; the VNNI kernel cannot execute locally. Land the emulation harness **before** the kernel so Task 7 is test-first-able. SDE is not in nixpkgs (unfree); it's a pinned fetchurl derivation.

**Files:**
- Modify: `devenv.nix`, `devenv.yaml`
- Create: `scripts/test-vnni.sh`
- Modify: `mise.toml`
- Modify: `.github/workflows/nightly.yml`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `sde64` on the devenv PATH; `mise run test-vnni` (Task 7's test runner); the nightly CI lane.

- [ ] **Step 1: Pin the SDE tarball**

Try: `nix-prefetch-url https://downloadmirror.intel.com/843185/sde-external-9.44.0-2024-08-22-lin.tar.xz`
If 404: find the current Linux tarball URL on Intel's SDE download page (intel.com/content/www/us/en/download/684897/) and prefetch that instead, adjusting `version` below to match. Record the returned hash.

- [ ] **Step 2: The devenv derivation**

In `devenv.nix`'s `let` block (beside `llama-cpp-cpu`):

```nix
  # Intel SDE — CPU emulator for the M4b.13 AVX-512 VNNI kernel's
  # correctness tests on hosts without AVX-512 (dev box is Zen 2; CI not
  # guaranteed). Not in nixpkgs (unfree ISDL license); pinned from Intel.
  intel-sde = pkgs.stdenv.mkDerivation rec {
    pname = "intel-sde";
    version = "9.44.0-2024-08-22";
    src = pkgs.fetchurl {
      url = "https://downloadmirror.intel.com/843185/sde-external-${version}-lin.tar.xz";
      hash = "<hash from Step 1>";
    };
    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib ];
    installPhase = ''
      mkdir -p $out/opt/sde $out/bin
      cp -r . $out/opt/sde/
      ln -s $out/opt/sde/sde64 $out/bin/sde64
    '';
    meta.license = pkgs.lib.licenses.unfree;
  };
```

Add `intel-sde` and `pkgs.jq` to `packages` (inside the existing `pkgs.lib.optionals pkgs.stdenv.isLinux [...]` list for `intel-sde`; jq unconditional). If evaluation rejects the unfree license, add `allowUnfree: true` under the nixpkgs config in `devenv.yaml`.

Verify: `devenv shell -- sde64 -help | head -3` prints SDE usage.

- [ ] **Step 3: `scripts/test-vnni.sh`**

```bash
#!/usr/bin/env bash
# M4b.13: run the inferno-kernels rig under Intel SDE's Ice Lake model so
# the AVX-512 VNNI kernel is exercised on hosts without AVX-512. SDE
# virtualizes CPUID, so KernelIsa::Avx512Vnni.available() sees VNNI and
# all_available() folds the variant into every rig property. The binary is
# built natively (fast); only the test run is emulated (slow → trimmed
# PROPTEST_CASES; override to taste).
set -euo pipefail
command -v sde64 >/dev/null || { echo "missing sde64 (run inside 'devenv shell')" >&2; exit 2; }
BIN=$(cargo test --release -p inferno-kernels --test rig --no-run --message-format=json \
  | jq -r 'select(.executable != null and .target.name == "rig") | .executable' | tail -1)
PROPTEST_CASES="${PROPTEST_CASES:-64}" sde64 -icl -- "$BIN"
```

`chmod +x scripts/test-vnni.sh`. Mise task:

```toml
[tasks.test-vnni]
description = "Kernel rig under Intel SDE Ice Lake emulation — AVX-512 VNNI correctness on any x86 host (run inside devenv shell)"
run = "bash scripts/test-vnni.sh"
```

- [ ] **Step 4: Run it (pre-kernel)**

Run: `devenv shell -- mise run test-vnni`
Expected: PASS — the rig runs under SDE; only Scalar/Avx2 variants exist yet, so this proves the harness, not the kernel.

- [ ] **Step 5: Nightly CI lane**

Add a job to `.github/workflows/nightly.yml` (setup mirrors ci.yml's `check` job, which is the existing devenv-shell pattern; no rustfmt/clippy needed here):

```yaml
  # M4b.13: kernel rig under Intel SDE Ice Lake emulation — deterministic
  # AVX-512 VNNI coverage regardless of the runner's silicon. Nightly, not
  # blocking: SDE emulation is slow (blocking tier budget is ≤ 5 min).
  vnni-sde:
    runs-on: ubuntu-latest
    permissions:
      id-token: write # FlakeHub Cache OIDC auth
      contents: read
    steps:
      - uses: actions/checkout@v4
      - uses: DeterminateSystems/nix-installer-action@v16
      - uses: DeterminateSystems/flakehub-cache-action@v3
      - run: nix profile install nixpkgs#devenv
      - uses: jdx/mise-action@v2
      - uses: Swatinem/rust-cache@v2
      - run: devenv shell -- mise run test-vnni
        env:
          PROPTEST_CASES: "32"
```

(Per the standing memory note: a FlakeHub "path is not valid" failure in this job is runner cache corruption — rerun once before debugging.)

(Nightly, not blocking: SDE emulation is slow and the blocking tier's ≤ 5 min budget is a standing constraint.)

- [ ] **Step 6: Commit**

```bash
git add devenv.nix devenv.yaml devenv.lock scripts/test-vnni.sh mise.toml .github/workflows/nightly.yml
git commit -m "devenv: Intel SDE + mise run test-vnni — AVX-512 emulation harness (M4b.13 Lever 2 prep)"
```

---

### Task 7 (GATED — only if Task 5 authorized Lever 2): `KernelIsa::Avx512Vnni` + the VNNI GEMM kernel

The 512-bit `vpdpbusd` GEMM path, wired through the registry and codegen. GEMM only: GEMV, quantize, and attention resolve to the AVX2 kernels under the new variant. New host symbol → `HOST_ABI_VERSION` bump.

**Exactness argument (goes in the kernel doc comment):** weights are clamped to −127 by `pack_q8_0_rs8` and activations to [−127, 127] by the q8a quantizers, so `|w|` fits u8 [0,127] and the mask-negation of x never sees −128. `vpdpbusd` sums four u8×i8 products (≤ 4·127·127 = 64,516) into an i32 lane from zero — integer-exact, no i16 intermediate. AVX-512 has no `vpsignb`; sign-adjusting x via `_mm512_mask_sub_epi8` under the w<0 byte mask differs from `_mm256_sign_epi8` only for w == 0 lanes (no zeroing of x), where `|w| = 0` makes the product 0 anyway. The per-(t,r) f32 combine keeps gemv's block order → bit-identity with scalar/AVX2 demanded by the rig, not a tolerance.

**Files:**
- Modify: `crates/inferno-kernels/src/lib.rs` (enum variant)
- Modify: `crates/inferno-kernels/src/q8_0.rs` (the kernel)
- Modify: `crates/inferno-kernels/src/registry.rs` (kernel set + `kernels_for`)
- Modify: `crates/inferno-kernels/tests/rig.rs` (helper match arms)
- Modify: `crates/inferno-codegen/src/loopir.rs` (symbol selection)
- Modify: `crates/inferno-codegen/src/lib.rs` (`HOST_ABI_VERSION`)
- Modify: `crates/inferno-core/src/artifact.rs` (`ensure_kernels_linked`)

**Interfaces:**
- Consumes: `MR`, `hsum8_i32`, the Task 2 tiling structure, `inferno_gemm_q8_0_rs8_avx2` (tail delegation), `mise run test-vnni` (Task 6).
- Produces: `pub unsafe extern "C" fn inferno_gemm_q8_0_rs8_avx512vnni(y: *mut f32, xq: *const u8, w: *const u8, k: usize, m: usize, rows: usize, row_start: usize, row_end: usize)`; `KernelIsa::Avx512Vnni`; codegen emits `inferno_gemm_q8_0_rs8_avx512vnni` for (Q8_0, Avx512Vnni) gemm sites and `_avx2` symbols everywhere else under the variant.

- [ ] **Step 1: Add the enum variant (compile errors become the to-do list)**

In `lib.rs`:

```rust
pub enum KernelIsa {
    Scalar,
    Avx2,
    /// AVX-512 F/BW/VL + VNNI batched-GEMM path (M4b.13). Only Q8_0 GEMM
    /// has a VNNI kernel; GEMV, quantize, and attention resolve to the
    /// AVX2 kernels under this variant (spec: prefill GEMM only).
    Avx512Vnni,
}
```

`available()` arm (VNNI kernel delegates tails to the AVX2 kernel and combines in ymm FMA, so AVX2+FMA are also required):

```rust
            KernelIsa::Avx512Vnni => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vl")
                    && std::arch::is_x86_feature_detected!("avx512vnni")
                    && std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
```

`all_available()`: append `KernelIsa::Avx512Vnni` to the candidate array. Then `cargo check --workspace` and fix every non-exhaustive match it reports, per Steps 3–5.

- [ ] **Step 2: The kernel in `q8_0.rs`**

```rust
/// # Safety
/// As [`inferno_gemm_q8_0_rs8_scalar`]; additionally requires AVX-512
/// F/BW/VL + VNNI (and AVX2+FMA for the delegated tails).
///
/// Exactness: pack clamps weights to −127 and q8a clamps activations to
/// [−127, 127], so |w| fits u8 and mask-negating x never sees −128.
/// vpdpbusd sums four u8×i8 products into an i32 lane from zero —
/// integer-exact. AVX-512 lacks vpsignb; `_mm512_mask_sub_epi8` under the
/// w<0 byte mask differs from `_mm256_sign_epi8` only at w == 0 bytes
/// (x not zeroed), where |w| = 0 zeroes the product anyway. The per-(t,r)
/// f32 combine keeps gemv's block order → bit-identical to scalar/AVX2.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_gemm_q8_0_rs8_avx512vnni(
    y: *mut f32,
    xq: *const u8,
    w: *const u8,
    k: usize,
    m: usize,
    rows: usize,
    row_start: usize,
    row_end: usize,
) {
    use std::arch::x86_64::*;
    let nb = k / WBLOCK;
    let act = Q8A_BLOCK_BYTES * nb;
    // Head rows before the first strip boundary and tail rows after the
    // last: delegate to the AVX2 kernel (bit-identical; vanishing share).
    let first_full = row_start.next_multiple_of(STRIP);
    if first_full >= row_end {
        unsafe { inferno_gemm_q8_0_rs8_avx2(y, xq, w, k, m, rows, row_start, row_end) };
        return;
    }
    if row_start < first_full {
        unsafe { inferno_gemm_q8_0_rs8_avx2(y, xq, w, k, m, rows, row_start, first_full) };
    }
    let full_end = first_full + (row_end - first_full) / STRIP * STRIP;
    if full_end < row_end {
        unsafe { inferno_gemm_q8_0_rs8_avx2(y, xq, w, k, m, rows, full_end, row_end) };
    }
    // Token tail (m % MR) over the full-strip span: one AVX2 call on the
    // panel suffix (y is token-major, so the suffix starts at m_full*rows).
    let m_full = m / MR * MR;
    if m_full < m {
        unsafe {
            inferno_gemm_q8_0_rs8_avx2(
                y.add(m_full * rows),
                xq.add(m_full * act),
                w,
                k,
                m - m_full,
                rows,
                first_full,
                full_end,
            )
        };
    }
    // VNNI fast path: full strips × full MR tiles.
    let mut r = first_full;
    while r < full_end {
        let strip = r / STRIP;
        let mut t0 = 0;
        while t0 + MR <= m_full {
            let mut acc = [_mm256_setzero_ps(); MR];
            for b in 0..nb {
                let g = unsafe { w.add((strip * nb + b) * GROUP_BYTES) };
                let qs = unsafe { g.add(32) };
                let dw = unsafe { _mm256_load_ps(g.cast()) };
                // 8 lanes × 32 B qs = four 64 B zmm loads (groups are only
                // 32-aligned → loadu). zmm j holds lanes 2j and 2j+1.
                let mut awz = [_mm512_setzero_si512(); 4];
                let mut neg = [0u64; 4];
                for j in 0..4 {
                    let wz = unsafe { _mm512_loadu_si512(qs.add(j * 64).cast()) };
                    awz[j] = _mm512_abs_epi8(wz);
                    neg[j] = _mm512_movepi8_mask(wz);
                }
                for (ti, at) in acc.iter_mut().enumerate() {
                    let xb = unsafe { xq.add((t0 + ti) * act + b * Q8A_BLOCK_BYTES) };
                    let dx = f32::from_le_bytes(unsafe { xb.cast::<[u8; 4]>().read_unaligned() });
                    let x256 = unsafe { _mm256_loadu_si256(xb.add(4).cast()) };
                    // Both 256-bit halves = this token's activation block.
                    let xv = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(x256), x256);
                    let mut p = [_mm256_setzero_si256(); STRIP];
                    for j in 0..4 {
                        let sx =
                            _mm512_mask_sub_epi8(xv, neg[j], _mm512_setzero_si512(), xv);
                        let d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), awz[j], sx);
                        // Low ymm = lane 2j's 8 partials, high = lane 2j+1's.
                        p[2 * j] = _mm512_castsi512_si256(d);
                        p[2 * j + 1] = _mm512_extracti64x4_epi64::<1>(d);
                    }
                    let isum = _mm256_cvtepi32_ps(hsum8_i32(p));
                    let dwdx = _mm256_mul_ps(dw, _mm256_set1_ps(dx));
                    *at = _mm256_fmadd_ps(dwdx, isum, *at);
                }
            }
            for (ti, at) in acc.iter().enumerate() {
                unsafe { _mm256_storeu_ps(y.add((t0 + ti) * rows + r), *at) };
            }
            t0 += MR;
        }
        r += STRIP;
    }
}
```

Intrinsic signatures drift between Rust versions (`_mm512_loadu_si512` pointer type, `_mm512_movepi8_mask` return) — adjust casts to what the pinned toolchain expects; the structure above is the contract.

- [ ] **Step 3: Registry wiring**

In `registry.rs::set()`: for the `DType::Q8_0` arm's `gemm`, add `KernelIsa::Avx512Vnni => q8_0::inferno_gemm_q8_0_rs8_avx512vnni`. Every other `match isa` in `set()` (q8_0 gemv/quantize; the whole f32 and q4_k arms) folds the new variant into the AVX2 arm: `KernelIsa::Avx2 | KernelIsa::Avx512Vnni => ...` with a one-line comment `// no VNNI kernel — AVX2 fns (spec: Q8_0 prefill GEMM only)`.

In `kernels_for()`:

```rust
    let kisa = match isa {
        Isa::X86_64v3 => KernelIsa::Avx2,
        // v4: the VNNI batched-GEMM set when the running CPU has it
        // (M4b.13); otherwise the AVX2 set, as since M2.
        Isa::X86_64v4 => {
            if KernelIsa::Avx512Vnni.available() {
                KernelIsa::Avx512Vnni
            } else {
                KernelIsa::Avx2
            }
        }
    };
```

`attention_kernel()`'s `match isa` keeps mapping both v3 and v4 through the same path (unchanged — it never consults KernelIsa variants beyond availability).

- [ ] **Step 4: Codegen symbol selection + ABI bump**

`loopir.rs`:

```rust
    // gemv_symbol: GEMV has no VNNI variant (spec: prefill GEMM only).
    let i = match isa {
        inferno_kernels::KernelIsa::Scalar => "scalar",
        inferno_kernels::KernelIsa::Avx2 | inferno_kernels::KernelIsa::Avx512Vnni => "avx2",
    };
```

```rust
pub fn gemm_symbol(dtype: &DType, isa: inferno_kernels::KernelIsa) -> String {
    // Only (Q8_0, Avx512Vnni) has a VNNI gemm; everything else derives
    // from the gemv symbol as before.
    if matches!(dtype, DType::Q8_0) && matches!(isa, inferno_kernels::KernelIsa::Avx512Vnni) {
        return "inferno_gemm_q8_0_rs8_avx512vnni".to_string();
    }
    gemv_symbol(dtype, isa).replace("_gemv_", "_gemm_")
}
```

`attention_symbol`: fold `Avx512Vnni` into the `"avx2"` arm the same way.

`lib.rs`: `HOST_ABI_VERSION` → `"8"`, prepending to the doc comment: `/// "8" = M4b.13's VNNI batched-GEMM symbol (inferno_gemm_q8_0_rs8_avx512vnni);`.

`artifact.rs::ensure_kernels_linked()`: add `p(inferno_kernels::inferno_gemm_q8_0_rs8_avx512vnni as *const ());` beside the other gemm symbols.

- [ ] **Step 5: Rig helper arms**

In `rig.rs`, every `match isa` helper (`gemv_f32`/`gemm_f32`/`gemv_q8_0`/`gemm_q8_0`/q4_k equivalents — the compiler lists them): `gemm_q8_0` gets `KernelIsa::Avx512Vnni => inferno_kernels::inferno_gemm_q8_0_rs8_avx512vnni(...)`; all others fold into the Avx2 arm (`KernelIsa::Avx2 | KernelIsa::Avx512Vnni =>`), which is exactly the dispatch the registry performs.

- [ ] **Step 6: Test — native (variant skipped) and under SDE (variant exercised)**

```bash
mise run test          # dev box: Avx512Vnni not available → rig skips it; snapshots unchanged (isa suffix is stripped in LoopIr dumps — verify the snapshot tests pass)
mise run test-vnni     # SDE: all rig properties now include Avx512Vnni
```
Expected: PASS ×2. If the SDE run fails a bit-identity property, the kernel is wrong — fix the kernel; never the test or a tolerance.

- [ ] **Step 7: Lint + commit**

```bash
mise run lint
git add crates/inferno-kernels crates/inferno-codegen/src/loopir.rs crates/inferno-codegen/src/lib.rs crates/inferno-core/src/artifact.rs
git commit -m "kernels: AVX-512 VNNI Q8_0 GEMM (KernelIsa::Avx512Vnni, vpdpbusd, ABI v8) — M4b.13 Lever 2"
```

---

### Task 8: Closing sessions + exit-criteria walk (runs on every path)

If Tasks 6–7 ran: merge the Lever-2 PR, then run fresh closing sessions on both boxes — workload: `mise run test` on the box first (native VNNI rig + differential on real AVX-512 silicon, before any recorded number), then `gate-bench-protocol.sh` + `gate-prefill-attr.sh`. If Tasks 6–7 were skipped (rule 1 or 3): Task 5's sessions **are** the closing data — no new provision.

**Files:**
- Modify: `docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md` (§Amendments: closing protocol runs, verbatim)
- Modify: `docs/superpowers/specs/2026-07-17-m4b13-prefill-gemm-register-tiles-vnni-design.md` (§Amendments: closing verdict / exit-criteria walk)
- Modify: `AGENTS.md` (only if Lever 2 landed: one note on the VNNI kernel path + `mise run test-vnni`)

**Interfaces:**
- Consumes: everything above.
- Produces: the milestone's closing record.

- [ ] **Step 1: (Lever-2 path only) closing sessions, both boxes, sequential**

Per the metal runbook; paste each box's on-box `mise run test` summary line, `gate-bench-protocol.out`, and t=1 prefill op table verbatim into the M4a spec §Amendments under `### 2026-MM-DD — M4b.13 closing benches`.

- [ ] **Step 2: Exit-criteria walk in the M4b.13 spec §Amendments**

Walk the spec's four exit criteria, quoting the recorded numbers: (1) Lever-1 dev data point recorded; (2) fresh split-bracket profiles + gate verdict with arithmetic; (3) every gate outcome recorded, no lever shipped without its gate; (4) closing pp vs llama best-of on both boxes — **pp ≥ 1.0x on both → v1 pp criterion MET**, else the STOP finding naming where the residual prefill gap lives (from the fresh profile's non-matmul shares). Record tg as context, never the gate.

- [ ] **Step 3: (Lever-2 path only) AGENTS.md note**

One entry in the style of the existing kernel notes: the VNNI gemm exists, selected at v4+VNNI runtime, tested via `mise run test-vnni` (SDE) off-AVX-512.

- [ ] **Step 4: Commit + PR**

```bash
mise run lint
git add docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md docs/superpowers/specs/2026-07-17-m4b13-prefill-gemm-register-tiles-vnni-design.md AGENTS.md
git commit -m "specs: M4b.13 closing benches + exit-criteria walk"
```

Open the closing PR per the repo's convention.
