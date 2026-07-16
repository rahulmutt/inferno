# M4b.11 — Decode Attention Attribution + Head-Sharded Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take M4b.2's open decode attribution fork against the de-throttled (post-M4b.10) baseline, and — only if the pre-registered gates authorize it — parallelize decode attention by sharding heads across the pool.

**Architecture:** This is a **data-gated** milestone. Task 1 builds the attribution measurement surface (a new quiet-hw gate capturing decode `--profile` tables at t=1 and best-t). Task 2 takes one quiet-hw session per machine (16c primary, 8c check), which also records the deferred uncapped tg re-bench. Task 3 applies the pre-registered gates (P1/P2 formulas, M4b.6 STOP thresholds). Tasks 4–7 implement **Lever 1 only if Gate 1 authorized**: head-span attention kernels (Task 4), a new pool dispatcher `inferno_par_attention_heads` (Task 5), the one-call decode codegen reroute + `HOST_ABI_VERSION` bump (Task 6), and the recorded data point (Task 7). Task 8 hands **Lever 2 (F16 KV)**, if Gate 2 authorized, to its own plan (the M4b.6 restructure-plan precedent). Task 9 records the closing data point. **No task before Task 3 may assume either gate's outcome.**

**Tech Stack:** Rust (workspace crates `inferno-kernels`, `inferno-pool`, `inferno-codegen`, `inferno-core`), bash (`scripts/quiet-hw/`), mise tasks, PhoenixNAP bare metal via `mise run metal`.

**Spec:** [`docs/superpowers/specs/2026-07-16-m4b11-decode-attention-f16kv-design.md`](../specs/2026-07-16-m4b11-decode-attention-f16kv-design.md) (committed `479b007`).

## Global Constraints

Copied verbatim from the spec. Every task's requirements implicitly include these.

- **Decode only.** Prefill (`inferno_par_gemm`, `inferno_par_attention`, `inferno_par_token_loop`) is untouched; the M4b.1 prefill gate is MET and closed.
- **Attribution-first, gated.** Levers ship only if their pre-registered gate fires (Task 3). A clean STOP with a recorded finding is a successful outcome.
- **No tolerance edits in Lever 1.** `cargo test -p inferno-codegen --test differential` and `cargo test -p inferno-core --test artifact` must pass **with existing bounds** — Lever 1 is numerics-free by construction (each head's math is unchanged and computed entirely within one lane).
- **Bit-identity bars:** hspan-vs-whole-call exact bit-identity; scalar↔AVX2 bit-identity extends to hspan; bit-identity across thread counts extends to head-sharded decode attention.
- **No pos-/size-threshold heuristics** in the decode dispatch. M4b.10 buried one tuning constant; do not plant another.
- **`INFERNO_DECODE_THREADS` bounds the new dispatcher's lanes** (via the existing `decode_threads` pool cap), the only override.
- **`HOST_ABI_VERSION` bumps only in Task 6** ("6" → "7"), together with the codegen reroute — never earlier, never separately.
- **Never edit a recorded data point.** Session output is pasted verbatim into spec Amendments.
- **Scripts never write to `docs/`** — verdicts are pasted in by a human (`docs/runbooks/quiet-hw-verification.md`).
- **Workflows are mise tasks:** `mise run test` / `lint` / `metal`. Run `mise run lint` before every push (CI runs clippy `-D warnings`; `mise run test` alone skips it).
- Perf numbers come only from quiet bare metal (`mise run metal`); no CI perf gates.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/quiet-hw/gate-decode-attr.sh` (create, Task 1) | Attribution gate: decode `--profile` at t=1 and t=phys-cores; prints both tables + parsed attention-share summary. |
| `scripts/quiet-hw/verify.sh` (modify, Task 1) | Run the new gate; add it to the summary table and pass check. |
| `docs/runbooks/quiet-hw-verification.md` (modify, Task 1) | Gate-table row → verdict destination (M4b.11 spec §Amendments). |
| `crates/inferno-kernels/src/attention.rs` (modify, Task 4) | Head-span cores (`h0..h1` loop bounds) + `inferno_attention_f32_{scalar,avx2}_hspan` symbols; whole-call symbols delegate. |
| `crates/inferno-kernels/tests/rig.rs` (modify, Task 4) | hspan-vs-whole-call and scalar↔AVX2-hspan bit-identity tests. |
| `crates/inferno-pool/src/pool.rs` (modify, Task 5) | `AttnHspanFn`, `AttnHeadsJob`, `JobKind::AttnHeads`, `run_attn_heads_span`, `Pool::par_attention_heads` + unit tests. |
| `crates/inferno-pool/src/lib.rs` (modify, Task 5) | `inferno_par_attention_heads` host dispatcher + re-exports. |
| `crates/inferno-pool/tests/par_attention_heads_fallback.rs` (create, Task 5) | Pool-absent serial fallback integration test. |
| `crates/inferno-codegen/src/loopir.rs` (modify, Task 6) | `attention_hspan_symbol`. |
| `crates/inferno-codegen/src/llvm/mod.rs` (modify, Task 6) | Declare hspan kernels + `inferno_par_attention_heads`; IR-contains test. |
| `crates/inferno-codegen/src/llvm/ops.rs` (modify, Task 6) | `lower_attention` reroute: dispatcher call replaces direct kernel call. |
| `crates/inferno-codegen/src/lib.rs` (modify, Task 6) | `HOST_ABI_VERSION` "6" → "7". |
| `crates/inferno-core/src/artifact.rs` (modify, Task 6) | Symbol retention for the new hspan kernels + dispatcher. |
| `AGENTS.md` (modify, Task 7) | Decode-threading bullet gains the head-sharded attention sentence. |

---

### Task 1: `gate-decode-attr.sh` — the attribution measurement surface

The gates consume two numbers per machine: **S** (attention's share of decode wall at best-t) and attention's **in-situ GB/s**. This gate captures the raw `--profile` tables the controller computes them from. Best-t is the protocol's operating point: `--threads 0` = physical cores, so the gate profiles at `t=1` and `t=$(phys_cores)`.

**Files:**
- Create: `scripts/quiet-hw/gate-decode-attr.sh`
- Modify: `scripts/quiet-hw/verify.sh` (gate list, summary loop, pass check)
- Modify: `docs/runbooks/quiet-hw-verification.md` (verdict-destination table)

**Interfaces:**
- Consumes: `lib.sh` helpers `smoke_header`, `machine_block`, `phys_cores` (all existing); `inferno run --prompt <s> --max-tokens <n> --threads <t> --profile` (existing CLI).
- Produces: `$QHW_OUT/profile-t1.txt`, `$QHW_OUT/profile-t<best>.txt`, and a `gate-decode-attr.out` tee from `verify.sh` — the raw record for the Task 3 arithmetic.

- [ ] **Step 1: Write the gate script**

Create `scripts/quiet-hw/gate-decode-attr.sh`:

```bash
#!/usr/bin/env bash
# M4b.11 attribution gate — the decode profiles the pre-registered gates
# consume: `inferno run --profile` at t=1 (comparable to the M4b.2
# 2026-07-07 baseline) and at t=<physical cores> (the operating point,
# exposing the serial-attention Amdahl share). Prints both profile outputs
# verbatim plus a parsed attention-share convenience summary. VERDICTS ARE
# HUMAN: paste the output into the M4b.11 spec §Amendments and compute
# S, in-situ GB/s, P1, P2 there per the spec's pre-registered formulas
# (docs/runbooks/quiet-hw-verification.md).
# Usage: gate-decode-attr.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-decode-attr.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PROMPT_BYTES=256; MAXTOK=8; else PROMPT_BYTES=2048; MAXTOK=64; fi

smoke_header "gate-decode-attr (M4b.11 attribution: decode profile t=1 + best-t)"
machine_block
echo

# The M4b.2 profile protocol: random base64 prompt (~1.3K tokens at 2048
# bytes), 64 generated tokens. --profile compiles a distinct (profiled)
# cache entry on first use; that build time is not part of any measurement.
PROMPT="$(head -c "$PROMPT_BYTES" /dev/urandom | base64 | tr -d '\n')"
TBEST="$(phys_cores)"

for T in 1 "$TBEST"; do
  echo "--- profile at --threads $T ---"
  cargo run --release -q -p inferno -- run "$MODEL" \
    --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads "$T" --profile \
    > "$OUT/profile-t$T.txt" 2>&1
  # The generated text is noise; show only the profile tables.
  sed -n '/^profile \[/,$p' "$OUT/profile-t$T.txt"
  echo
done

# Convenience summary: the decode-table attention share per thread count.
# The gate arithmetic (S, S', P1, P2) is controller work in the spec.
echo "attention decode share (parsed):"
for T in 1 "$TBEST"; do
  share=$(sed -n '/^profile \[decode\]/,$p' "$OUT/profile-t$T.txt" \
    | awk '$1 == "attention" { print $3; exit }')
  echo "  t=$T: ${share:-NOT-FOUND}"
done
```

Then: `chmod +x scripts/quiet-hw/gate-decode-attr.sh`

- [ ] **Step 2: Smoke-run it locally to verify the plumbing**

Run (devpod is fine for plumbing — smoke output is never recordable):

```bash
QHW_SMOKE=1 bash scripts/quiet-hw/gate-decode-attr.sh models/qwen2.5-0.5b-instruct-q8_0.gguf
```

(Fetch the model first with `bash scripts/fetch-qwen-gguf.sh` if absent — check how other gates obtain it in `docs/runbooks/quiet-hw-verification.md`.)

Expected: the SMOKE header, machine block, two `--- profile at --threads N ---` sections each ending in `profile [prefill]` + `profile [decode]` tables, and a final `attention decode share (parsed):` block with two non-`NOT-FOUND` percentage values. If the parse prints `NOT-FOUND`, fix the `sed`/`awk` extraction against the actual table format before proceeding.

- [ ] **Step 3: Wire the gate into `verify.sh`**

In `scripts/quiet-hw/verify.sh`, add after the `run_gate bench-protocol` line:

```bash
run_gate decode-attr     bash "$HERE/gate-decode-attr.sh" "$MODEL"
```

and add `decode-attr` to **both** loops that currently enumerate the gates:

```bash
  for g in prefill-scaling decode-cap bw-curve pf-dist bench-protocol decode-attr intel-ab; do
```

```bash
for g in prefill-scaling decode-cap bw-curve pf-dist bench-protocol decode-attr; do
```

- [ ] **Step 4: Add the runbook verdict row**

In `docs/runbooks/quiet-hw-verification.md`, append to the verdict-destination table (after the `gate-intel-ab.out` row, matching its format):

```markdown
| `gate-decode-attr.out` | [M4b.11 spec](../superpowers/specs/2026-07-16-m4b11-decode-attention-f16kv-design.md) §Amendments | S (attention decode share, t=1 + best-t); in-situ GB/s; Gate 1 (P1) and Gate 2 (P2) arithmetic and verdicts |
```

- [ ] **Step 5: Smoke-run the full verify pass**

Run: `bash scripts/quiet-hw/verify.sh models/qwen2.5-0.5b-instruct-q8_0.gguf --smoke`
Expected: summary table includes a `| decode-attr | PASS |` row (preflight will be UNFIT on a devpod; `--smoke` continues past it).

- [ ] **Step 6: Commit**

```bash
git add scripts/quiet-hw/gate-decode-attr.sh scripts/quiet-hw/verify.sh docs/runbooks/quiet-hw-verification.md
git commit -m "quiet-hw: gate-decode-attr — M4b.11 attribution profiles (t=1 + best-t)"
```

---

### Task 2: Attribution sessions (manual, quiet hardware — one per machine)

This task is a **manual protocol run**, not code. It produces the milestone's baseline and everything Task 3 consumes. Follow `docs/runbooks/metal.md` (provisioning) and `docs/runbooks/quiet-hw-verification.md` (session discipline) exactly.

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-m4b11-decode-attention-f16kv-design.md` (§Amendments — session records)
- Modify: `docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md` (§Amendments — the uncapped tg re-bench, cross-referenced)

**Interfaces:**
- Consumes: Task 1's gate, `mise run metal`, `scripts/quiet-hw/verify.sh`.
- Produces: recorded amendments — per machine: the uncapped tg re-bench (`gate-bench-protocol.out`), the t=1 and best-t decode profiles (`gate-decode-attr.out`), and that machine's `t_best` (= physical cores used by the protocol).

- [ ] **Step 1: Session on the 16c primary (`d2.c1.medium`)**

Provision via `mise run metal` per `docs/runbooks/metal.md`. On the box, inside the devenv shell: run the smoke pass first (`verify.sh <model> --smoke` — plumbing only), then the real pass (`verify.sh <model>`). Preflight UNFIT is a hard stop — reschedule, don't override, per the runbook.

- [ ] **Step 2: Record the 16c amendments**

Paste verbatim, per the runbook's destination table: `gate-bench-protocol.out` → M4a spec §Amendments (this is the deferred **uncapped tg re-bench** — note it as such, referencing M4a's 2026-07-15 forward note); `gate-decode-attr.out` → M4b.11 spec §Amendments. Date-stamped headings, never edit once recorded. Commit:

```bash
git add docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md docs/superpowers/specs/2026-07-16-m4b11-decode-attention-f16kv-design.md
git commit -m "specs: M4b.11 attribution session A (d2.c1.medium, 16c) — uncapped tg re-bench + decode profiles"
```

- [ ] **Step 3: Session on the 8c check (`s2.c2.medium`) — same protocol**

Repeat Steps 1–2 on `s2.c2.medium`; commit as session B.

```bash
git commit -m "specs: M4b.11 attribution session B (s2.c2.medium, 8c) — uncapped tg re-bench + decode profiles"
```

---

### Task 3: Gate verdicts — the pre-registered arithmetic

Controller work: compute the gates from the recorded sessions, exactly as the spec pre-registers them. **Do not look at the implementation tasks while doing this; the formulas were fixed before the data.**

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-m4b11-decode-attention-f16kv-design.md` (§Amendments — verdict record)

**Interfaces:**
- Consumes: Task 2's recorded amendments; each machine's recorded M4b.10 bandwidth curve (`gate-bw-curve` amendments in the M4b.10 spec) for `BW_ceiling`.
- Produces: the Gate 1 / Gate 2 verdicts that decide whether Tasks 4–7 and Task 8 run.

- [ ] **Step 1: Compute S and in-situ GB/s per machine**

From each machine's best-t decode profile table: `S` = the attention row's share. In-situ GB/s = unique KV bytes streamed / attention wall seconds, where attention wall = decode wall × S, and for decode steps at positions `p0 .. p0+T-1` (p0 = prompt length, T = generated tokens), per layer per token unique bytes = `2 × (p+1) × kv_dim × 4` (K + V, f32); multiply by `n_layers` and sum over tokens. Get `kv_dim` and `n_layers` from `inferno inspect <model>`. Show every number in the amendment.

- [ ] **Step 2: Compute P1 and P2 per machine and apply the thresholds**

`P1 = S × (1 − 1/min(t_best, 14))`. `P2 = S′ × ½ × min(1, BW_insitu / BW_ceiling)` with `S′ = S / min(t_best, 14)` if Gate 1 authorizes Lever 1 (both-machines decision, each machine's own numbers), else `S′ = S`. Thresholds per lever: **≥5% on both machines → authorized; <3% on both → STOP (record the finding, close the lever as a diagnostic); anything between or split → controller judgment call, recorded either way.**

- [ ] **Step 3: Record the verdict amendment and commit**

The amendment shows: the parsed S values, the byte arithmetic, `BW_ceiling` provenance (which M4b.10 amendment), P1/P2 per machine, each gate's verdict, and — per the spec — the **headroom-set tg target** for the closing data point (state the arithmetic: baseline tg × (1 + authorized levers' combined projected reduction), with the conservatism stated explicitly).

```bash
git add docs/superpowers/specs/2026-07-16-m4b11-decode-attention-f16kv-design.md
git commit -m "specs: M4b.11 gate verdicts — P1/P2 arithmetic and lever authorization"
```

**⛔ GATE: If Gate 1 = STOP, skip Tasks 4–7. If Gate 2 = STOP, skip Task 8. If both STOP, go straight to Task 9 (which then only re-records context, closing the milestone as a diagnostic).**

---

### Task 4 (GATED on Gate 1): head-span attention kernels

The kernel's per-head loop is already independent per head; this task makes the loop bounds parameters. Whole-call symbols delegate to the same core with `(0, n_heads)`, so nothing forks and bit-identity is by construction.

**Files:**
- Modify: `crates/inferno-kernels/src/attention.rs`
- Test: `crates/inferno-kernels/tests/rig.rs` (append)

**Interfaces:**
- Consumes: existing `attn_core_scalar`, `inferno_attention_f32_avx2` body, `expf` helpers.
- Produces: `inferno_attention_f32_scalar_hspan` and `inferno_attention_f32_avx2_hspan`, C-ABI: `(out: *mut f32, q: *const f32, kv: *mut f32, scores: *mut f32, kv_base: usize, v_off: usize, pos: usize, kv_dim: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, h_start: usize, h_end: usize)` — the whole-call ABI plus the head range. `n_heads` stays the FULL head count (the GQA group divisor); only the loop range narrows.

- [ ] **Step 1: Write the failing bit-identity tests**

Append to `crates/inferno-kernels/tests/rig.rs`:

```rust
mod attention_hspan {
    //! M4b.11: the head-span kernels must be bitwise-identical to the
    //! whole-call kernels — per head, under any tiling of the head range —
    //! and scalar↔AVX2 bit-identity must extend to hspan.
    use inferno_kernels::{inferno_attention_f32_scalar, inferno_attention_f32_scalar_hspan};
    #[cfg(target_arch = "x86_64")]
    use inferno_kernels::{inferno_attention_f32_avx2, inferno_attention_f32_avx2_hspan};

    struct Case {
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        pos: usize,
    }

    // Bench-model GQA shape (14/2), MHA, and a small odd-group shape.
    const CASES: &[Case] = &[
        Case { n_heads: 14, n_kv_heads: 2, head_dim: 8, pos: 0 },
        Case { n_heads: 14, n_kv_heads: 2, head_dim: 8, pos: 37 },
        Case { n_heads: 8, n_kv_heads: 8, head_dim: 16, pos: 100 },
        Case { n_heads: 6, n_kv_heads: 3, head_dim: 8, pos: 5 },
    ];

    fn lcg_fill(mut seed: u64, buf: &mut [f32]) {
        for v in buf.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((seed >> 40) as f32 / (1u64 << 23) as f32) - 1.0;
        }
    }

    /// Head-range tilings to check: whole, per-head, and a ragged 3-way
    /// split that crosses GQA group boundaries for every CASES shape.
    fn tilings(n: usize) -> Vec<Vec<(usize, usize)>> {
        vec![
            vec![(0, n)],
            (0..n).map(|h| (h, h + 1)).collect(),
            vec![(0, 1), (1, 5.min(n - 1)), (5.min(n - 1), n)],
        ]
    }

    fn buffers(c: &Case) -> (Vec<f32>, Vec<f32>, usize, usize) {
        let kv_dim = c.n_kv_heads * c.head_dim;
        let seq = c.pos + 1;
        let v_off = seq * kv_dim;
        let mut q = vec![0f32; c.n_heads * c.head_dim];
        let mut kv = vec![0f32; 2 * v_off];
        lcg_fill(0x5eed_0001, &mut q);
        lcg_fill(0x5eed_0002, &mut kv);
        (q, kv, kv_dim, v_off)
    }

    fn whole_scalar(c: &Case) -> Vec<f32> {
        let (q, mut kv, kv_dim, v_off) = buffers(c);
        let mut out = vec![0f32; c.n_heads * c.head_dim];
        let mut scores = vec![0f32; c.pos + 1];
        // SAFETY: buffers sized per the AttnFn contract above.
        unsafe {
            inferno_attention_f32_scalar(
                out.as_mut_ptr(), q.as_ptr(), kv.as_mut_ptr(), scores.as_mut_ptr(),
                0, v_off, c.pos, kv_dim, c.n_heads, c.n_kv_heads, c.head_dim,
            );
        }
        out
    }

    fn hspan_scalar(c: &Case, spans: &[(usize, usize)]) -> Vec<f32> {
        let (q, mut kv, kv_dim, v_off) = buffers(c);
        let mut out = vec![0f32; c.n_heads * c.head_dim];
        for &(h0, h1) in spans {
            // Fresh scratch per span, mimicking lane-local scratch.
            let mut scores = vec![0f32; c.pos + 1];
            // SAFETY: buffers sized per the hspan contract; spans tile 0..n_heads.
            unsafe {
                inferno_attention_f32_scalar_hspan(
                    out.as_mut_ptr(), q.as_ptr(), kv.as_mut_ptr(), scores.as_mut_ptr(),
                    0, v_off, c.pos, kv_dim, c.n_heads, c.n_kv_heads, c.head_dim, h0, h1,
                );
            }
        }
        out
    }

    fn assert_bits_eq(a: &[f32], b: &[f32], ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: element {i}: {x} vs {y}");
        }
    }

    #[test]
    fn hspan_scalar_bitwise_matches_whole_call_under_any_tiling() {
        for (ci, c) in CASES.iter().enumerate() {
            let whole = whole_scalar(c);
            for (ti, spans) in tilings(c.n_heads).iter().enumerate() {
                let tiled = hspan_scalar(c, spans);
                assert_bits_eq(&whole, &tiled, &format!("case {ci} tiling {ti}"));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn whole_avx2(c: &Case) -> Vec<f32> {
        let (q, mut kv, kv_dim, v_off) = buffers(c);
        let mut out = vec![0f32; c.n_heads * c.head_dim];
        let mut scores = vec![0f32; c.pos + 1];
        // SAFETY: contract as scalar, plus the avx2 guard in the test below.
        unsafe {
            inferno_attention_f32_avx2(
                out.as_mut_ptr(), q.as_ptr(), kv.as_mut_ptr(), scores.as_mut_ptr(),
                0, v_off, c.pos, kv_dim, c.n_heads, c.n_kv_heads, c.head_dim,
            );
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    fn hspan_avx2(c: &Case, spans: &[(usize, usize)]) -> Vec<f32> {
        let (q, mut kv, kv_dim, v_off) = buffers(c);
        let mut out = vec![0f32; c.n_heads * c.head_dim];
        for &(h0, h1) in spans {
            let mut scores = vec![0f32; c.pos + 1];
            // SAFETY: contract as scalar hspan, plus the avx2 guard below.
            unsafe {
                inferno_attention_f32_avx2_hspan(
                    out.as_mut_ptr(), q.as_ptr(), kv.as_mut_ptr(), scores.as_mut_ptr(),
                    0, v_off, c.pos, kv_dim, c.n_heads, c.n_kv_heads, c.head_dim, h0, h1,
                );
            }
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn hspan_avx2_bitwise_matches_whole_call_and_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            eprintln!("skipping: no avx2");
            return;
        }
        for (ci, c) in CASES.iter().enumerate() {
            let whole = whole_avx2(c);
            // scalar↔AVX2 bit-identity (M4b.3) must extend to hspan.
            assert_bits_eq(&whole, &whole_scalar(c), &format!("case {ci} isa"));
            for (ti, spans) in tilings(c.n_heads).iter().enumerate() {
                let tiled = hspan_avx2(c, spans);
                assert_bits_eq(&whole, &tiled, &format!("case {ci} avx2 tiling {ti}"));
            }
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p inferno-kernels --test rig attention_hspan`
Expected: COMPILE ERROR — `inferno_attention_f32_scalar_hspan` not found (the symbols don't exist yet).

- [ ] **Step 3: Implement the hspan kernels**

In `crates/inferno-kernels/src/attention.rs`:

(a) Extend the module doc comment's last sentence: `One call = one query token.` → `One call = one query token; the *_hspan variants (M4b.11) run the same per-head math over a caller-chosen head range for the pool's decode head-sharding.`

(b) Give `attn_core_scalar` head-range bounds — change its signature and loop head:

```rust
#[allow(clippy::too_many_arguments)]
fn attn_core_scalar(
    out: &mut [f32],
    q: &[f32],
    kv: &[f32],
    scores: &mut [f32],
    kv_base: usize,
    v_off: usize,
    pos: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    h_start: usize,
    h_end: usize,
) {
```

and `for h in 0..n_heads {` → `for h in h_start..h_end {`. (`group = n_heads / n_kv_heads` stays computed from the full `n_heads`.) Nothing else in the body changes.

(c) The existing whole-call scalar wrapper passes the full range — its inner call becomes:

```rust
        attn_core_scalar(
            out, q, kv, scores, kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim, 0,
            n_heads,
        );
```

(d) Add the scalar hspan symbol after the whole-call wrapper:

```rust
/// Head-span variant (M4b.11): identical per-head math to
/// [`inferno_attention_f32_scalar`] restricted to heads `[h_start, h_end)`,
/// so any tiling of `0..n_heads` reproduces the whole call bit-for-bit.
/// `n_heads` stays the FULL head count (the GQA group divisor).
///
/// # Safety
/// As [`inferno_attention_f32_scalar`], plus `h_start <= h_end <= n_heads`.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_attention_f32_scalar_hspan(
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    scores: *mut f32,
    kv_base: usize,
    v_off: usize,
    pos: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    h_start: usize,
    h_end: usize,
) {
    // SAFETY: contract above; same slice bounds as the whole-call wrapper.
    unsafe {
        let q = std::slice::from_raw_parts(q, n_heads * head_dim);
        let out = std::slice::from_raw_parts_mut(out, n_heads * head_dim);
        let scores = std::slice::from_raw_parts_mut(scores, pos + 1);
        let kv = std::slice::from_raw_parts(kv, kv_base + 2 * v_off);
        attn_core_scalar(
            out, q, kv, scores, kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim,
            h_start, h_end,
        );
    }
}
```

(e) AVX2: move the entire body of `inferno_attention_f32_avx2` into a range-parameterized core, and make both symbols call it:

```rust
/// # Safety
/// As [`inferno_attention_f32_scalar`], plus the running CPU has AVX2+FMA.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_attention_f32_avx2(
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    scores: *mut f32,
    kv_base: usize,
    v_off: usize,
    pos: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    // SAFETY: forwarding the contract for the full head range.
    unsafe {
        attn_core_avx2(
            out, q, kv, scores, kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim, 0,
            n_heads,
        );
    }
}

/// Head-span variant (M4b.11); see [`inferno_attention_f32_scalar_hspan`].
///
/// # Safety
/// As [`inferno_attention_f32_avx2`], plus `h_start <= h_end <= n_heads`.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_attention_f32_avx2_hspan(
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    scores: *mut f32,
    kv_base: usize,
    v_off: usize,
    pos: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    h_start: usize,
    h_end: usize,
) {
    // SAFETY: forwarding the contract for the caller's head range.
    unsafe {
        attn_core_avx2(
            out, q, kv, scores, kv_base, v_off, pos, kv_dim, n_heads, n_kv_heads, head_dim,
            h_start, h_end,
        );
    }
}

/// The AVX2 per-head loop, bounds-parameterized (M4b.11). Body is the
/// former `inferno_attention_f32_avx2` verbatim except `for h in
/// h_start..h_end`.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn attn_core_avx2(
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    scores: *mut f32,
    kv_base: usize,
    v_off: usize,
    pos: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    h_start: usize,
    h_end: usize,
) {
    // SAFETY: contract as the public symbols; head_dim is a mult of 8.
    unsafe {
        // ... the former inferno_attention_f32_avx2 body, verbatim,
        // with the loop head changed:  for h in h_start..h_end {
    }
}
```

(In (e), move the existing body — do not retype it. The ONLY body change is the loop head.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p inferno-kernels --test rig`
Expected: PASS — the new `attention_hspan` tests and every pre-existing rig test (the whole-call outputs must be untouched by the refactor).

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/src/attention.rs crates/inferno-kernels/tests/rig.rs
git commit -m "kernels: head-span attention variants (M4b.11) — bit-identical under any head tiling"
```

---

### Task 5 (GATED on Gate 1): pool dispatcher `inferno_par_attention_heads`

The fourth sibling dispatcher. The pool stays kernel-agnostic: the hspan kernel arrives as a fn pointer from generated code. Sharding is align-1 over `0..n_heads`, lane budget `min(active_threads, decode_threads)` — decode work, so the `INFERNO_DECODE_THREADS` override applies exactly like `par_gemv`.

**Files:**
- Modify: `crates/inferno-pool/src/pool.rs`
- Modify: `crates/inferno-pool/src/lib.rs`
- Test: `crates/inferno-pool/src/pool.rs` (unit tests), `crates/inferno-pool/tests/par_attention_heads_fallback.rs` (create)

**Interfaces:**
- Consumes: existing `shard_table_aligned`, `JobKind`/epoch machinery, `DISPATCH_CLAIMED`, `GLOBAL`.
- Produces: `pub type AttnHspanFn` (13-arg C ABI matching Task 4's symbols), `pub struct AttnHeadsJob { kernel: AttnHspanFn, out: *mut f32, q: *const f32, kv: *mut f32, pos: usize, kv_base: usize, v_off: usize, kv_dim: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize }`, `pub unsafe fn Pool::par_attention_heads(&self, job: &AttnHeadsJob)`, and the host symbol `inferno_par_attention_heads(kernel, out, q, kv, pos, kv_base, v_off, kv_dim, n_heads, n_kv_heads, head_dim)` — Task 6 declares and calls it by this exact name and argument order.

- [ ] **Step 1: Write the failing unit tests**

Append inside `mod tests` in `crates/inferno-pool/src/pool.rs` (after the token-loop tests):

```rust
    /// Fake head-span attention kernel with the real M4b.11 ABI:
    /// deterministic function of (h, d, pos), writes only its span's rows,
    /// touches scores[pos] to prove each lane's scratch covers pos + 1.
    unsafe extern "C" fn stamp_attn_heads(
        out: *mut f32,
        q: *const f32,
        _kv: *mut f32,
        scores: *mut f32,
        _kv_base: usize,
        _v_off: usize,
        pos: usize,
        _kv_dim: usize,
        _n_heads: usize,
        _n_kv_heads: usize,
        head_dim: usize,
        h_start: usize,
        h_end: usize,
    ) {
        // SAFETY: run_attn_heads_span sizes scores to pos + 1.
        unsafe { *scores.add(pos) = pos as f32 };
        for h in h_start..h_end {
            for d in 0..head_dim {
                let i = h * head_dim + d;
                // SAFETY: out/q rows are n_heads*head_dim per the contract.
                unsafe { *out.add(i) = *q.add(i) + (h * 31 + d + pos) as f32 };
            }
        }
    }

    const AH_NH: usize = 14; // bench-model head count
    const AH_HD: usize = 4;

    fn attn_heads_dispatch(pool: &Pool, pos: usize) -> Vec<f32> {
        let q: Vec<f32> = (0..AH_NH * AH_HD).map(|i| i as f32).collect();
        let mut out = vec![f32::NAN; AH_NH * AH_HD];
        let mut kv = [0f32; 1];
        let job = AttnHeadsJob {
            kernel: stamp_attn_heads,
            out: out.as_mut_ptr(),
            q: q.as_ptr(),
            kv: kv.as_mut_ptr(),
            pos,
            kv_base: 0,
            v_off: 0,
            kv_dim: 0,
            n_heads: AH_NH,
            n_kv_heads: 2,
            head_dim: AH_HD,
        };
        // SAFETY: buffers sized per stamp_attn_heads' expectations.
        unsafe { pool.par_attention_heads(&job) };
        out
    }

    fn attn_heads_expected(pos: usize) -> Vec<f32> {
        (0..AH_NH * AH_HD)
            .map(|i| {
                let (h, d) = (i / AH_HD, i % AH_HD);
                i as f32 + (h * 31 + d + pos) as f32
            })
            .collect()
    }

    #[test]
    fn attn_heads_matches_serial_expectation_across_pool_sizes() {
        for threads in [1, 2, 4, 8, 16] {
            let pool = Pool::new(threads);
            for pos in [0, 9, 100] {
                assert_eq!(
                    attn_heads_dispatch(&pool, pos),
                    attn_heads_expected(pos),
                    "threads={threads} pos={pos}"
                );
            }
        }
    }

    #[test]
    fn attn_heads_respects_decode_cap_without_changing_result() {
        // Decode work: min(active, decode_threads) lanes, like par_gemv.
        let pool = Pool::new(8);
        pool.set_decode_threads(2);
        assert_eq!(attn_heads_dispatch(&pool, 5), attn_heads_expected(5));
        pool.set_decode_threads(1);
        assert_eq!(attn_heads_dispatch(&pool, 5), attn_heads_expected(5));
    }

    #[test]
    fn attn_heads_threads_exceeding_heads_collapses() {
        let pool = Pool::new(16); // 16 lanes > 14 heads
        assert_eq!(attn_heads_dispatch(&pool, 3), attn_heads_expected(3));
    }
```

- [ ] **Step 2: Write the failing fallback integration test**

Create `crates/inferno-pool/tests/par_attention_heads_fallback.rs`:

```rust
//! `inferno_par_attention_heads` without an initialized global pool: the
//! entry point must degrade to one serial full-range hspan call — this
//! file never calls `init_global`, and an integration test binary is its
//! own process, so the pool is guaranteed absent.
#![allow(unsafe_code)] // FFI entry-point tests; same justification as the sibling test files.

use inferno_pool::inferno_par_attention_heads;

unsafe extern "C" fn stamp_attn_heads(
    out: *mut f32,
    q: *const f32,
    _kv: *mut f32,
    scores: *mut f32,
    _kv_base: usize,
    _v_off: usize,
    pos: usize,
    _kv_dim: usize,
    _n_heads: usize,
    _n_kv_heads: usize,
    head_dim: usize,
    h_start: usize,
    h_end: usize,
) {
    // SAFETY: the dispatcher sizes scores to pos + 1.
    unsafe { *scores.add(pos) = pos as f32 };
    for h in h_start..h_end {
        for d in 0..head_dim {
            let i = h * head_dim + d;
            // SAFETY: out/q rows are n_heads*head_dim per the contract.
            unsafe { *out.add(i) = *q.add(i) + (h * 31 + d + pos) as f32 };
        }
    }
}

const NH: usize = 14;
const HD: usize = 4;

fn dispatch(n_heads: usize, pos: usize) -> Vec<f32> {
    let q: Vec<f32> = (0..NH * HD).map(|i| i as f32).collect();
    let mut out = vec![f32::NAN; NH * HD];
    let mut kv = [0f32; 1];
    // SAFETY: buffers sized per stamp_attn_heads' expectations.
    unsafe {
        inferno_par_attention_heads(
            stamp_attn_heads,
            out.as_mut_ptr(),
            q.as_ptr(),
            kv.as_mut_ptr(),
            pos,
            0,
            0,
            0,
            n_heads,
            2,
            HD,
        );
    }
    out
}

fn expected(pos: usize) -> Vec<f32> {
    (0..NH * HD)
        .map(|i| {
            let (h, d) = (i / HD, i % HD);
            i as f32 + (h * 31 + d + pos) as f32
        })
        .collect()
}

#[test]
fn uninitialized_pool_falls_back_to_serial_full_range() {
    for pos in [0, 41] {
        assert_eq!(dispatch(NH, pos), expected(pos), "pos={pos}");
    }
}

#[test]
fn zero_heads_is_a_noop() {
    let out = dispatch(0, 0);
    assert!(out.iter().all(|v| v.is_nan()), "no head row may be written");
}
```

- [ ] **Step 3: Run both to verify they fail**

Run: `cargo test -p inferno-pool`
Expected: COMPILE ERROR — `AttnHeadsJob`, `par_attention_heads`, `inferno_par_attention_heads` not found.

- [ ] **Step 4: Implement the pool side**

In `crates/inferno-pool/src/pool.rs`:

(a) After the `AttnFn` type alias, add:

```rust
/// The M4b.11 head-span attention kernel ABI: [`AttnFn`] plus
/// `(h_start, h_end)`. Must match `inferno-kernels`'
/// `inferno_attention_f32_*_hspan` symbols exactly.
pub type AttnHspanFn = unsafe extern "C" fn(
    *mut f32,
    *const f32,
    *mut f32,
    *mut f32,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);
```

(b) After the `AttnJob` struct, add:

```rust
/// Per-dispatch invariants of one head-sharded decode-attention dispatch
/// (M4b.11): ONE query token at `pos`; the head index `h in 0..n_heads`
/// is the sharded axis. `n_heads` is the full head count — shards narrow
/// the kernel's loop range, never its GQA group divisor.
#[derive(Clone, Copy)]
pub struct AttnHeadsJob {
    pub kernel: AttnHspanFn,
    pub out: *mut f32,
    pub q: *const f32,
    pub kv: *mut f32,
    pub pos: usize,
    pub kv_base: usize,
    pub v_off: usize,
    pub kv_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}
```

(c) Add the `JobKind` variant and `run_shard` arm:

```rust
    AttnHeads(AttnHeadsJob),
```

```rust
        // SAFETY: forwarding the caller's contract for the disjoint head span.
        JobKind::AttnHeads(job) => unsafe { run_attn_heads_span(&job, start, end) },
```

(d) After `run_attn_span`, add:

```rust
/// Run heads `[start, end)` of one decode-attention dispatch: a single
/// head-span kernel call with a lane-local `scores` scratch (`pos + 1`
/// entries — same Vec-per-lane reasoning as [`run_attn_span`]). The
/// kernel computes each head exactly as the whole-call kernel does, so
/// sharding never changes output bits.
///
/// # Safety
/// The dispatcher's caller contract must cover heads `[start, end)`:
/// `out`/`q` valid for `n_heads * head_dim` f32 (out head rows disjoint
/// per shard), `kv` fully appended for positions `<= pos` and read-only
/// for the duration, `kernel` a valid `AttnHspanFn`.
pub(crate) unsafe fn run_attn_heads_span(j: &AttnHeadsJob, start: usize, end: usize) {
    let mut scores = vec![0f32; j.pos + 1];
    // SAFETY: forwarding the caller's contract for the head span.
    unsafe {
        (j.kernel)(
            j.out,
            j.q,
            j.kv,
            scores.as_mut_ptr(),
            j.kv_base,
            j.v_off,
            j.pos,
            j.kv_dim,
            j.n_heads,
            j.n_kv_heads,
            j.head_dim,
            start,
            end,
        );
    }
}
```

(e) Add `Pool::par_attention_heads` after `par_attention` (same publish/wake/join skeleton — copy `par_attention`'s body shape exactly, with these differences):

```rust
    /// Head-sharded decode attention (M4b.11): splits `0..job.n_heads`
    /// into align-1 contiguous shards across up to
    /// `min(active_threads(), decode_threads())` lanes — decode work, so
    /// the `INFERNO_DECODE_THREADS` override applies like `par_gemv`.
    /// Each head's out row is computed entirely by one lane with the
    /// per-head math unchanged, so thread count never changes output bits.
    ///
    /// # Safety
    /// As [`run_attn_heads_span`] over `0..job.n_heads`; calls must not
    /// overlap (one job at a time).
    pub unsafe fn par_attention_heads(&self, job: &AttnHeadsJob) {
        let n_heads = job.n_heads;
        if n_heads == 0 {
            return;
        }
        let active = self.active_threads().min(self.decode_threads());
        let shards = shard_table_aligned(n_heads, active, 1);
        if shards.len() == 1 {
            // SAFETY: caller contract covers the full head range.
            unsafe { run_attn_heads_span(job, 0, n_heads) };
            return;
        }
        let n_worker = shards.len() - 1;
        let (s0, e0) = shards[0];
        let kind = JobKind::AttnHeads(*job);
        // SAFETY (job write): the previous dispatch ended with
        // `remaining == 0`, and shardless workers never read `job`
        // (packed-epoch protocol) — no reader exists here.
        unsafe {
            *self.shared.job.get() = Job {
                kind: Some(kind),
                y: std::ptr::null_mut(),
                xq: std::ptr::null(),
                w: std::ptr::null(),
                k: 0,
                shards,
            };
        }
        self.shared.remaining.store(n_worker, Ordering::SeqCst);
        let counter =
            (self.shared.epoch.load(Ordering::SeqCst) >> PACKED_SHARD_BITS).wrapping_add(1);
        self.shared.epoch.store(
            (counter << PACKED_SHARD_BITS) | (n_worker + 1),
            Ordering::SeqCst,
        );
        // Wake exactly the workers that hold a shard (same handshake as
        // par_gemv; see there for the lost-wakeup argument).
        for slot in &self.shared.slots[..n_worker] {
            if slot.parked.load(Ordering::SeqCst) {
                slot.thread
                    .get()
                    .expect("worker registered in Pool::new")
                    .unpark();
            }
        }
        // SAFETY: caller contract; shard 0's heads are disjoint from
        // worker shards.
        unsafe { run_attn_heads_span(job, s0, e0) };
        let mut spins = 0u32;
        while self.shared.remaining.load(Ordering::Acquire) != 0 {
            if spins < SPIN_ITERS {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
```

In `crates/inferno-pool/src/lib.rs`:

(f) Extend the re-export: `pub use pool::{AttnFn, AttnHeadsJob, AttnHspanFn, AttnJob, GemmFn, GemvFn, Pool, TokenBodyFn};` and the module doc's first line: four dispatchers → `the five inferno_par_{gemv,gemm,attention,attention_heads,token_loop} dispatchers`.

(g) Add the host dispatcher after `inferno_par_attention`:

```rust
/// Host dispatcher for head-sharded decode attention (M4b.11): ONE query
/// token, the head range `0..n_heads` sharded align-1 across up to
/// `min(active_threads, decode_threads)` lanes (decode work — the
/// `INFERNO_DECODE_THREADS` override applies, like `inferno_par_gemv`).
/// Same single-dispatcher guard + serial fallback as its four siblings;
/// shares `DISPATCH_CLAIMED` deliberately — within one forward pass all
/// pool dispatches are issued serially and never overlap. On the CAS-loss
/// (or uninitialized-pool) path this runs one serial hspan call over the
/// full head range, bit-identical to the pooled path since each head is
/// computed by unchanged per-head math either way.
///
/// A panic inside the dispatcher or kernel aborts the process at this
/// `extern "C"` boundary — there is no unwind across FFI.
///
/// # Safety
/// Same contract as [`Pool::par_attention_heads`]; additionally `kernel`
/// must be a valid non-null function pointer with the M4b.11 head-span
/// attention ABI, and the KV cache must already contain every position
/// `<= pos` (decode codegen appends this token's k/v first). Generated
/// code guarantees all of this by construction (M3 trust model).
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_par_attention_heads(
    kernel: AttnHspanFn,
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    pos: usize,
    kv_base: usize,
    v_off: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    if n_heads == 0 {
        return;
    }
    let job = pool::AttnHeadsJob {
        kernel,
        out,
        q,
        kv,
        pos,
        kv_base,
        v_off,
        kv_dim,
        n_heads,
        n_kv_heads,
        head_dim,
    };
    match GLOBAL.get() {
        Some(p) => {
            if DISPATCH_CLAIMED
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: forwarding the caller's contract unchanged.
                unsafe { p.par_attention_heads(&job) };
                DISPATCH_CLAIMED.store(false, Ordering::Release);
            } else {
                // Lost the race for the pool's single dispatcher slot: run
                // serially over the full head range instead of overlapping
                // another thread's in-flight pool dispatch.
                // SAFETY: forwarding the caller's contract for the full range.
                unsafe { pool::run_attn_heads_span(&job, 0, n_heads) };
            }
        }
        // SAFETY: forwarding the caller's contract for the full range.
        None => unsafe { pool::run_attn_heads_span(&job, 0, n_heads) },
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p inferno-pool`
Expected: PASS — new unit tests, the fallback file, and every pre-existing pool test.

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-pool/src/pool.rs crates/inferno-pool/src/lib.rs crates/inferno-pool/tests/par_attention_heads_fallback.rs
git commit -m "pool: inferno_par_attention_heads — head-sharded decode attention dispatcher (M4b.11)"
```

---

### Task 6 (GATED on Gate 1): codegen reroute + `HOST_ABI_VERSION` bump

Decode `lower_attention` swaps its direct kernel call for one dispatcher call. This is the generated-code change: bump the ABI version with it, in the same commit, and retain the new symbols so `dlopen` resolves them.

**Files:**
- Modify: `crates/inferno-codegen/src/loopir.rs` (after `attention_symbol`, ~line 125)
- Modify: `crates/inferno-codegen/src/llvm/mod.rs` (`declare_kernels` + IR test ~line 380)
- Modify: `crates/inferno-codegen/src/llvm/ops.rs` (`lower_attention`, ~line 1516)
- Modify: `crates/inferno-codegen/src/lib.rs` (`HOST_ABI_VERSION`)
- Modify: `crates/inferno-core/src/artifact.rs` (`ensure_kernels_linked`)
- Modify: `crates/inferno-pool/src/lib.rs` (stale M4b.8 comment on `inferno_par_attention`)

**Interfaces:**
- Consumes: Task 4's `inferno_attention_f32_{scalar,avx2}_hspan` symbols (13-arg ABI); Task 5's `inferno_par_attention_heads` symbol (11 args: `kernel, out, q, kv, pos, kv_base, v_off, kv_dim, n_heads, n_kv_heads, head_dim`).
- Produces: generated decode code that calls `inferno_par_attention_heads`; `HOST_ABI_VERSION = "7"` (cache-key change → old artifacts recompile).

- [ ] **Step 1: Write the failing IR test**

In `crates/inferno-codegen/src/llvm/mod.rs`, find the existing test asserting `ir.contains("inferno_par_attention")` (~line 382) and add alongside it:

```rust
        assert!(ir.contains("inferno_par_attention_heads"));
```

Run: `cargo test -p inferno-codegen --lib`
Expected: FAIL — the emitted IR does not reference the dispatcher yet.

- [ ] **Step 2: Add the symbol helper in `loopir.rs`**

After `attention_symbol` (~line 125):

```rust
/// `inferno_attention_f32_{isa}_hspan`: the head-span variant (M4b.11),
/// selected identically to [`attention_symbol`]. Passed by pointer to
/// `inferno_par_attention_heads`; never called directly by generated code.
pub fn attention_hspan_symbol(isa: inferno_kernels::KernelIsa) -> String {
    format!("{}_hspan", attention_symbol(isa))
}
```

- [ ] **Step 3: Declare the new symbols in `declare_kernels`**

In `crates/inferno-codegen/src/llvm/mod.rs`, after the existing `inferno_attention_f32_{isa}` declaration loop, add:

```rust
        // void inferno_attention_f32_<isa>_hspan(ptr out, ptr q, ptr kv,
        //   ptr scores, i64 kv_base, i64 v_off, i64 pos, i64 kv_dim,
        //   i64 n_heads, i64 n_kv_heads, i64 head_dim, i64 h_start, i64 h_end)
        // — the M4b.11 head-span variant; passed as a function pointer to
        // inferno_par_attention_heads, never called directly.
        let attn_hspan_ty = void.fn_type(
            &[
                ptr.into(),
                ptr.into(),
                ptr.into(),
                ptr.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
            ],
            false,
        );
        for isa in ["scalar", "avx2"] {
            self.module.add_function(
                &format!("inferno_attention_f32_{isa}_hspan"),
                attn_hspan_ty,
                Some(Linkage::External),
            );
        }
```

and, after the `inferno_par_attention` declaration:

```rust
        // void inferno_par_attention_heads(ptr kernel, ptr out, ptr q,
        //   ptr kv, i64 pos, i64 kv_base, i64 v_off, i64 kv_dim,
        //   i64 n_heads, i64 n_kv_heads, i64 head_dim)
        // — the M4b.11 decode-attention dispatcher: one query token,
        // heads sharded across the pool (min(active, decode) lanes).
        let par_attn_heads_ty = void.fn_type(
            &[
                ptr.into(),
                ptr.into(),
                ptr.into(),
                ptr.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
            ],
            false,
        );
        self.module.add_function(
            "inferno_par_attention_heads",
            par_attn_heads_ty,
            Some(Linkage::External),
        );
```

- [ ] **Step 4: Reroute `lower_attention` in `ops.rs`**

In `lower_attention` (~line 1516), replace everything from the `--- Single-token attention read via the kernel (M4b.3). ... ---` comment through the closing `.unwrap();` of its `build_call` with:

```rust
        // --- Single-token attention read, head-sharded across the pool
        // (M4b.11): the hspan kernel goes in as a function pointer and the
        // dispatcher owns lane-local scores scratch (the entry alloca is
        // gone). The dispatcher's serial fallbacks keep the pool-absent
        // path equivalent to the former direct kernel call. ---
        let q_ptr = self.arena_row_ptr(frame, q);
        let out_ptr = self.arena_row_ptr(frame, out);
        let isa = self.module_isa();
        let sym = crate::loopir::attention_hspan_symbol(isa);
        let afn = self
            .module
            .get_function(&sym)
            .expect("hspan attention kernel declared");
        let pfn = self
            .module
            .get_function("inferno_par_attention_heads")
            .expect("decode attention dispatcher declared");
        let kv_dim_c = self.const_i64(kv_dim);
        let v_off = self.const_i64(seq_len * kv_dim);
        let kv_base_c = self.const_i64(kv_base);
        self.builder
            .build_call(
                pfn,
                &[
                    afn.as_global_value().as_pointer_value().into(),
                    out_ptr.into(),
                    q_ptr.into(),
                    frame.kv.into(),
                    frame.pos.into(),
                    kv_base_c.into(),
                    v_off.into(),
                    kv_dim_c.into(),
                    self.const_i64(n_heads as u64).into(),
                    self.const_i64(n_kv_heads as u64).into(),
                    self.const_i64(hd).into(),
                ],
                "par_attention_heads",
            )
            .unwrap();
```

(The removed lines include the `let scores = self.entry_alloca(...)` — the pool provides scratch now.) Also update `lower_attention`'s doc comment: `then reads:` → `then reads via `inferno_par_attention_heads` (M4b.11, heads sharded across the pool):`.

- [ ] **Step 5: Bump `HOST_ABI_VERSION`**

In `crates/inferno-codegen/src/lib.rs`, change the constant and prepend to its doc comment's version history:

```rust
/// "7" = M4b.11's head-sharded decode attention
/// (`inferno_par_attention_heads` dispatch + `inferno_attention_f32_*_hspan`
/// kernel symbols); "6" = M4b.9's `inferno_par_token_loop` dispatch; ...
pub const HOST_ABI_VERSION: &str = "7";
```

- [ ] **Step 6: Retain the new symbols in `inferno-core`**

In `crates/inferno-core/src/artifact.rs`, `ensure_kernels_linked`, add after the existing attention lines:

```rust
    p(inferno_kernels::inferno_attention_f32_scalar_hspan as *const ());
    p(inferno_kernels::inferno_attention_f32_avx2_hspan as *const ());
```

and after `p(inferno_pool::inferno_par_token_loop as *const ());`:

```rust
    p(inferno_pool::inferno_par_attention_heads as *const ());
```

- [ ] **Step 7: Retire the stale M4b.8 comment in the pool**

In `crates/inferno-pool/src/lib.rs`, `inferno_par_attention`'s doc comment still says decode codegen invokes the kernel directly. Update the parenthetical to: `(decode does not call this dispatcher — since M4b.11 its codegen calls inferno_par_attention_heads; the m <= 1 arm here covers T=1 prefill tiles)`.

- [ ] **Step 8: Run the correctness gates**

Run, in order:

```bash
cargo test -p inferno-codegen --lib          # IR test from Step 1 now passes
cargo test -p inferno-codegen --test differential
cargo test -p inferno-core --test artifact
mise run test
mise run lint
```

Expected: ALL PASS **with zero tolerance changes** (Lever 1 is numerics-free; any differential failure here is a bug in Tasks 4–6, never a tolerance problem). If insta snapshots diff, review with `cargo insta review` — only the attention call site may change; never blind-accept.

- [ ] **Step 9: Commit**

```bash
git add crates/inferno-codegen crates/inferno-core/src/artifact.rs crates/inferno-pool/src/lib.rs
git commit -m "codegen: route decode attention through inferno_par_attention_heads; HOST_ABI_VERSION 7 (M4b.11)"
```

---

### Task 7 (GATED on Gate 1): Lever-1 quiet-hw data point + docs

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-m4b11-decode-attention-f16kv-design.md` (§Amendments)
- Modify: `AGENTS.md` (decode-threading bullet)

**Interfaces:**
- Consumes: Tasks 4–6 landed on the branch; Task 2's baseline amendments (same machines).
- Produces: the recorded Lever-1 win/loss vs the same-session parent-commit baseline, and the short-context diagnostic.

- [ ] **Step 1: Quiet-hw A/B session on both machines**

Per machine (`d2.c1.medium`, then `s2.c2.medium`), per the metal runbook: build **both** the parent commit (pre-Task-4) and the lever commit; run the M4a protocol on each, interleaved in the same session (the `gate-intel-ab.sh` A/B pattern — within-session ratios, standing M4b discipline):

```bash
cargo run --release -q -p inferno -- bench <model> --pp 512 --tg 128 --reps 5 --threads 0 --json
```

Also record the **short-context diagnostic** on both builds (dispatch-overhead check near pos ≈ 0 — diagnostic only, never a gate, and never a reason to add a threshold heuristic):

```bash
cargo run --release -q -p inferno -- bench <model> --pp 16 --tg 32 --reps 5 --threads 0 --json
```

- [ ] **Step 2: Record the amendments**

Paste both machines' outputs verbatim into the M4b.11 spec §Amendments with the computed within-session tg ratios (lever/parent) against Gate 1's projected P1. State plainly whether the projection held. If tg regressed, the pre-registered mitigation is reverting the Task 6 call-site change — record the finding first, decide as a spec amendment.

- [ ] **Step 3: Update AGENTS.md**

In the `AGENTS.md` decode-threading bullet (the one beginning **"Decode threading is uncapped (M4b.10 …)"**), append the sentence:

```markdown
  Decode attention is head-sharded through `inferno_par_attention_heads`
  (M4b.11) under the same `INFERNO_DECODE_THREADS` bound; the head-span
  kernels must stay bit-identical to the whole-call kernels (the rig's
  hspan tiling tests are the guard).
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-07-16-m4b11-decode-attention-f16kv-design.md AGENTS.md
git commit -m "specs: M4b.11 Lever-1 data point (both machines) + AGENTS.md decode-attention note"
```

---

### Task 8 (GATED on Gate 2): F16 KV — hand off to its own plan

F16 KV spans the interpreter, codegen KV-append, kernel ABIs, and a tolerance re-derivation — the M4b.6 precedent (`2026-07-09-m4b6-reduce-unpack-restructure.md`) is that a post-gate lever of this size gets its own plan written against the recorded data.

- [ ] **Step 1: Write the Lever-2 implementation plan**

If Gate 2 authorized: invoke the writing-plans skill against the M4b.11 spec's Lever 2 section plus the recorded attribution amendments, producing `docs/superpowers/plans/YYYY-MM-DD-m4b11-f16-kv.md`. Its non-negotiables, restated from the spec so the plan inherits them: interpreter and compiled path switch **together**; append converts f32→f16 RNE (`vcvtps2ph`), reads widen losslessly (`vcvtph2ps`); scalar↔F16C conversion bit-identity in the rig; `HOST_ABI_VERSION` bump; `attn_rel_tol`/`logits_abs_tol` re-derived against observed distributions and documented in the spec — never loosened-to-green; the M3 "KV stays f32" note in AGENTS.md retires explicitly. If Lever 1 also landed, the f16 read path goes into the hspan core.

If Gate 2 = STOP: record the finding in the spec §Amendments and skip.

---

### Task 9: Closing data point + milestone closure

- [ ] **Step 1: Closing protocol run**

If any lever landed after Task 2's baseline **and Task 8 is not pending** (if Lever 2 was authorized, closure moves to that plan): run the M4a protocol (`gate-bench-protocol.sh`) on both machines against the final commit. Judge tg against the **headroom-set target recorded in the Task 3 amendment** — that is the milestone's exit bar. Record the v1 criterion line (tg ≥ 1x vs llama best-of) as context, never as the gate.

- [ ] **Step 2: Record and close**

Paste outputs into M4a §Amendments (protocol home) and cross-reference from M4b.11 §Amendments with the verdict against the headroom target. If both gates were STOPs, the closing amendment instead states the milestone closed as a diagnostic, with the two findings. Commit:

```bash
git add docs/superpowers/specs/
git commit -m "specs: M4b.11 closing data point — verdict vs the headroom-set target"
```

---

## Self-Review

- **Spec coverage:** attribution protocol → Tasks 1–2; pre-registered gates → Task 3; Lever 1 (kernels/pool/codegen/deployment class) → Tasks 4–6; short-context diagnostic + AGENTS.md → Task 7; Lever 2 handoff → Task 8; exit criteria 1–4 → Tasks 2, 3, 7, 8, 9. The spec's out-of-scope items (flash-decoding, NUMA, prefill, CI gates, threshold heuristics) appear as constraints, not tasks. ✔
- **Placeholder scan:** every code step carries complete code; the only "do not retype" instruction (Task 4 Step 3e) moves an existing verbatim body with a one-line change, stated exactly. Manual-protocol tasks (2, 7, 9) reference the runbooks that own those procedures. ✔
- **Type consistency:** `AttnHspanFn` is 13 args (4 ptr + 9 usize) everywhere — kernels (Task 4), pool type + stamp kernels (Task 5), LLVM declaration (Task 6). `inferno_par_attention_heads` is 11 args in the same order in the host fn (Task 5), the LLVM declaration, and the `build_call` (Task 6). `AttnHeadsJob` field order matches every initializer. ✔
