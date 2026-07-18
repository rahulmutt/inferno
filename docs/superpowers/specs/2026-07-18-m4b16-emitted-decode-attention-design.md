# M4b.16 — Codegen-Emitted Geometry-Specialized Decode Attention Design

**Date:** 2026-07-18
**Status:** Approved design, pre-implementation
**Milestone:** M4b.16 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.15](2026-07-17-m4b15-decode-attention-kernel-design.md))

This milestone takes up the **kernel-quality finding** M4b.15's closing
verdict recorded as the natural next decode lever: the decode attention
kernel leaves **28–37% whole-call cross-machine** on the table by
compiling with runtime dimensions — a dim-monomorphized kernel is
bit-neutral-able (arithmetic order unchanged) and sits ahead of any
KV-split parallelism work. M4b.16 ships that lever as a **codegen
emission**: the compiler emits, per compiled model, a private decode
attention function with the model's geometry baked in as constants,
bit-identical to the Rust kernels, dispatched through the existing pool
contract.

A toolchain prerequisite lands first as its own PR: **inkwell 0.6 →
0.9, LLVM 18.1 → 22.1** (see §Prerequisite).

## Motivation

At the M4b.15 closing benches (2026-07-18, both quiet-hw boxes, M4a
protocol, best-of-builds basis per the erratum):

| machine | pp512 vs best-of | tg128 vs best-of |
|---|---|---|
| d2.c1.medium (6336Y 16c) | 0.79x | 0.93x |
| s2.c2.medium (E-2388G 8c) | 0.70x | 0.87x |

The attribution record feeding this milestone, in full:

1. **Decode attention is neither bandwidth- nor dispatch-bound**
   (M4b.11 Gate 2; M4b.12 blame tables) — the wall is in the hspan
   kernel itself, plus drain-side lane imbalance at 16c only (8c drain
   is 2.1%: sub-head granularity has nothing to reclaim there).
2. **The M4b.15 instrument finding is the existence proof for this
   lever.** The phase-marginal µbench was ruled inadmissible precisely
   because its bench-local copies of the kernel compiled
   const-geometry-specialized — and ran **20–35% faster whole-call**
   than the shipping generic symbol at protocol geometry (dev box;
   28–37% cross-machine per the closing verdict). Same source, same
   LLVM backend, the only delta being const dims. M4b.16 makes that
   the shipping path.
3. **Attention's share of decode wall at best-t:** 29.1% (16c) / 20.6%
   (8c) (M4b.15 closing residual-wall statement). If emission
   recaptures most of the genericity cost, expected e2e decode tg gain
   is roughly **6–10% (16c)** and **4–7% (8c)** — this arithmetic is
   scoping context and feeds no gate.

## Approaches considered

- **A — stamped const-generic Rust variants** (`_hspan_hd64`, …):
  cheapest and near-zero risk, but caps at the stamped head_dim set,
  may leave the kv_dim-stride residue of the finding on the table
  (the µbench copies were full-geometry specialized), goes
  combinatorial if `(head_dim, kv_dim)` stamping is needed, and
  becomes dead code the day emission lands. Rejected in favor of B.
- **B — codegen-emitted attention (chosen):** the compiler emits the
  decode attention function per model with exact geometry baked.
  Full coverage across model architectures (head_dim 64/80/96/128,
  any GQA ratio), and the structural precedent already ships: M4b.9's
  `tok_body.*` outlining passes codegen-emitted private functions by
  pointer to pool dispatchers. Cost: attention exists twice (Rust
  oracle + emission code), held together by an exact-equality harness
  — the same discipline as today's scalar≡AVX2 bit-identity, extended
  to a third lane.
- **C — kernel bitcode const-propagated at `inferno compile` time:**
  killed on version arithmetic. The pinned rustc bundles LLVM 22.1.6;
  inkwell is pinned to LLVM 18.1, and LLVM bitcode has no forward
  compatibility (an 18 reader cannot parse 22 bitcode). The gap is
  structural — rustc adopts new LLVM majors ahead of llvm-sys/inkwell
  ceilings permanently. Rescues all lose: pinning a 2024-era rustc,
  chasing rustc's internal LLVM forever, invoking rustc at model
  compile time (reverses the v1 embedded-LLVM/no-external-toolchain
  decision), or maintaining a C-language kernel copy for clang-18
  bitcode (B's duplication cost with none of its payoffs). Note: after
  the §Prerequisite upgrade, rustc 22.1.6 and LLVM 22.1.8
  coincidentally share a major and bitcode would momentarily load —
  nothing is built on that alignment; it decays at the next rustc
  bump.

## Prerequisite: inkwell 0.9 / LLVM 22 (own PR, lands first)

Verified available 2026-07-18: inkwell **0.9.0** has an `llvm22-1`
feature; the **already-locked** nixpkgs rev
(`9e92285f211dad236540fd617d7e30e0b99bc0e1`) carries
`llvmPackages_22` at **22.1.8** — no flake bump needed.

- `inferno-codegen`: inkwell `0.6`/`llvm18-1` → `0.9`/`llvm22-1`,
  plus whatever API migration 0.7–0.9 require.
- `devenv.nix`: `llvmPackages_18` → `llvmPackages_22`;
  `LLVM_SYS_181_PREFIX` → `LLVM_SYS_221_PREFIX`.
- `AGENTS.md`: update the LLVM coupling note (feature flag ↔ devenv
  major.minor must continue to match exactly).
- Verification: correctness gates only — `inferno-codegen`
  differential, `inferno-core` artifact, `mise run test`,
  `mise run lint`, zero tolerance edits. No quiet-hw spend: a new
  LLVM major shifts generated code and therefore perf, which is
  exactly why the upgrade lands **before** the milestone's sessions —
  M4b.16's fresh baselines absorb it, and the lever-vs-baseline
  comparison is toolchain-invariant by construction (same binary).

Sequencing rationale: (1) baselines and lever runs sit on the same
toolchain, unconfounded; (2) B's emission code is written once
against the inkwell 0.9 API.

## Architecture

**The emitted function.** For each compiled model, codegen emits a
private function `attn_hspan.<n>` (naming per the `tok_body.*`
precedent) with the **exact 13-arg hspan C ABI**
(`out, q, kv, scores, kv_base, v_off, pos, kv_dim, n_heads,
n_kv_heads, head_dim, h_start, h_end` — see
`inferno_attention_f32_scalar_hspan`). The signature is unchanged
because the pool calls through the same `AttnFn` pointer type; the
emitted body simply **ignores the geometry parameters** (`kv_dim`,
`n_heads`, `n_kv_heads`, `head_dim`) in favor of the model's values
baked as IR constants, along with the derived scale `1/√head_dim`.
Genuinely per-call arguments stay live: the four pointers, `kv_base`,
`v_off`, `pos`, `h_start`, `h_end` (`kv_base`/`v_off` vary per layer
and stay runtime).

**Dispatch.** The decode attention lowering passes the emitted
function's pointer to `inferno_par_attention_heads` instead of the
declared runtime symbol (`attention_hspan_symbol`). The pool contract
is byte-unchanged — **no `HOST_ABI_VERSION` bump for the lever**. The
`m <= 1`/threads=1 bypass path goes through the same pointer and is
covered by the same guarantees.

**Emission content.** A port of the decode `attn_core` as explicit
8-lane vector IR (`<8 x float>` + `llvm.fma` intrinsics), preserving
the Rust kernels' arithmetic order exactly:

- QK dot: `dot8`'s lane-partitioned partial sums and `reduce8`'s
  fixed reduction tree, verbatim order;
- softmax: max pass, then exp+denominator with `expf`'s constants and
  FMA order ported literally (clamp, `round_ties_even`, the degree-6
  Horner chain, exponent-bit scale — `expf.rs` is the reference);
- AV accumulate: same per-position FMA-into-output-row order.

Both ISA flavors come from the same emission code; the module's
target features decide legalization (AVX2 → 256-bit ops; scalar
baseline → 2×4-wide SSE2 with per-lane order unchanged, hence
bit-identical — the same trick the Rust scalar kernel plays). No
fast-math flags anywhere in the emitted function; strict IEEE
semantics are what make "bit-neutral" arguable from construction.
`head_dim` stays constrained to multiples of 8 (existing kernel
contract).

**What the Rust kernels remain:** the interpreter path, the prefill
path (M4b.14 qblock kernels untouched — prefill is out of scope per
its all-STOP), the rig oracle, and the fallback.

**The switch.** A codegen-level flag (`INFERNO_EMITTED_ATTN`,
default per gate outcome) selects emitted-function vs runtime-symbol
dispatch at model-compile time. The flag is **hashed into the
artifact cache key** (as `HOST_ABI_VERSION` already is), so the two
variants never collide in the cache. This is not just a kill switch:
it is how the gate sessions produce lever-vs-baseline artifacts from
one binary.

## Gating protocol

**Comparison:** lever = emitted-attention artifact, baseline =
runtime-symbol artifact, both compiled by the same binary on the
post-upgrade toolchain, switched by the cache-keyed flag. Both
comparands are real shipping code paths in real artifacts — the
M4b.15 instrument problem (bench-local copies compiling specialized)
is moot by construction.

**Gate ladder (M4b.11 thresholds verbatim):** e2e decode tg at
best-t, M4a protocol model and geometry, lever vs baseline:

- **≥5% on both boxes → ship** (flag flips default-on; runtime-symbol
  path kept as env kill-switch);
- **<3% on both → STOP** (flag stays default-off; milestone closes as
  a diagnostic with the recorded data);
- anything between or split → recorded judgment, arithmetic shown
  once, per standing convention.

**Order of evidence, cheapest first:**

1. **Local admission (no PNAP spend until green):** bit-exactness
   harness, same-logits invariant, codegen differential, core
   artifact test, `mise run test` + `mise run lint`, zero
   tolerance-file edits. Then a local dev-box lever-vs-baseline e2e
   run — context only, never a gate, but a sanity check that the
   lever isn't inverted before money is spent.
2. **Session A (d2.c1.medium 16c) + Session B (s2.c2.medium 8c):**
   fresh llama.cpp baselines are **mandatory** this cycle (the
   toolchain changed); protocol tables recorded verbatim in the M4a
   spec §Amendments; lever-vs-baseline runs and gate arithmetic in
   this spec's §Amendments.
3. **Closing verdict:** exit-criteria walk below, including the
   residual-decode-wall-shape statement for next-milestone scoping
   (post-lever the wall should be even more GEMV/bandwidth-shaped —
   state it with numbers).

**v1 context ratios** (vs llama.cpp best-of-builds) recorded as
always — context, never the gate.

## Testing

1. **Bit-exactness harness** (new, `inferno-codegen` tests,
   proptest-driven): random geometries (head_dim ∈ {64, 80, 96, 128},
   GQA ratios including 1, kv_dim derived), random head sub-spans
   (`h_start`/`h_end` as the pool actually calls), pos swept
   including large positions (M4b.15's inversions appeared at
   pos ≥ 1023), random Q/KV contents. Compile a module exposing the
   emitted function, dlopen, assert **exact bit equality** against
   both `inferno_attention_f32_scalar_hspan` and `_avx2_hspan`. This
   is the third lane of the scalar≡AVX2 rig discipline.
2. **Same-logits invariant** (new): compile the fixture model twice —
   flag off and flag on — and assert the two artifacts produce
   **bit-identical logits** over the fixture prompts. Catches
   integration-level mistakes (wrong pointer wiring, wrong baked
   constant) that per-function tests can't.
3. **Existing suites, untouched and green:** kernel rig (Rust kernels
   byte-untouched), codegen differential, core artifact differential,
   `mise run test`, `mise run lint`. Zero edits to `tolerance.rs`.
   No parser changes → no fuzz obligation.

## Exit criteria

1. Prerequisite PR (inkwell 0.9 / LLVM 22 / devenv / AGENTS.md
   coupling note) landed first, all correctness gates green.
2. Emitted decode attention shipped behind the cache-keyed codegen
   flag; default set by the gate outcome (ship → default-on with the
   runtime-symbol env kill-switch; STOP → default-off, lever recorded
   as a finding).
3. Bit-exactness harness and same-logits invariant green, including
   non-protocol geometries.
4. Standing invariants held: rig green, differentials green, zero
   tolerance changes, `mise run test`/`lint` green.
5. Both quiet-hw sessions run with fresh llama.cpp baselines;
   protocol tables in the M4a spec §Amendments; gate arithmetic
   recorded once in this spec's §Amendments; ship/STOP per the
   ladder.
6. Closing verdict includes the residual-decode-wall-shape statement
   for next-milestone scoping.
7. Every STOP recorded as a finding (M4b.12 precedent).
8. AGENTS.md decode-attention paragraph updated to describe the
   emitted path and its flag.

## Risks

- **Perf miss:** LLVM's scheduling/register allocation of emitted
  vector IR could trail the hand-intrinsic Rust AVX2 kernel despite
  const dims. The gate ladder handles this honestly (STOP is a
  finding; the flag defaults off). The M4b.15 evidence — same
  backend, const geometry, 20–35% faster — bounds this risk but does
  not eliminate it.
- **Dual-implementation drift:** every future decode-attention change
  must land in both the Rust kernel and the emission code. The
  bit-exactness harness makes drift a hard test failure, not a silent
  divergence; the cost is accepted as the price of full geometry
  coverage and future fusion (rope-into-attention, F16 KV, structural
  variants), none reachable from stamped Rust variants.
- **Toolchain upgrade fallout:** LLVM 18 → 22 may shift codegen
  behavior beyond attention (differential/artifact gates are the
  net) and inkwell 0.7–0.9 API churn may be nontrivial. Contained by
  landing as its own PR with full correctness verification before
  any milestone work.
- **8c ceiling honesty:** attention is 20.6% of the 8c decode wall;
  even a full recapture of the genericity cost yields ~4–7% e2e
  there. If the 16c box gates ≥5% and the 8c lands in 3–5%, the
  split-verdict judgment must be recorded, not hand-waved.

## Amendments

Session records, gate verdicts, and the closing exit-criteria walk are
appended here as they happen.

### 2026-07-18 — Local admission + local context point (non-quiet dev box)

**Local admission (Task 7, at e870f78): PASS.** attn_emit 4/4,
differential 6/6, artifact 5/5 (nextest and `--test-threads=1`; the
parallel plain-cargo runner shows the known pre-existing intra-process
env race — reproduced with the pre-Task-6 test file, so not a
regression; CI's nextest lane is the gate of record), `mise run test`
69/69 (+315 unit), `mise run lint` clean, tolerance.rs diff vs main
empty, kernels diff vs main = expf.rs/lib.rs visibility-only.

**Local context point — context only, never a gate.** Machine:
operator devpod, AMD Ryzen 9 3900 (12c/24t), SHARED AND NOISY —
recorded verbatim, noise dominates. Two back-to-back pairs
(pp512/tg128/reps5/threads0, binary at e870f78, llama.cpp 6f4f53f):

| run | variant | inferno t=12 tg | inferno t=1 tg | llama t=12 tg (control) |
|---|---|---|---|---|
| 1 | baseline | 11.12 ± 1.04 | 13.32 ± 0.08 | 12.31 ± 2.36 |
| 1 | lever    |  5.58 ± 1.67 | 12.50 ± 0.39 |  7.82 ± 3.88 |
| 2 | baseline |  7.50 ± 0.59 | 12.42 ± 0.66 | 14.81 ± 0.49 |
| 2 | lever    | 13.55 ± 1.03 | 13.34 ± 0.17 | 10.24 ± 2.18 |

The llama.cpp control swings 7.82–14.81 across runs and the
inferno-vs-itself sign flips run to run (t=12 AND t=1), so no
directional read exists in this data. Sanity-check outcome for the
plan's stop-condition ("lever slower by more than noise"): NOT
TRIGGERED — the lever is not consistently slower (run 2 has it faster
in every row; t=1 means are dead even at ~12.9 both variants).
Sessions may proceed; the quiet-hw boxes are the only valid
comparison venue, which is exactly why the gate lives there.
