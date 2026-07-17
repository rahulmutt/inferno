# M4b.13 — Prefill GEMM Per-Thread Kernel Quality (Register Tiles, Gated VNNI) Design

**Date:** 2026-07-17
**Status:** Approved design, pre-implementation
**Milestone:** M4b.13 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.12](2026-07-16-m4b12-decode-attention-headroom-attribution-design.md))

This milestone takes up the prefill half of the v1 win criterion — the
larger of the two open gaps — via the register-blocked GEMM escalation
pre-gated (and never run) in the
[M4b.2 amendment](2026-07-07-m4b2-per-thread-gap-design.md), laddered with
the AVX-512 VNNI kernel path the v1 design planned for M2 ("AVX2 then
AVX-512") and never built.

## Motivation

At the M4b.12 closing benches (2026-07-17, both quiet-hw boxes, M4a
protocol), the v1 win criterion (pp > 1x AND tg > 1x vs llama.cpp at its
best) stands at:

| machine | pp512 vs best-of | tg128 vs best-of | t=1 pp | t=1 tg |
|---|---|---|---|---|
| d2.c1.medium (6336Y 16c) | 0.75x | 0.91x | 0.54x | 0.98x |
| s2.c2.medium (E-2388G 8c) | 0.60x | 0.86x | 0.55x | 1.06x |

Prefill is the blocking gap, and it is a **per-thread kernel quality** gap,
not a scaling gap: prefill scaling was closed by M4b.8/M4b.9 (10.63x @
t=12, gate MET), and inferno's prefill scales *better* than llama.cpp's on
the 16c box (13.2x vs 10.5x) — while t=1 prefill sits at 0.54x/0.55x on
both machines.

Attribution for the per-thread gap is already on record. The M4b.3 closing
profile (dev box, t=1) puts **matmul at 66.7% of prefill cycles** — lm_head
19.5%, FFN gate/up/down 14.9/14.8/14.8%, attn q/o 2.8/2.7%, v/k 0.4% each —
at ~15.8 effective GB/s, with attention down to 26.4–26.8% post-M4b.3. The
M4b.2 amendment gated a register-blocked GEMM follow-up on exactly this
shape of profile; M4b.3's closing verdict named it "the gap-closing" lever
and explicitly did not start it. This milestone runs it.

Two kernel-level deficits are visible by inspection of
`inferno_gemm_q8_0_rs8_avx2` (`inferno-kernels/src/q8_0.rs`):

- **Weight vectors are re-loaded once per token.** Per k-block the loop
  order is token → lane, so the 8 weight vectors of a row-strip (256 B) are
  loaded `m` times per block instead of once. They hit L1 after the first
  token, but the inner loop becomes load-port-bound rather than FMA-bound.
- **Accumulators live in a heap `vec![__m256; m]`**, read-modify-written
  once per token per k-block — a load+FMA+store round-trip per update that
  register-resident accumulators would eliminate entirely.

Separately, the kernel set is AVX2+FMA only (`_mm256_maddubs_epi16` +
`_mm256_madd_epi16` int8 dot). Both criterion boxes (Ice Lake SP 6336Y,
Rocket Lake E-2388G) have AVX-512 VNNI, which llama.cpp uses:
`vpdpbusd` fuses the u8×i8→i32 dot into one instruction at 512-bit width.
`inferno-target` already detects it (`Feature::Vnni`) and the registry
already has the dispatch seam (`Isa::X86_64v4` currently falls back to the
AVX2 set, annotated "no v4-specific kernels exist in M2").

**Ladder arithmetic** (why both levers are in one milestone): with matmul
at ~66.7% of t=1 prefill, a combined 3x matmul speedup yields ~1.8x
prefill — pp ~1.35x/1.08x at full threads if per-thread gains carry
(scaling is proven). The 8c box needs 1.67x end-to-end: register blocking
alone was bounded at "necessary but not sufficient" by the M4b.2
amendment's own arithmetic, so stopping the milestone there would
guarantee an M4b.14 with no new information. The VNNI lever stays in
scope, behind a pre-registered gate.

## Scope Decisions (M4b.13)

| Decision | Choice |
|---|---|
| Levers | **Ladder, both pre-registered:** Lever 1 = register-blocked AVX2 GEMM tiles (the M4b.2 escalation); Lever 2 = AVX-512 VNNI GEMM path, authorized only by the mid-milestone gate |
| Exit criterion | **Hard: pp vs llama best-of ≥ 1.0x on both quiet-hw boxes.** Sanctioned STOP-out: ladder complete + fresh split-bracket profile shows the residual gap outside matmul → record the finding, close as diagnostic (M4b.12 precedent) |
| Phase | **Prefill GEMM (`m > 1`) only.** GEMV and every decode path untouched — decode is bandwidth-bound (M4b.6/M4b.12 findings); no tg claim is made |
| Dtype | **Q8_0 only** (the criterion model). Q4_K keeps its existing kernels and invariants |
| Machines | 16c `d2.c1.medium` (6336Y) + 8c `s2.c2.medium` (E-2388G), M4b.12 precedent |
| Standing invariants | `gemm(m=1)` bit-equals `gemv`; scalar-vs-SIMD bit-identity per ISA (VNNI joins the rig); cross-thread and cross-`prefill_tile` bit-identity; compiled-vs-interpreter differential green with **no tolerance loosening** |
| Attribution freshness | The mid-milestone quiet-hw session doubles as the fresh t=1 profile on the criterion machines (the 66.7% figure is dev-box, 2026-07-07, pre-M4b.9 bracket splits) — no separate attribution session is provisioned |

**Explicitly out of scope:** prefill attention (~27% of t=1 prefill — its
own future item, scoped against the fresh profile this milestone records);
the `quantize`/`kv_append` brackets (sub-1% each, M4b.9); any decode lever;
NEON/AOT (v2); prefill batching strategy changes (`PREFILL_TILE` stays).

## Lever 1 — Register-Blocked AVX2 GEMM Tiles

**Kernel restructuring** (`inferno_gemm_q8_0_rs8_avx2`; the scalar sibling
mirrors the loop order so the rig compares like to like):

- Tokens are processed in groups of **MR** (4–8, fixed at plan time by
  µbench) against one 8-row strip. Loop order becomes strip → token-group →
  k-block → lane, with the MR-token loop innermost inside each lane.
- Per k-block, each lane's weight vector is loaded **once** and reused
  across all MR tokens (today: reloaded per token). Activation loads are
  already minimal (once per token per block) and stay as-is.
- The MR accumulators are **YMM-register-resident across the entire
  k-loop**; the heap `acc` round-trip per block disappears. Head/tail
  tokens (`m % MR`) and partial strips fall back to the existing paths.

**Bit-identity by construction.** For each output `(t, r)` the f32
accumulation remains sequential over k-blocks in today's order — reordering
the token/lane loops never touches a single output's accumulation chain.
All standing invariants hold with no tolerance re-derivation.

**Lever-1 local gate (dev box, before any metal spend):** criterion µbench
on the three profile-blamed shapes — FFN 4864×896 (k=896), 896×4864
(k=4864), lm_head 151936×896 — must show the tiled kernel beating the
current `gemm_*_rs8` on Gelem/s, and dev-box t=1 pp must improve in a
same-session before/after `bench-compiled` pair. Only then is the quiet-hw
session provisioned. This is a local iteration gate, not a pre-registered
decision rule: if the tiled kernel does not beat the current one, iterate
on the tile shape — metal spend stays blocked until it does.

## Mid-Milestone Gate → Lever 2

One quiet-hw session per box after Lever 1 lands: the M4a protocol run
**plus** a fresh split-bracket t=1 prefill profile (post-M4b.9 brackets:
matmul, attention, kv_append, quantize, small ops). Pre-registered decision
rule, applied per the recorded numbers with arithmetic shown:

1. **pp ≥ 1.0x on both boxes** → exit criterion met; Lever 2 does not run.
2. **pp < 1.0x on either box AND the ceiling check passes** — measured pp
   ratio ÷ (1 − matmul_share × ½) ≥ 1.0 on every box still under 1.0x,
   i.e. an assumed ≤2x VNNI matmul speedup arithmetically reaches the
   bar — → **Lever 2 authorized**.
3. **pp < 1.0x AND the ceiling check fails** → **STOP-out**: even the
   full assumed VNNI gain cannot close the measured gap from the matmul
   share alone; the residual lives elsewhere (likely prefill attention or
   quantize). Record the finding with the profile table; close as
   diagnostic. An all-STOP with the finding is a successful outcome
   (M4b.12 precedent).

The ½ factor is the pre-registered assumed ceiling for VNNI over the
Lever-1 kernel (2x: width doubling × instruction fusion, discounted for
memory-bound stretches); it is fixed here, before data, and is not
adjustable at gate time.

## Lever 2 — AVX-512 VNNI GEMM Path (gated)

- New `KernelIsa::Avx512Vnni` registry variant; the `Isa::X86_64v4` arm
  selects it when the running CPU reports `Feature::Vnni` (runtime check
  mirrors `KernelIsa::available()`), else falls back to AVX2 exactly as
  today. GEMM only — GEMV and all decode paths keep the AVX2 kernels.
- The kernel keeps Lever 1's register-tile structure and per-output f32
  accumulation order, replacing the `maddubs+madd` pair with `vpdpbusd`
  (u8×i8→i32, integer-exact, no i16 intermediate) at 512-bit width. The
  integer dot is exact in both ISAs, so **scalar-vs-VNNI bit-identity is
  demanded by the rig**, same bar as AVX2 — not a tolerance.
- **Dev-loop constraint, stated honestly:** the dev box is Zen 2 and cannot
  execute this path. Correctness runs under Intel SDE locally and in CI
  (devenv addition per developer-environment rules; the rig skips-not-fails
  where the ISA is absent, matching the registry's refusal behavior).
  Performance iteration happens only in the quiet-hw sessions.

## Testing

- **Rig invariants** (existing bit-identity rig, extended): scalar vs AVX2
  vs VNNI bit-identity per shape; `gemm(m=1)` ≡ `gemv`; cross-thread and
  cross-`prefill_tile` identity; compiled-vs-interpreter differential
  unchanged.
- **µbench** (criterion): the three blamed shapes, current vs tiled vs VNNI
  where runnable, recorded in Amendments.
- **SDE lane in CI** for the VNNI kernel's correctness tests; native rig
  runs on the metal boxes before any protocol number is recorded.
- No CI perf gates (standing policy).

## Task Sequence

1. Register-tiled AVX2 GEMM kernel + scalar mirror + rig extension.
2. µbench the three shapes; fix MR; record the dev-box data point
   (Lever-1 local gate) in Amendments.
3. Quiet-hw session, both boxes: M4a protocol + fresh split-bracket t=1
   profile; record verbatim; apply the gate rule with arithmetic shown.
4. *(gated)* SDE in devenv + CI lane; `KernelIsa::Avx512Vnni` registry
   variant + VNNI kernel + rig extension.
5. *(gated)* Closing quiet-hw session, both boxes: full protocol, recorded
   verbatim in M4a §Amendments; exit-criteria walk in this spec's
   Amendments.

## Exit Criteria

1. Lever-1 dev-box data point (µbench + t=1 pp) recorded.
2. Fresh split-bracket t=1 prefill profile from both criterion boxes
   recorded; gate verdict recorded once, arithmetic shown.
3. Every gate outcome recorded honestly, including STOP; no lever ships
   without its gate.
4. Closing protocol run on both boxes recorded verbatim in the M4a spec
   §Amendments: **pp vs llama best-of ≥ 1.0x on both boxes**, or the
   STOP finding naming where the residual prefill gap lives.

## Risks

- **The 8c box needs 1.67x prefill** — the tightest margin in the ladder
  arithmetic. The sanctioned STOP-out exists precisely for this: if the
  ladder lands and the residual is attention-shaped, that finding scopes
  the next milestone instead of this one overrunning.
- **`vpdpbusd` operand semantics** (u8×i8 ordering vs the current
  `sign_epi8` trick) are verified by the bit-identity rig, not assumed.
- **llama.cpp is a moving target** — the reference build is re-pinned at
  session time per the M4a protocol; ratios are judged within-session.
- **Metal session economics** — no parallel provisions; `metal-gc` after
  any failed session (standing runbook).
