# M4b.14 — Prefill Attention Per-Thread Kernel Quality (Query-Blocked, Gated Lever 2) Design

**Date:** 2026-07-17
**Status:** Approved design, pre-implementation
**Milestone:** M4b.14 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.13](2026-07-17-m4b13-prefill-gemm-register-tiles-vnni-design.md))

This milestone takes up the **prefill attention** tail — the residual
prefill gap that M4b.13's closing verdict scoped out of GEMM work and
named as the next prefill target. It follows the M4b.13 discipline
exactly: a bit-identity per-thread kernel-quality lever first, a second
lever pre-registered behind a mid-milestone gate that only fires if a
fresh profile blames a shape the lever can close.

## Motivation

At the M4b.13 closing benches (2026-07-17, both quiet-hw boxes, M4a
protocol), the v1 win criterion (pp > 1x AND tg > 1x vs llama.cpp at its
best) stands at:

| machine | pp512 vs best-of | tg128 vs best-of |
|---|---|---|
| d2.c1.medium (6336Y 16c) | 0.74x | 0.96x |
| s2.c2.medium (E-2388G 8c) | 0.70x | 0.86x |

Prefill is the blocking gap, and M4b.13 established two things about it
that scope this milestone:

1. **The GEMM side is exhausted.** M4b.13's Lever 1 (register-tiled Q8_0
   prefill GEMM) shipped; the mid-milestone split-bracket profiles put
   matmul at 58.57% / 58.32% of t=1 prefill cycles, running at
   memory-stream rates (47.6 GB/s on the 16c box, ~68 GB/s on the 8c),
   and the ceiling arithmetic showed no GEMM-side lever — including a
   perfect ½-ceiling VNNI — can reach pp 1.0x on the 8-core box. Lever 2
   (VNNI) was gated out (rule 3 STOP). The prefill gap is not in matmul.

2. **The residual is attention-shaped.** Attention is the largest single
   t=1 prefill bracket on both boxes — **35.5% (16c) / 36.5% (8c)** — and
   dominates the ~41% non-matmul tail. M4b.13's closing verdict states it
   directly: *"The next prefill milestone, if any, must target the ~41%
   non-matmul tail, which attention dominates."* This is that milestone.

Prefill attention **scaling** is already closed — M4b.8 tiled it across
`active_threads()` via `inferno_par_attention` (10.63x @ t=12, gate MET)
and M4b.9 token-sharded the serial tail. So, exactly as M4b.13 was for
GEMM, this is a **per-thread kernel-quality** gap, not a scaling gap: the
kernel each lane runs (`inferno_attention_f32_avx2`, one call per query
token) is where the wall time sits.

### The kernel-level deficit (by inspection)

`inferno_attention_f32_avx2` / `attn_core_avx2`
(`inferno-kernels/src/attention.rs`) computes **one query token per
call**. The pool's `run_attn_span` (`inferno-pool/src/pool.rs`) loops it
per token within a lane's shard: `for t in start..end { (j.kernel)(...) }`.
For a query row-strip of `m` tokens a lane owns, this streams the entire
visible K and V region of the layer's KV cache **once per query token**:

- Scores pass: for each `t`, for each head `h`, dot `q[t,h]` against
  every visible `k[·,g]` — reads the whole K region.
- Output pass: for each `t`, weighted sum over every visible `v[·,g]` —
  reads the whole V region.

Across a lane's `m` tokens the K and V regions are re-streamed `m` times.
At prefill sequence lengths the visible KV region is far larger than L2,
so every query token pays main-memory bandwidth for KV it could have
reused from cache had multiple query tokens been processed against the
same KV block while it was resident. This is the attention analogue of
M4b.13's GEMM finding ("weight vectors re-loaded once per token"): the
loop order streams the large operand once per token instead of once per
block.

**Ladder arithmetic** (why the milestone is scoped as a ladder, not a
single lever): with attention at ~36% of t=1 prefill and matmul at ~58%
(and already at memory-stream rates), taking matmul and the small
remaining tail as fixed, the pp reachable by removing a fraction `c` of
the attention bracket is `pp_ratio / (1 - attn_share * c)`. To clear
**1.0x on the 8c box** (`pp_ratio = 0.70`, `attn_share ≈ 0.365`) needs
`c ≥ 0.82` — i.e. **~82% of the entire attention bracket** must be
removed; the 16c box needs `c ≥ 0.73`. Fully eliminating attention would
reach only 1.10x (8c) / 1.15x (16c), so there is *some* headroom in
principle — but a single bandwidth lever that recovers the KV-refetch
waste is very unlikely to hit an 82% bracket reduction on its own. Query
blocking (Lever 1) is therefore bounded by the same "necessary but not
sufficient" logic M4b.13 applied to register blocking, and a second
attention lever stays **in scope, behind a pre-registered gate**. The
milestone's honest exit may be a STOP-out diagnostic (M4b.12/M4b.13
precedent) rather than a criterion win.

## Scope Decisions (M4b.14)

| Decision | Choice |
|---|---|
| Levers | **Ladder, both pre-registered:** Lever 1 = query-blocked attention kernel (K/V streamed once per query block, bit-identical to the per-token kernel); Lever 2 = authorized only by the mid-milestone gate, chosen from a blame-keyed menu (below) |
| Exit criterion | **Hard: pp vs llama best-of ≥ 1.0x on both quiet-hw boxes.** Sanctioned STOP-out: ladder complete + fresh split-bracket profile shows the residual prefill gap is outside what the scoped attention levers can close → record the finding, close as diagnostic (M4b.12/M4b.13 precedent) |
| Phase | **Prefill attention (`m > 1`) only.** Decode attention (`inferno_par_attention_heads`, one query token) and every GEMM/GEMV path untouched — decode attention is the M4b.11/M4b.12 finding's territory and is bandwidth/kernel-bound separately; no tg claim is made |
| Dtype | **f32 KV** (the M3 invariant; KV is f32 in the compiled path). No F16 KV — that was M4b.11's gated-out Lever 2 and stays closed |
| Machines | 16c `d2.c1.medium` (6336Y) + 8c `s2.c2.medium` (E-2388G), M4b.12/M4b.13 precedent |
| Standing invariants | Query-blocked kernel bit-equals the per-token kernel over any query-block tiling; scalar-vs-AVX2 bit-identity per ISA; the M4b.11 `_hspan` head-range identity preserved; cross-thread and cross-`prefill_tile` bit-identity; `m == 1` still bit-equals the decode/GEMV attention path; compiled-vs-interpreter differential green with **no `attn_rel_tol` loosening** |
| Attribution freshness | The mid-milestone quiet-hw session doubles as the fresh t=1 split-bracket profile on the criterion machines — no separate attribution session is provisioned (M4b.13 precedent) |

**Explicitly out of scope:**

- **Decode attention.** `inferno_par_attention_heads` and its `_hspan`
  kernels are untouched. The M4b.12 attribution record (hspan kernel
  itself + 16c lane imbalance) is the starting point for any future
  decode-attention work; this milestone makes no tg claim.
- **F16 KV / any KV dtype change.** KV stays f32 (M3 invariant). Changing
  it re-derives every prefill tolerance and was M4b.11's gated-out lever.
- **Flash-attention-style online softmax as Lever 1.** A single-sweep
  fused kernel breaks bit-identity immediately (running-max renormalize
  changes the reduction order and the exp count), forcing a tolerance
  re-derivation *before* any win is proven. It is available to the Lever 2
  menu only, gated, if and only if the profile blames the softmax/exp
  sub-bracket — never as the un-gated first move.
- **Attention-as-GEMM** (routing Q·Kᵀ and P·V through the register-tiled
  GEMM path). Large planner/codegen change, materializes score matrices,
  and its numerics differ from the current fused per-head kernel. Out of
  scope for this milestone; notable as a possible future direction only.

## Lever 1 — Query-Blocked Attention Kernel

**The idea.** Replace the per-token call (`run_attn_span`'s
`for t { kernel(t) }`) with a kernel that takes a **query block** of `mb`
tokens `[t0, t0+mb)` and, for each visible KV position, updates all `mb`
query tokens' partial scores/outputs while that KV vector is cache-hot.
K and V are streamed **once per block** instead of once per token. The
per-head math each query token receives is unchanged — same `dot8`
partition order, same `reduce8` tree, same `expf` polynomial, same
`mul_add` V-accumulation order.

**Why it can be bit-identical.** The per-token kernel's numeric identity
for query token `t` is a pure function of `(q[t], visible K, visible V,
pos_t)`. Blocking only changes the *loop order over the block axis* — the
order in which distinct query tokens are visited — not the order of any
reduction *within* a query token. Each query token still: (1) computes
`scores[t'] = dot8(q, k[t']) * scale` for `t' = 0..=pos` in ascending
`t'`; (2) takes the max, then the block-of-8 `reduce8` denom + scalar
tail exactly as `attn_core_scalar`/`attn_core_avx2` do today; (3)
accumulates `out += (w/denom) * v[t']` for ascending `t'`. Nothing in any
one query token's arithmetic depends on whether a sibling query token was
computed before or after it. Therefore, for **any** partition of the
query axis into blocks, and independently of `mb`, the result is
bitwise-identical to the per-token kernel. This is the same structural
argument M4b.11 used for `_hspan` on the head axis, now applied to the
query axis.

Two implementation shapes are admissible under this identity; the plan
picks one and the rig proves it:

- **Two-pass, scores materialized per block.** For the block, compute the
  full `scores[mb][visible]` (scores pass, KV-K streamed once), then the
  softmax per row, then the output pass (KV-V streamed once). Scratch is
  `mb * visible` f32 instead of the per-token `visible`. Simplest to prove
  identical because each row's reduction is textually the current loop.
- **Register-blocked scores pass.** Hold `mb` query tokens' accumulators
  across the KV-K loop so each `k[t']` load feeds `mb` dot-products — the
  direct bandwidth analogue of M4b.13's register tiles. Requires the
  8-lane partition + `reduce8` tree to be preserved per query token
  (accumulate into `mb` separate `[f32;8]`/`__m256` lanes, reduce each
  independently), which keeps identity.

Both stream KV once per block; the register-blocked pass additionally cuts
the scores-pass load-port traffic. The plan may land the two-pass shape
first (identity is most obviously preserved) and register-block the inner
pass as a follow-on within Lever 1 if the µbench shows the load ports are
the bind — each step guarded by the same rig identity test.

**ABI and plumbing.** The kernel gains a query-block entry alongside the
existing per-token symbols:

- New kernel symbol `inferno_attention_f32_{scalar,avx2}_qblock` (or the
  per-token symbol generalized to take `m_block` + strides — plan's
  choice; the ABI is internal, chosen for cleanliness). It takes the
  current `AttnFn` arguments plus the block length and the `q`/`out`
  row strides so it can walk the block's rows.
- `inferno-pool`: `run_attn_span` calls the query-block kernel **once per
  lane shard** (`kernel(shard_start, shard_len, …)`) instead of looping
  per token. The lane's `scores` scratch grows from `pos0 + end` to the
  two-pass block scratch (`mb * visible`) when the two-pass shape is used;
  this stays a per-lane Vec (M4b.8's "one Vec per lane per tile per layer
  is noise" holds — it is a scratch resize, not a new allocation site).
  The `AttnJob` fields are unchanged except the kernel type; the sharded
  axis is still the query-token index, so cross-thread disjointness (each
  lane owns a disjoint query-row range, writes disjoint `out` rows) is
  preserved verbatim.
- Codegen (`inferno-codegen`) is **unchanged**: it already emits a single
  `inferno_par_attention` call per prefill attention op with the tile's
  query rows; the query-blocking lives entirely below that boundary in the
  pool + kernel. This keeps the change off the generated-code / cache-key
  surface. `HOST_ABI_VERSION` (`inferno-codegen/src/lib.rs`, currently
  `"7"`) is bumped **only if** the `inferno_par_attention` signature
  changes; if the change is contained to the internal kernel symbol +
  pool call it does not, and the bump is unnecessary. The plan states
  which and bumps iff the host-visible ABI moved (a bump forces a clean
  recompile, which is correct and cheap, but claiming one when the ABI
  didn't move pollutes the record — so it is decided by the actual
  signature, not by default).

**`m == 1` and the GEMV/decode identity.** `inferno_par_attention` already
special-cases `m == 1` to `run_attn_span(&job, 0, 1)` (single serial
token). Query blocking with block length 1 must reduce to the current
per-token call bit-for-bit; the rig asserts `qblock(mb=1) == per_token`.
Decode (`inferno_par_attention_heads`) does not go through this path at
all and is untouched.

## Mid-Milestone Gate — Lever 2 Authorization

After Lever 1 lands and is measured on both boxes, a fresh quiet-hw
split-bracket t=1 prefill profile decides Lever 2, using the same
arithmetic discipline as M4b.13's ladder.

**Inputs (measured, per box):**

- `pp_ratio` — post-Lever-1 pp512 vs llama best-of, from the session's
  `gate-bench-protocol.sh` run.
- The split-bracket t=1 prefill op table from
  `gate-prefill-attn-split.sh` (new; §Testing), which additionally splits
  the **attention** bracket into its sub-brackets: `attn:scores` (Q·Kᵀ +
  scale), `attn:softmax` (max + exp + denom), `attn:output` (P·V). This
  is the M4b.12 dispatch-split precedent applied to the prefill attention
  kernel so the gate can blame a *sub-bracket*, not just "attention".

**Gate rule (pre-registered, human-computed and pasted into §Amendments):**

1. Compute `attn_share = attn_total / prefill_total` from the fresh table.
2. **Ceiling check.** For a candidate Lever-2 that targets a sub-bracket
   of fractional cost `f` (of prefill) with a best-case ceiling factor
   `c` (fraction of that sub-bracket's cost the lever can remove; `c` is
   the lever's *pre-registered optimistic* ceiling, e.g. ½ for a
   fused-pass that halves memory traffic), the post-lever prefill time is
   `1 - f*c` of current, so the achievable pp is
   `pp_ratio / (1 - f*c)`. **Lever 2 is authorized on a box only if
   `pp_ratio / (1 - f*c) >= 1.0`** there — i.e. only if the lever's own
   optimistic ceiling can, alone, reach the criterion. If not on either
   box, STOP (an all-STOP with the finding is a successful diagnostic
   outcome, M4b.13 precedent).
3. **Blame gate.** The menu entry chosen must be the one whose target
   sub-bracket the fresh table actually blames (largest admissible `f`).
   A lever targeting a sub-bracket that the profile shows is already
   small does not get authorized even if its arithmetic closes — the
   measurement, not the menu order, picks the lever.

**Pre-registered Lever 2 menu** (blame-keyed; the gate picks at most one):

| If the fresh profile blames… | Lever 2 candidate | Optimistic ceiling `c` | Numerics |
|---|---|---|---|
| `attn:scores` + `attn:output` memory traffic (KV streaming still the bind after blocking) | **Wider query blocks / KV-panel prefetch tuning** within the existing bit-identical kernel | modest (bounded by remaining refetch) | Bit-identical (same kernel family) |
| `attn:softmax` (exp/denom dominates a short-context prefill) | **Vectorized/batched softmax** over the block's score rows (shared max/denom reductions), still using the `expf` poly and `reduce8` tree | modest | Bit-identical if the per-row reduction order is preserved; **re-derive `attn_rel_tol` only if** the block-shared reduction changes the order (flagged, gated) |
| `attn:scores`+`attn:output` compute-bound after blocking (memory no longer the bind) | **Flash-attention online-softmax fused pass** (single KV sweep, running-max renormalize) | ½ (one KV sweep instead of two) | **Not bit-identical** — running renormalize changes reduction order + exp count; requires an `observed_error` re-derivation of `attn_rel_tol` from the fused kernel's error sweep, gated behind this authorization, differential re-green with the new (data-armed, not loosened) tolerance |

The flash-attention entry is the only one that breaks bit-identity, and
it is authorized **only** by the gate blaming the compute (not memory)
sub-bracket *and* its ½-ceiling arithmetic clearing 1.0x — the exact
STOP-out logic that kept M4b.13's VNNI unbuilt.

## Standing Invariants & Testing

**Bit-identity invariants (the rig is the guard):**

1. **Query-block == per-token.** For random shapes and every block length
   `mb ∈ {1, 2, small, tile-sized}`, the query-blocked kernel's output
   equals the per-token kernel's output **bit-for-bit** over the same
   query rows. Extends `inferno-kernels/tests/rig.rs`'s attention section
   (which today drives `attn_kernel_scalar`/`attn_kernel_avx2` for one
   token) with a block driver and an exact-equality property test.
2. **Scalar == AVX2** for the query-blocked kernel, bit-for-bit
   (`attention_isa_variants_bitwise_equal` extended to the block kernel).
3. **`_hspan` head-range identity preserved** — Lever 1 touches the query
   axis, not the head axis; the M4b.11 head-span kernels and their
   identity tests are untouched and must stay green.
4. **Cross-thread & cross-`prefill_tile` identity.** `run_attn_span`
   sharding the query axis across any lane count reproduces the
   single-lane result bit-for-bit (the pool `par_rig.rs` /
   `par_attention_fallback.rs` fallback tests extended to the block
   kernel); tiling the prefill query rows across any `prefill_tile`
   boundary is bit-neutral.
5. **`qblock(mb=1) == per_token == gemv/decode attention`** — the block-1
   reduction preserves the standing `gemm(m=1)`-style identity so decode
   and the `m == 1` prefill special-case are unaffected.

**Tolerance discipline.** `attn_rel_tol()`
(`inferno-graph/src/tolerance.rs`) is **not touched** for Lever 1 (the
kernel is bit-identical to the current one the tolerance was armed
against, so the compiled-vs-interpreter differential must stay green with
zero change). It is re-derived from an `observed_error` sweep **only** if
the gate authorizes the flash-attention Lever 2 — and then from data (the
fused kernel's measured error distribution), never loosened to make a red
test green (the standing `LOGIT_TIE_EPSILON`/`gemv_rel_tol` rule).

**Differential gates that must stay green (no tolerance loosening):**
`cargo test -p inferno-codegen --test differential`,
`cargo test -p inferno-core --test artifact`, and `mise run differential`
(per `AGENTS.md`).

## Quiet-HW Gate Script & Task Ladder

**New gate script** `scripts/quiet-hw/gate-prefill-attn-split.sh`,
mirroring `gate-prefill-attr.sh`: runs the criterion model at `--threads 1
--profile`, prints the t=1 prefill op table **with the attention bracket
split into `attn:scores` / `attn:softmax` / `attn:output`**, and leaves
the ceiling/blame arithmetic HUMAN (pasted into this spec's §Amendments,
computed there per the gate rule). The `attn:*` sub-bracket instrument is
the prefill-side analogue of M4b.12's `pool-profile` dispatch-split —
built behind a cargo feature / profile flag so it is **off in every
shipping and bench build** and only the gate scripts enable it. `pp_ratio`
comes from `gate-bench-protocol.sh` in the same session (unchanged).

**Two-box protocol** (M4b.12/M4b.13): 16c `d2.c1.medium` (6336Y) + 8c
`s2.c2.medium` (E-2388G), quiet-hw runbook, devenv shell, release build,
no parallel PNAP provisions, `metal-gc` to zero after every session (per
memory: metal provisioning quirks).

**Task ladder:**

1. **Lever 1 kernel** — query-blocked `inferno_attention_f32_{scalar,avx2}`
   block entry + rig identity tests (invariants 1–2, 5). Land two-pass
   shape first; register-block the scores pass as a guarded follow-on iff
   the local µbench shows load-port bind.
2. **Pool plumbing** — `run_attn_span` one-call-per-shard; `AttnJob`
   scratch resize; fallback/`par_rig` identity tests (invariant 4).
   `HOST_ABI_VERSION` bump iff the `inferno_par_attention` signature moved
   (decided here, recorded).
3. **Differential + full test green** — codegen differential, artifact
   differential, `mise run differential`, no `attn_rel_tol` change.
4. **Gate-script** — `gate-prefill-attn-split.sh` + the `attn:*`
   sub-bracket instrument (feature-gated, off in shipping/bench).
5. **Local dev gate** — non-quiet interleaved µbench + t=1 pp delta
   (honestly labeled non-quiet, M4b.13 Task-3 precedent) to unblock metal
   spend on evidence, not hope.
6. **Mid-milestone quiet-hw gate** — both boxes: fresh split-bracket
   profile + `pp_ratio`; compute `attn_share`, the ceiling check, and the
   blame gate; record the verdict (authorize one Lever-2 menu entry, or
   STOP).
7. **Lever 2** — built **only** if Task 6 authorized it; the chosen menu
   entry, with its numerics discipline (bit-identical → no tolerance
   change; flash → data-armed `attn_rel_tol` re-derivation). If Task 6
   STOPs, Tasks 7–8 do not run (M4b.13 precedent: an all-STOP with the
   finding is a successful outcome).
8. **Closing quiet-hw session + exit-criteria walk** — both boxes:
   closing pp vs llama best-of, the four exit criteria walked, verdict
   recorded (criterion MET, or STOP-out diagnostic with the residual-gap
   finding). tg is context-only, never the gate.

## Risks

- **Query blocking may be memory-bound end-to-end, not kernel-bound.** If
  the KV region is streamed once per block but the block is still large
  relative to L2, the once-per-block stream can itself be the bind and the
  gain is smaller than the refetch arithmetic suggests. Mitigation: the
  local µbench (Task 5) measures the per-thread attention kernel in
  isolation before any metal spend; the mid-milestone gate's sub-bracket
  profile confirms where the residual sits before Lever 2.
- **Two-pass scratch growth.** `mb * visible` f32 per lane can be large at
  long context; if it spills L2 it undercuts the locality win. Mitigation:
  `mb` is a tuning parameter, not fixed; the register-blocked scores pass
  avoids materializing the full block score matrix. Bounded by the rig's
  identity test regardless of `mb`.
- **One attention lever is unlikely to carry the 8c criterion.** The
  ladder arithmetic shows the 8c box needs ~82% of the attention bracket
  removed to reach pp 1.0x; a single KV-refetch bandwidth lever is very
  unlikely to reach that alone, so the milestone's honest exit may be a
  STOP-out diagnostic. This is a recorded, sanctioned outcome
  (M4b.12/M4b.13), not a failure — but it means pp 1.0x on the 8c box may
  remain open after M4b.14, scoping a future combined-lever or
  attention-as-GEMM milestone.
- **Sub-bracket instrument perturbation.** Splitting the attention bracket
  into scores/softmax/output adds `rdtsc` reads inside the hot kernel.
  Mitigation: feature-gated off in shipping/bench builds (M4b.12
  `pool-profile` precedent); the gate session's admissibility check
  (sum-identity + perturbation bound, M4b.12) confirms the instrument
  didn't move the measurement it reports.

## Amendments

_(quiet-hw session records, gate verdicts, and the closing exit-criteria
walk are appended here as the milestone runs — data points recorded
verbatim, never edited; ceiling/blame arithmetic shown; per the standing
`inferno bench` manual-protocol rule.)_

### Local dev data point (Task 7) — 2026-07-17, NON-QUIET shared devpod (24c, loadavg 8.7–9.4)

**Never a quiet-hw verdict; local gate input only (M4b.13 Task-3 precedent).**

µbench (criterion, `benches/attention.rs`, per_token AVX2 loop vs one qblock
call, 14h/2kv/hd64, m_block=64; ratio = per_token/qblock point estimates,
commit 6a95aef):
- pos0=64: 0.915 · pos0=256: 0.952 · pos0=512: 1.054 · geomean 0.972
- 4 repeats: only pos0=512 consistently favors qblock; box noise ±10%;
  one heavily-loaded run excluded (uniform regression both variants).
  Kernel-level µbench alone: inconclusive at short visible lengths.

t=1 end-to-end prefill (pinned binaries base@35c51bd vs qblock@3789eeb,
interleaved 3 rounds, same 2048-random-bytes-base64 prompt (2732 chars),
`--max-tokens 32 --threads 1 --profile`, criterion model):

| round | base wall/cyc | base attn cyc (share) | qblock wall/cyc | qblock attn cyc (share) |
|---|---|---|---|---|
| r1 | 35.369s / 109,639,783,449 | 29,744,715,850 (27.1%) | 34.697s / 107,556,961,975 | 27,524,195,994 (25.6%) |
| r2 | 35.469s / 109,948,338,471 | 29,888,868,713 (27.2%) | 37.566s / 116,451,073,162 | 30,030,547,951 (25.8%) |
| r3 | 38.409s / 119,064,434,458 | 32,665,111,390 (27.4%) | 37.670s / 116,770,923,534 | 30,394,231,085 (26.0%) |

- min-of-3 total prefill: 109,639,783,449 → 107,556,961,975 cyc = **−1.9%**
- min-of-3 attention bracket: 29,744,715,850 → 27,524,195,994 cyc = **−7.5%**
- attention share dropped in EVERY round (27.1/27.2/27.4 → 25.6/25.8/26.0)
  — share is load-robust; the direction is consistent, not a noise artifact.

**LOCAL GATE: PASS (with caveat).** The end-to-end t=1 run shows a real,
round-consistent per-thread win on the blamed bracket (attention cycles
−5–7%, total −1.9% min-of-3). Caveat recorded honestly: the kernel µbench
geomean is 0.972 (win only at pos0=512) — the win grows with visible
length and is modest overall; quiet-hw Task 8 owns the real verdict.
Metal spend unblocked per the pre-registered rule.

### 2026-07-17 — mid-milestone gate sessions (Lever 1 on both boxes)

Both sessions bench branch `m4b14-qblock` at `83c183f` (Tasks 1–7;
`git_dirty=true` = untracked `models/` + SDD ledger only). Protocol:
`mise run metal` per `docs/runbooks/metal.md`; on-box `verify.sh --smoke`
first, then the real pass (preflight + the two gates standalone); preflight
FIT both boxes. Session A raw: `target/metal/d2.c1.medium-20260717T201908Z`
(real pass `.../quiet-hw/20260717T205013Z`). Session A trap delete did not
stick (M4b.13 recurrence); gc hit 409s then a 403 but the delete landed —
second gc confirmed zero before Session B. Session B (CHI; PHX has no
s2.c2.medium stock): raw `target/metal/s2.c2.medium-20260717T210046Z`
(real pass `.../quiet-hw/20260717T212433Z`); same delete-403-then-clear
pattern, zero servers confirmed twice after.

#### Session A — d2.c1.medium (Xeon Gold 6336Y, 16c, PHX): gate-prefill-attn-split.out

```
# gate-prefill-attn-split (M4b.14: attn scores/softmax/output sub-brackets) — 2026-07-17T20:50:23Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

--- t=1 prefill profile + attn sub-brackets ---
profile [prefill] 33.148s wall, 102756652522 cyc total
  op                                   cycles   share        GB/s
  attention                       24325232698   23.7%           -
  matmul:lm_head.weight           19641736136   19.1%        49.3
  matmul:layers.*.ffn.gate_proj.weight    15016429648   14.6%        49.6
  matmul:layers.*.ffn.up_proj.weight    15006980710   14.6%        49.6
  matmul:layers.*.ffn.down_proj.weight    14952208820   14.6%        49.8
  swiglu                           4292050616    4.2%           -
  matmul:layers.*.attn.q_proj.weight     2765300442    2.7%        49.6
  matmul:layers.*.attn.o_proj.weight     2761605764    2.7%        49.6
  rmsnorm                          1196825042    1.2%           -
  rope                              869086750    0.8%           -
  quantize                          530282878    0.5%           -
  matmul:layers.*.attn.k_proj.weight      397887570    0.4%        49.2
  matmul:layers.*.attn.v_proj.weight      397068830    0.4%        49.3
  add                               335778160    0.3%           -
  bias                              134280006    0.1%           -
  kv_append                         115689000    0.1%           -
  embed                              18209452    0.0%           -
profile [decode] 1.939s wall, 5963657248 cyc total
  op                                   cycles   share        GB/s
  attention                        2040674806   34.2%           -
  matmul:lm_head.weight            1043891364   17.5%        14.4
  matmul:layers.*.ffn.down_proj.weight      806462214   13.5%        14.4
  matmul:layers.*.ffn.up_proj.weight      802884778   13.5%        14.4
  matmul:layers.*.ffn.gate_proj.weight      802215154   13.5%        14.4
  matmul:layers.*.attn.o_proj.weight      151065526    2.5%        14.1
  matmul:layers.*.attn.q_proj.weight      149382282    2.5%        14.3
  swiglu                             68285562    1.1%           -
  matmul:layers.*.attn.k_proj.weight       22027908    0.4%        13.8
  matmul:layers.*.attn.v_proj.weight       21857178    0.4%        13.9
  rope                               19923346    0.3%           -
  rmsnorm                            19468036    0.3%           -
  add                                 8873922    0.1%           -
  bias                                5891304    0.1%           -
  embed                                556092    0.0%           -
  quantize                             197776    0.0%           -
  kv_append                                 0    0.0%           -
attn [prefill sub-brackets] 24301091998 cyc instrumented
attn:scores      11316917634
attn:softmax      2032174764
attn:output      10951999600

--- attn sub-brackets (grep) ---
attn:scores      11316917634
attn:softmax      2032174764
attn:output      10951999600
```

#### Session A — gate-bench-protocol.out

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-17T20:51:04Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (16 physical / 32 logical cores)
inferno 0.1.0 (83c183f) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      911.10 ± 32.19       58.20 ± 0.50 
inferno (t=1 diag)           1       69.99 ± 0.04        21.93 ± 0.00 
llama.cpp                   16     1207.63 ± 251.80       59.49 ± 0.40 
llama.cpp (t=1 diag)         1      118.55 ± 0.08        23.13 ± 0.09 

ratio (inferno/llama.cpp): pp 0.75x | tg 0.98x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 513.42 | tg 61.99 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.79x | tg 0.94x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

#### Session B — s2.c2.medium (Xeon E-2388G, 8c, CHI): gate-prefill-attn-split.out

```
# gate-prefill-attn-split (M4b.14: attn scores/softmax/output sub-brackets) — 2026-07-17T21:24:43Z
machine: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (GenuineIntel) | 16 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

--- t=1 prefill profile + attn sub-brackets ---
profile [prefill] 23.196s wall, 74041128516 cyc total
  op                                   cycles   share        GB/s
  attention                       18525303066   25.0%           -
  matmul:lm_head.weight           14070624920   19.0%        70.0
  matmul:layers.*.ffn.up_proj.weight    10763919578   14.5%        70.3
  matmul:layers.*.ffn.down_proj.weight    10750665936   14.5%        70.4
  matmul:layers.*.ffn.gate_proj.weight    10706141898   14.5%        70.7
  swiglu                           2797747810    3.8%           -
  matmul:layers.*.attn.q_proj.weight     1989105443    2.7%        70.1
  matmul:layers.*.attn.o_proj.weight     1963575765    2.7%        71.0
  rmsnorm                           703591821    1.0%           -
  rope                              437476435    0.6%           -
  quantize                          385024603    0.5%           -
  matmul:layers.*.attn.v_proj.weight      283951926    0.4%        70.1
  matmul:layers.*.attn.k_proj.weight      282957506    0.4%        70.4
  add                               210382660    0.3%           -
  kv_append                          83579362    0.1%           -
  bias                               79132937    0.1%           -
  embed                               7946850    0.0%           -
profile [decode] 1.219s wall, 3870785562 cyc total
  op                                   cycles   share        GB/s
  attention                        1218812574   31.5%           -
  matmul:lm_head.weight             707337838   18.3%        22.0
  matmul:layers.*.ffn.down_proj.weight      546167282   14.1%        21.9
  matmul:layers.*.ffn.gate_proj.weight      544964554   14.1%        21.9
  matmul:layers.*.ffn.up_proj.weight      543611279   14.0%        22.0
  matmul:layers.*.attn.q_proj.weight      101689087    2.6%        21.7
  matmul:layers.*.attn.o_proj.weight      100533687    2.6%        21.9
  swiglu                             47517422    1.2%           -
  matmul:layers.*.attn.v_proj.weight       15139739    0.4%        20.8
  matmul:layers.*.attn.k_proj.weight       15054783    0.4%        20.9
  rope                               13463784    0.3%           -
  rmsnorm                            11217539    0.3%           -
  add                                 3004174    0.1%           -
  bias                                2027484    0.1%           -
  quantize                             158438    0.0%           -
  embed                                 85898    0.0%           -
  kv_append                                 0    0.0%           -
attn [prefill sub-brackets] 18510337643 cyc instrumented
attn:scores       8122310464
attn:softmax      1441213287
attn:output       8946813892

--- attn sub-brackets (grep) ---
attn:scores       8122310464
attn:softmax      1441213287
attn:output       8946813892
```

#### Session B — gate-bench-protocol.out

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-17T21:25:11Z
machine: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (GenuineIntel) | 16 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (8 physical / 16 logical cores)
inferno 0.1.0 (83c183f) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)           8      738.57 ± 3.92        62.36 ± 0.10 
inferno (t=1 diag)           1      107.97 ± 0.04        34.58 ± 0.02 
llama.cpp                    8     1042.85 ± 2.16        72.77 ± 0.02 
llama.cpp (t=1 diag)         1      165.44 ± 0.71        33.72 ± 0.03 

ratio (inferno/llama.cpp): pp 0.71x | tg 0.86x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 628.80 | tg 72.56 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.70x | tg 0.86x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

#### Gate verdict (pre-registered rule, human-computed)

**Instrument admissibility** (sum of `attn:*` sub-brackets vs the whole
`attention` bracket, same run):

- A: (11,316,917,634 + 2,032,174,764 + 10,951,999,600) = 24,301,091,998 /
  24,325,232,698 = **99.90%** — admissible.
- B: (8,122,310,464 + 1,441,213,287 + 8,946,813,892) = 18,510,337,643 /
  18,525,303,066 = **99.92%** — admissible.

**Step 1 — attn_share:**

- A: 24,325,232,698 / 102,756,652,522 = **0.2367** (was 35.5% at M4b.13's
  sessions on the same box type — Lever 1 cut the bracket 43.76 → 24.33 Gcyc,
  −44.4%, and total t=1 prefill 123.45 → 102.76 Gcyc, −16.8%).
- B: 18,525,303,066 / 74,041,128,516 = **0.2502** (was 36.5%; total
  90.78 → 74.04 Gcyc, −18.4%).

**Step 2 — pp_ratio (post-Lever-1, vs llama best-of):** A **0.79x**
(was 0.74x), B **0.70x** (was 0.70x — t=8 pp512 738.57 vs BLAS-build 1042.85;
the t=1 diagnostic improved 1.04x→relative but the t=8 ratio is the gate).

**Step 3 — ceiling checks** (`pp_ratio / (1 - f*c) >= 1.0`), sub-bracket
fractions of prefill total — A: scores 0.11013, softmax 0.01978, output
0.10658 (scores+output 0.21672); B: scores 0.10970, softmax 0.01947,
output 0.12084 (scores+output 0.23054):

| menu entry | f (A) | c | A: pp/(1−f·c) | f (B) | B: pp/(1−f·c) | verdict |
|---|---|---|---|---|---|---|
| flash online-softmax fused pass | 0.21672 | ½ | 0.79/0.89164 = **0.886** | 0.23054 | 0.70/0.88473 = **0.791** | STOP both |
| vectorized/batched softmax | 0.01978 | even c=1 | 0.79/0.98022 = **0.806** | 0.01947 | 0.70/0.98053 = **0.714** | STOP both (blame gate also fails — softmax is the smallest sub-bracket) |
| wider blocks / KV-panel prefetch | 0.21672 | modest < ½ | bounded above by the flash row | 0.23054 | bounded above by the flash row | STOP both |

Upper bound for ANY attention-only lever (whole bracket, impossible c=1):
A 0.79/0.76327 = 1.035; B 0.70/0.74980 = **0.934 < 1.0** — on the 8c box
no attention lever, even one that deletes the entire bracket, reaches the
criterion. This is the same rule-3 shape that kept M4b.13's VNNI unbuilt.

**VERDICT: all-STOP. No Lever-2 menu entry is authorized on either box
(rule 2 fails everywhere; rule 3 bound fails outright on the 8c box).
Task 9 is skipped per the pre-registered rule. All-STOP is a successful
diagnostic outcome (M4b.12/M4b.13 precedent).**

**Finding (for the next milestone's scoping):** after query blocking, the
t=1 prefill residual is matmul-shaped again — matmul:* sums to ~66.4% (A)
/ ~67.3% (B) of prefill at memory-stream rates (~49.6 / ~70.3 GB/s), with
attention at 23.7% / 25.0% (split ~evenly between scores and output;
softmax ~2%) and lm_head alone at 19.1% / 19.0%. With GEMM exhausted
(M4b.13) and attention now below the reach of any single-bracket lever on
the 8c box, closing pp ≥ 1.0x there requires either a cross-bracket lever
(e.g. quantized KV / attention-as-GEMM — both explicitly out of scope
here) or accepting the 16c box as the criterion machine. Recorded as a
diagnostic; no further prefill-attention lever in this milestone.

### Closing verdict: exit-criteria walk (2026-07-17)

The Task 8 sessions are the closing protocol runs (M4b.13 precedent —
same commit `83c183f`, same day, same `gate-bench-protocol.sh`; no new
provision).

1. **Local dev data point recorded (Task 7)?** YES — µbench + interleaved
   t=1 pinned-binary comparison above, honestly labeled non-quiet; local
   gate PASS with the µbench caveat recorded.
2. **Fresh split-bracket profiles + gate verdict with arithmetic
   recorded (Task 8)?** YES — both boxes FIT, instrument admissibility
   99.90% / 99.92%, ceiling + blame arithmetic shown verbatim above.
3. **Every gate outcome recorded; no lever shipped without its gate?**
   YES — Lever 1 shipped only after the Task 7 local gate PASS; Lever 2
   was NOT built (all-STOP; Task 9 skipped per the pre-registered rule).
4. **Closing pp512 vs llama best-of ≥ 1.0x on both boxes?** **NO** —
   0.79x (16c) / 0.70x (8c). **v1 pp criterion NOT MET; milestone closes
   as a diagnostic (M4b.12/M4b.13 precedent).** STOP finding: the t=1
   prefill residual is matmul-shaped again (matmul:* ≈ 66.4% / 67.3% at
   memory-stream rates ~49.6 / ~70.3 GB/s; lm_head alone ~19% both), with
   attention reduced to 23.7% / 25.0% (scores ≈ output, softmax ~2%). On
   the 8c box even deleting the entire attention bracket reaches only
   0.934 — the remaining pp gap is not closable from inside any single
   prefill bracket; the next lever must be cross-bracket (quantized KV,
   attention-as-GEMM — both out of scope here) or the criterion machine
   question must be revisited. tg is context only (0.94x / 0.86x), never
   the gate.

Lever 1 (query-blocked prefill attention, bit-identical) SHIPPED: t=1
prefill total −16.8% (16c) / −18.4% (8c), attention bracket −44%, pp
0.74x→0.79x on the 16c box, with zero tolerance change (`attn_rel_tol`
untouched; differential green throughout).

#### Erratum (2026-07-17, pre-merge review): matmul-share figures in the Finding

The gate verdict's Finding paragraph and closing-verdict item 4 state
"matmul:* ≈ 66.4% (A) / 67.3% (B)". Summing the recorded tables above gives
the correct figures: **A 70,939,217,920 / 102,756,652,522 = 69.0%; B
50,810,942,972 / 74,041,128,516 = 68.6%**. The recorded tables and every
pre-registered-rule input (admissibility, attn_share, ceiling rows, bounds)
are unaffected — matmul share feeds no rule — and the corrected figures make
the "residual is matmul-shaped" conclusion slightly stronger. Recorded text
above is left unedited per the standing rule; this erratum is the correction
of record.
