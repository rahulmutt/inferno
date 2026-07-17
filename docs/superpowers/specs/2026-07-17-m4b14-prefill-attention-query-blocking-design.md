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
