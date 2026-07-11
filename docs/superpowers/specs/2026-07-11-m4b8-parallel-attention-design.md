# M4b.8 — Parallel Prefill Attention Design

**Date:** 2026-07-11
**Status:** Approved design, pre-implementation
**Milestone:** M4b.8 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; the amendment authorized by
[M4b.1](2026-07-06-m4b1-threading-design.md)'s attribution fork)

This milestone takes up the follow-up the M4b.1 gate clause reserved:
"if pp scaling stalls below 6x and the profile blames serial attention,
parallel attention becomes a scoped follow-up task — an explicit spec
amendment, not silent scope growth." Both conditions are now met and
recorded in the M4b.1 Amendments (2026-07-11, second quiet-hw session):
prefill scale @ t=12 = **4.11x against the ≥6x gate (NOT MET,
reproduced)**, and the attribution fork was **taken toward the serial
fraction** — llama.cpp's pure-CPU build scales pp ~9.4x @ t=12 on the
same silicon in the same session, eliminating the memory-bandwidth
branch. This spec is that authorized amendment, given its own document
per repo convention (M4b.2–M4b.7 precedent); the M4b.1 Amendments carry
a pointer here.

## Motivation — the Amdahl arithmetic

Since M4b.3, prefill attention is a single-threaded C-ABI kernel call:
`lower_tile`'s `Step::Attention` arm wraps KV-append plus
`inferno_attention_f32_{isa}` in a serial `range_loop(m)` over the
tile's tokens (`crates/inferno-codegen/src/llvm/ops.rs`), and the
kernel loops heads serially inside each call. Matmuls meanwhile go
through `inferno_par_gemm`. The M4b.2/M4b.3 profiles put attention at
**68.5–70.6% of prefill cycles**; with ~70% of the work serial, Amdahl
caps prefill scaling almost exactly where both quiet-hw sessions
measured it (~4.1x @ t=12), and it is a slope deficit from t=2 onward —
not a high-t tail-off — which is the serial-fraction signature.

Parallelizing attention leaves rope, KV-append, norms and the other
small ops serial (the "~5–10% combined" from the M4b.1 gate clause).
At s ≈ 0.08 the predicted scale @ t=12 is ~6.4x: the gate is
**clearable but tight**, so the design keeps the remaining serial
fraction from growing (the KV-append split below adds no new math) and
the verdict protocol distinguishes "remaining serial ops" from
"dispatch overhead" if the gate still misses.

## Scope Decisions (M4b.8)

| Decision | Choice |
|---|---|
| Phase | **Prefill only.** Decode attention stays serial: its bandwidth branch was never eliminated (M4b.2's decode fork is still open) and the authorized gate is prefill scaling. A decode-attention lever needs its own attribution first |
| Parallel axis | **Tile tokens** (approach A of the brainstorm). One pool dispatch per tile shards the tile's `m` tokens (`PREFILL_TILE = 64` default) across threads; each worker calls the existing per-token kernel over its token sub-range. Parallelism = min(m, threads) — ample for the 12–16 threads under test. Head-axis sharding (finer, but a kernel-ABI change) is a gated follow-up, not silent scope |
| Kernel ABI | **Unchanged.** `inferno_attention_f32_{scalar,avx2}` are not modified; scalar↔AVX2 bit-identity and the `attn_rel_tol` derivation carry over untouched. No `HOST_ABI_VERSION` bump (no numeric change) |
| Pool entry | New `#[no_mangle] inferno_par_attention` in `inferno-pool`, mirroring `inferno_par_gemm`: same `DISPATCH_CLAIMED` CAS guard, same serial full-range fallback when the pool is uninitialized or the claim is lost. New `JobKind::Attention` carrying the per-token kernel fn pointer (ISA chosen at lowering time, as GEMM does) plus base pointers/strides; `run_shard` loops its token sub-range computing per-token args |
| Thread budget | `active_threads`, uncapped — attention here is prefill work. The M4b.5 decode cap (`decode_threads`) does **not** apply |
| Sharding | **Align-1 contiguous token shards**, via an alignment parameter on `shard_table` (GEMM call sites keep `SHARD_ALIGN = 8`; one sharding function, no fork). Load-bearing: reusing `shard_table`'s 8-row `SHARD_ALIGN` would split m=64 into at most 8 shards and silently cap attention at 8 threads on a 12-thread run |
| m == 1 guard | `inferno_par_attention` calls the kernel directly (no CAS, no job publish) when `m == 1`, so decode-shaped calls never touch the pool — zero new decode overhead by construction |
| KV-append split | `lower_tile`'s attention arm becomes: serial `range_loop(m)` writing the whole tile's K/V into the cache, **then** one `inferno_par_attention(m, …)` call. Bit-safe: token *i*'s causal loop reads KV rows `0..=pos_i` only, so rows written for later tile tokens are never read. The append stays serial — it is m × kv_dim stores, negligible |
| Interpreter | **Untouched.** `inferno-graph`'s serial attention remains the reference oracle |
| Tolerances | **None touched.** `attn_rel_tol`, `logits_abs_tol`, `gemv_rel_tol` unchanged; every differential must pass as-is. Sharding must be bit-neutral (each (token, head) output is computed by exactly one thread with unchanged kernel math) |
| Measurement discipline | Same-box ratios only on the devpod; the formal verdict comes from quiet bare metal via `mise run metal` (standing M4b discipline) |

**Explicitly out of scope:**

- **Decode attention parallelism** — needs its own attribution fork
  (bandwidth vs serial) before any lever is authorized.
- **Head-axis / flattened (token × head) sharding** — a kernel-ABI
  change; becomes a gated follow-up only if the verdict shows a
  granularity or tail problem token sharding can't fix.
- **Flash-style query-panel blocking** — the other M4b.3 deferral;
  a locality lever, not a threading lever.
- **Cost-weighted shard balancing** — token cost grows with position,
  so early tiles are imbalanced, but early tiles are also the cheapest;
  contiguous shards are accepted and the risk is recorded below.
- **Per-format PF_DIST (M4b.4) and the M4b.5 cap-formula revision** —
  separate authorized threads, unchanged by this work.

## Design

### inferno-pool

- `JobKind::Attention { kernel, q, kv, out, pos0, m, dims/strides }`
  alongside `Gemv`/`Gemm` (`pool.rs`); `run_shard` gains the matching
  arm: for each token in the shard's sub-range, compute the per-token
  `q` row / `out` row / `pos` and call the kernel — the same serial
  per-token call the current `range_loop(m)` makes, just partitioned.
- `Pool::par_attention` mirrors `par_gemm` (publish job, bump epoch,
  run shard 0 on the dispatcher, spin on `remaining`), sharding
  `0..m` with align-1 contiguous shards.
- `inferno_par_attention` (C ABI, `lib.rs`): `m == 1` → direct kernel
  call; otherwise CAS `DISPATCH_CLAIMED`, fall back to the serial
  full-range loop when uninitialized or the claim is lost — exactly
  the `par_gemm` shape.

### inferno-codegen

- `lower_tile`'s `Step::Attention` arm splits into the KV-append loop
  followed by one call to `inferno_par_attention` (declared alongside
  the existing pool externs in `llvm/mod.rs`), passing the ISA-selected
  `inferno_attention_f32_{isa}` fn pointer plus the same base
  pointers/dims the serial path computes today.
- Every other `lower_tile` arm, `lower_rope`, and the decode path are
  untouched.

### Invariants (all inherited, none loosened)

1. **Thread count never changes output bits** — extends to the new
   dispatcher, same as `shard_table` guarantees for GEMM.
2. **Tiling bit-gate** — bitwise-identical prefill logits across
   `prefill_tile` sizes must survive the KV-append split.
3. **Scalar↔AVX2 attention bit-identity** — untouched kernels.
4. **No tolerance loosening** — compiled-vs-interpreter
   (`inferno-codegen`) and artifact (`inferno-core`) differentials
   green with existing bounds.

## Testing plan

- **inferno-pool unit tests** for `JobKind::Attention`: sharded output
  bit-equal to the serial fallback across awkward shapes — `m ∈ {1, 7,
  63, 64}`, threads > m, threads = 1; plus align-1 shard-table cases.
- **Threads bit-gate** in the codegen differential suite: prefill the
  fixture at pool t=1 vs t=8, assert bitwise-identical logits (the
  thread-axis analogue of the existing tiling gate).
- **m == 1 guard test**: decode-shaped calls never claim the dispatch.
- **Existing gates**: `cargo test -p inferno-codegen --test
  differential` and `cargo test -p inferno-core --test artifact` green
  with no tolerance edits (AGENTS.md standing rule).

## Verification protocol and verdict gate

1. **`mise run bench-compiled` stays green.** It is pinned
   `--threads 1` on purpose; here it doubles as the check that the
   KV-append restructure does not regress single-thread codegen
   quality.
2. **Quiet-hw verdict** via `mise run metal` (d2.c1.medium class,
   PREFLIGHT FIT), re-running the M4b.7 `gate-prefill-scaling`
   protocol: **prefill scale @ t=12 ≥ 6x → MET**, recorded as
   amendments in the M4b.1 spec (the verdict ledger) and here. Also
   re-record the M4a headline row in the same session — pp vs
   llama.cpp best-of should move materially off 0.23x.
3. **NOT MET →** record the verdict plus a fresh attribution before
   any further lever: the sweep's shape distinguishes the candidates
   (remaining serial ops → slope deficit persists across t; dispatch
   overhead → low-t degradation with recovery at high t). No silent
   scope growth.

## Risks

- **Remaining serial fraction is bigger than the gate clause's
  5–10%.** The Amdahl margin at 6x is thin (~6.4x predicted). If the
  gate misses with a persistent slope deficit, the recorded
  attribution points at rope/norm/append parallelization as the next
  authorized lever — not at loosening the gate.
- **Dispatch overhead per tile per layer.** One pool job per tile per
  layer, same order as existing GEMM dispatches; the `m == 1` guard
  keeps decode at zero. Visible as low-t degradation in the sweep if
  it matters.
- **Early-tile load imbalance.** Token cost ∝ position + 1, so the
  first tiles split unevenly under contiguous shards; those tiles are
  also the cheapest, so the effect is bounded. Accepted; cost-weighted
  sharding stays out of scope unless the verdict blames it.
