# M4b.9 — Serial-Tail Parallelization Design

**Date:** 2026-07-12
**Status:** Approved design, pre-implementation
**Milestone:** M4b.9 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; the amendment authorized by the
[M4b.8](2026-07-11-m4b8-parallel-attention-design.md) NOT-MET attribution,
recorded in the [M4b.1](2026-07-06-m4b1-threading-design.md) verdict ledger)

This milestone takes up the lever the M4b.8 verdict authorized. The
third quiet-hw session (2026-07-12, d2.c1.medium → Xeon Gold 6336Y,
PREFLIGHT FIT, inferno @ 823437f) recorded **prefill scale @ t=12 =
5.67x against the ≥6x gate — NOT MET**, with the attribution fork taken
both ways: memory bandwidth ruled out (llama.cpp's pure-CPU control
scales 9.2x on the same box), and the sweep's persistent slope deficit
identified as the remaining-serial-ops signature. Per the M4b.8 Risks
clause, the authorized next lever is parallelizing the remaining serial
prefill ops — not loosening the gate, which stays owned by the M4b.1
spec.

## Motivation — the Amdahl arithmetic, and a measurement gap

The Amdahl fit from the third session puts the residual serial fraction
at **≈ 10.2%**; the 6x line needs **≈ 9.1%**. The ops the verdict names
(rope, norms, append) are not all measurable in the recorded profiles:

- **rope 0.2%, rmsnorm 0.2%, add 0.1%, swiglu 0.6%, bias/embed ~0%**
  (M4b.3 profile, % of total t=1 prefill cycles) — the *named* small
  ops sum to ~1.1pp, which would land the serial fraction exactly on
  the 9.1% line with **zero margin**.
- **KV-append has never had its own profile line** — it is lowered
  inside the attention step's bracket, so its serial cost hides in the
  70.6% attention figure. Post-M4b.8 the attention *read* is parallel
  but the append loop is not.
- **Activation quantization has never had its own profile line
  either** — the panel fill is folded into `lower_gemm` inside the
  matmul bracket, and it runs serially before every `inferno_par_gemm`
  dispatch (7 per layer per tile on quantized weights).

So the residual ≈ 10.2% is composed of: serial KV-append + serial
quantize panel fill + the measured ~1.1pp of small ops + dispatch/join
overhead — with the split between them unknown. This design therefore
(a) parallelizes **all** serial per-token tile work, not just the three
named ops, so the lever does not depend on the unknown split, and
(b) **splits the profile brackets** (kv_append and quantize get their
own labels) so the next verdict can attribute the residual directly
instead of by sweep-shape inference.

Structurally the work is favorable: every op in scope is already a
per-token `range_loop(m)` over the tile with disjoint per-token outputs
and no cross-token reductions, so token-sharding is bit-neutral by the
same argument M4b.8 used for attention. What is new is that these ops
have **no C-ABI kernel** — they are inline LLVM IR, so there is no
fn-pointer boundary for the pool to dispatch. The mechanism below
creates one.

## Scope Decisions (M4b.9)

| Decision | Choice |
|---|---|
| Phase | **Prefill only.** The decode path is untouched; the dispatcher's `m <= 1` guard keeps decode-shaped calls off the pool by construction (M4b.8 precedent) |
| Op coverage | **All eight serial per-token op-sites** in the prefill tile loop: rmsnorm, rope, residual add, swiglu, bias, embed (the generic `_ =>` arm of `lower_tile`), the KV-append loop (`lower_tile_attention`), and the activation-quantize panel fill (`lower_gemm`). Chosen over the literal "rope/norm/append" because the named ops alone leave zero Amdahl margin (brainstorm decision) |
| Mechanism | **Outlined token bodies + one generic dispatcher** (approach A of the brainstorm). Codegen outlines each op-site's per-token loop body into an emitted `extern "C"` function `body(ctx, t0, t1)`; one new pool entry `inferno_par_token_loop(body, ctx, m)` shards `0..m` and calls the body per shard span. Per-op kernels in `inferno-kernels` (approach B) rejected: 6–8 new ABIs plus bit-exactness risk reimplementing inline-IR math (SiLU's `exp`, FMA contraction) against the interpreter differential. Job fusion (approach C) reserved as the fallback lever |
| Body ABI | `unsafe extern "C" fn(ctx: *const u8, t0: usize, t1: usize)`. **ctx is opaque to the pool** — only codegen knows its layout (`{tokens, pos_off, weights, kv, arena, tile_start}`, rebuilt on the stack per tile, ~6 stores). The body loops its span, deriving `row = tile_start + ti` and the existing `Frame` exactly as the inline loop does today; the op math IR is unchanged |
| Kernel ABI | `inferno-kernels` untouched — no new kernels, no new bit-identity rigs. `HOST_ABI_VERSION` bumps 5→6: numerics don't change, but the emitted code's host-call shape gains `inferno_par_token_loop` ("bump whenever the host-call shape changes", M4b.1/2/3/8 precedent) |
| Pool entry | New `#[no_mangle] inferno_par_token_loop` in `inferno-pool`, mirroring `inferno_par_attention`: `m == 0` no-op, `m == 1` direct body call (no CAS, no publish), same `DISPATCH_CLAIMED` CAS guard, same serial full-range fallback when the pool is uninitialized or the claim is lost. New `JobKind::TokenLoop { body, ctx, m }`; `run_shard` calls `body(ctx, span.start, span.end)` |
| Thread budget | `active_threads`, uncapped — prefill work; the M4b.5 decode cap does **not** apply (M4b.8 precedent) |
| Sharding | **Align-1 contiguous token shards** via the existing `shard_table_aligned` — identical to `par_attention`. No new sharding code |
| Ordering | The tile's KV-append dispatch **joins before** the attention-read dispatch is issued. Pool dispatches are synchronous joins, so the existing append-then-read sequencing holds for free; token *i*'s causal read still never reaches KV rows past `pos_i` |
| Profiling | The append dispatch moves out of the attention bracket into its own **`kv_append`** label, and the panel-fill dispatch moves out of the matmul bracket into its own **`quantize`** label (sequential brackets, not nested). This intentionally changes profile-table continuity: attention and matmul lines shrink by their split-out shares from this milestone on |
| Interpreter | **Untouched.** `inferno-graph` remains the reference oracle |
| Tolerances | **None touched.** Every differential must pass as-is; sharding must be bit-neutral (each token's row is computed by exactly one lane running the unchanged body IR) |
| Measurement discipline | Same-box ratios only on the devpod; the formal verdict comes from quiet bare metal via `mise run metal` (standing M4b discipline) |

**Explicitly out of scope:**

- **Decode-path parallelism of any kind** — decode attention's
  attribution fork (M4b.2) remains open; nothing here touches decode.
- **Logits copy-out** — a single row per prefill call; nothing to
  shard.
- **Per-op SIMD kernels** (approach B) — a numerics-bearing change
  with its own rig burden; only worth it if a profile someday shows
  these ops dominating *parallel* time.
- **Multi-phase job fusion** (approach C — append as a barriered
  pre-phase of the attention job, quantize as a pre-phase of gemm) —
  the pre-authorized fallback lever if the verdict blames dispatch
  overhead, not part of this scope.
- **Cost-weighted shard balancing** — unchanged from M4b.8; these ops
  are position-independent anyway (only append/rope touch `pos`, and
  their per-token cost is flat).

## Design

### inferno-pool

- `TokenBodyFn = unsafe extern "C" fn(*const u8, usize, usize)`
  (pool.rs, alongside `GemvFn`/`GemmFn`/`AttnFn`).
- `JobKind::TokenLoop { body: TokenBodyFn, ctx: *const u8, m: usize }`;
  `run_shard` gains the matching arm: one `body(ctx, start, end)` call
  per shard — the body loops its own span, so per-token call overhead
  is paid once per shard, not per token.
- `Pool::par_token_loop` mirrors `par_attention` (publish job, bump
  epoch, run shard 0 on the dispatcher, spin on `remaining`), sharding
  `0..m` with align-1 contiguous shards; `shards.len() == 1` collapses
  to a direct call.
- `inferno_par_token_loop` (C ABI, lib.rs): `m == 0` → return;
  `m == 1` → direct `body(ctx, 0, 1)`; otherwise CAS
  `DISPATCH_CLAIMED`, serial full-range fallback when uninitialized or
  the claim is lost — exactly the `par_attention` shape.

### inferno-codegen

- A new outlining helper emits, per op-site, a private module function
  containing today's `range_loop` body looped over `t0..t1`, reading
  `{tokens, pos_off, weights, kv, arena, tile_start}` from the ctx
  struct. The ctx struct type is codegen-internal; the pool never
  learns its layout.
- `lower_tile`'s generic `_ =>` arm becomes: build ctx (per tile),
  outline the step body, call `inferno_par_token_loop`.
- `lower_tile_attention`'s append loop becomes the same
  outline-plus-dispatch, wrapped in its own `kv_append` profile
  bracket, joining before the `inferno_par_attention` call.
- `lower_gemm`'s quantize panel fill becomes the same
  outline-plus-dispatch (the body calls the already-declared quantize
  kernel per row), wrapped in its own `quantize` profile bracket,
  joining before the `inferno_par_gemm` call.
- `inferno_par_token_loop` declared alongside the existing pool
  externs (`llvm/mod.rs`); added to `ensure_kernels_linked`
  (`inferno-core/src/artifact.rs`) and the differential suite's
  retention clone. `HOST_ABI_VERSION` 5→6 (`inferno-codegen/src/lib.rs`).
- The decode path and `inferno-graph` are untouched.

### Invariants (all inherited, none loosened)

1. **Thread count never changes output bits.** Every op in scope
   writes disjoint per-token rows (rmsnorm/rope/add/swiglu/bias/embed
   write arena row `t`; append writes KV row `pos0 + t`; quantize
   writes panel row `t`) and has no cross-token reduction (rmsnorm's
   reduction is within-token). Same body IR + disjoint outputs + one
   lane per token ⇒ bit-neutral, the M4b.8 argument verbatim.
2. **Tiling bit-gate** — per-token body semantics are unmodified, so
   bitwise-identical prefill logits across `prefill_tile` sizes must
   survive the restructure.
3. **Append-before-read** — preserved by the synchronous join between
   the `kv_append` dispatch and the attention dispatch.
4. **No tolerance loosening** — compiled-vs-interpreter
   (`inferno-codegen`) and artifact (`inferno-core`) differentials
   green with existing bounds.

## Testing plan

- **inferno-pool unit tests** for `JobKind::TokenLoop`, mirroring
  `attention_parallel_matches_serial_expectation`: a Rust
  `extern "C"` stub body writing a recognizable per-token pattern;
  sharded output bit-equal to the serial fallback across `m ∈ {1, 7,
  63, 64}`, threads > m, threads = 1.
- **`par_token_loop_fallback.rs`** mirroring
  `par_attention_fallback.rs`: uninitialized-pool serial fallback,
  `m == 1` direct path (no dispatch claim), `m == 0` no-op.
- **`par_rig.rs`**: extend the ABI-coercion assertions to
  `TokenBodyFn` so fn-pointer drift is a compile error.
- **Threads bit-gate** (existing, codegen differential suite): t=1 vs
  t=8 bitwise logits now also covers the sharded small ops; verify the
  fixture prefill spans multiple shards (m > 8 tile) so shard
  boundaries are actually exercised.
- **Tiling bit-gate** (existing) must stay green through the
  restructure.
- **Existing gates**: `cargo test -p inferno-codegen --test
  differential` and `cargo test -p inferno-core --test artifact` green
  with no tolerance edits (AGENTS.md standing rule).
- No new scalar↔SIMD bit-identity rigs — no new kernel math exists.

## Verification protocol and verdict gate

1. **`mise run bench-compiled` stays green.** Pinned `--threads 1` on
   purpose; here it is the guard against single-thread regression from
   outlining (per-op call overhead, lost cross-op LLVM optimization).
2. **Quiet-hw verdict** via `mise run metal` (d2.c1.medium class,
   PREFLIGHT FIT), re-running the `gate-prefill-scaling` protocol:
   **prefill scale @ t=12 ≥ 6x → MET**, recorded as amendments in the
   M4b.1 spec (the verdict ledger) and here. Re-record the M4a
   headline row in the same session. The split `kv_append`/`quantize`
   profile brackets ship **before** this run.
3. **NOT MET →** record the verdict plus a fresh attribution before
   any further lever. With the split brackets, attribution is now
   direct (read the per-label profile) rather than sweep-shape
   inference: a residual dominated by dispatch/join overhead
   authorizes job fusion (approach C) as the next lever; a residual
   still dominated by a serial op label means that op's dispatch is
   not engaging and is a bug, not a new milestone. No silent scope
   growth.

## Risks

- **Dispatch-count growth.** Added joins per layer per tile: 2 norms +
  rope + append + 2 adds + swiglu + up to 7 quantizes ≈ 8–14, roughly
  2.5x the current ~8–9. Each join costs the same publish/spin as the
  dispatches M4b.8 accepted, but the ops being sharded are far
  smaller, so the overhead ratio is worse. The low-t rows of the sweep
  are the detector (dispatch overhead = low-t degradation, high-t
  recovery), and fusion is the pre-authorized fallback lever.
- **Outlining perturbs single-thread codegen quality.** Outlined
  bodies cross a call boundary the optimizer previously saw through.
  The t=1 `bench-compiled` gate catches it; if it regresses, the
  outlined body can be marked for aggressive inlining into the serial
  fallback path or the affected op exempted.
- **The residual after this lever is join/setup overhead itself.**
  Token-sharding cannot remove per-dispatch costs. If the gate still
  misses with the serial-ops labels near zero, the next fork is
  dispatch consolidation (approach C), not more sharding — recorded
  here so it is an authorized branch, not scope growth.

## Amendments

### 2026-07-12 — quiet-hw verdict: **MET at 10.63x @ t=12**; both named risks did not materialize

Shipped @ 2387266 (PR #15, squash-merged; CI green first run). Verification
protocol item 2 executed on d2.c1.medium (Xeon Gold 6336Y, PREFLIGHT FIT) —
the full verdict and the gate table live in the
[M4b.1 spec](2026-07-06-m4b1-threading-design.md) §Amendments, the verdict
ledger. Summary as it bears on *this* design:

| | pre-M4b.9 (823437f) | post-M4b.9 (2387266) |
|---|---|---|
| prefill scale @ t=12 | 5.67x (NOT MET) | **10.63x (MET)** |
| pp tok/s @ t=12 | 346.8 | 652.4 (+88%) |
| pp tok/s @ t=1 | 61.18 | 61.37 (unchanged) |
| Amdahl residual serial fraction | ≈10.2% | ≈1.2% |

The two Risks this design named were both detectors, and both read negative:

1. **Dispatch-count growth** (≈2.5x the joins per layer per tile). Predicted
   signature: low-t degradation with high-t recovery. Observed: 2.00x @ t=2
   and 3.92x @ t=4 — at or within 2% of ideal, i.e. the added joins are not
   measurable even where they are most costly relative to the work. **Job
   fusion (approach C) is therefore not exercised and stays a dormant,
   pre-authorized lever, not a follow-up.**
2. **Outlining perturbs single-thread codegen.** Predicted detector: the t=1
   `bench-compiled` gate. Observed: t=1 prefill 61.37 vs 61.18 tok/s — no
   regression from the call boundary. The nightly still runs as the standing
   guard, but the quiet-hw number already answers it on quiet silicon.

Verification item 3 (the NOT-MET fork — attribution via the new per-label
`kv_append`/`quantize` profile brackets) was **not needed**: the gate is met,
so no residual attribution was taken. The split brackets remain as shipped
profiling surface.

The M4b.1 exit criterion (**≥6x @ t=12**) is now satisfied, three sessions
after it was first taken to bare metal. M4b.9's premise — that the serial
tail, not dispatch overhead or memory bandwidth, was the binding constraint —
is confirmed by the residual-serial-fraction collapse from ≈10.2% to ≈1.2%.
