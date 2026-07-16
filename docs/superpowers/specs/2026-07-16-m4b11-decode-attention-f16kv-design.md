# M4b.11 — Decode Attention Parallelism + F16 KV (Attribution-Gated) Design

**Date:** 2026-07-16
**Status:** Approved design, pre-implementation
**Milestone:** M4b.11 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.10](2026-07-12-m4b10-decode-cap-formula-design.md))

This milestone takes up [M4b.2](2026-07-07-m4b2-per-thread-gap-design.md)'s
pre-registered **decode attribution fork**, open since 2026-07-07 and named
by M4b.10's scope table as the next item — deliberately sequenced after the
decode-cap removal so the fork is taken against a de-throttled baseline
rather than a cap-5 one.

## Motivation

The fork's only recorded decode profile blamed **attention at 81.5% of
decode cycles** (M4b.2 §Amendments, 2026-07-07). That number is stale three
times over: it was taken in the 8-CPU quota'd devpod, *before* M4b.3
vectorized the attention kernel, and *before* M4b.10 uncapped decode
threading. It cannot take the fork.

What has not gone stale is the structural fact behind it: **decode
attention is the only serial op left in a decode step whose GEMVs now shard
across the full `active_threads`** (decode codegen invokes the attention
kernel directly — it never reaches the pool at all; [M4b.8](2026-07-11-m4b8-parallel-attention-design.md)'s
`m == 1` guard covers T=1 prefill tiles only — and KV is still f32 per the
M3 note). Its Amdahl share of decode wall time
therefore *grows* with every core the uncap freed. Meanwhile the v1 win
criterion needs tg > 1x vs llama.cpp best-of, and the last recorded tg
ratios are 0.74x–0.85x — on the capped baseline, with M4b.10 measuring the
cap itself as worth 7–12% on those boxes. The true starting point is
unknown until re-measured.

So M4b.11 is **attribution-first**: a fresh quiet-hw decode attribution
(which doubles as the deferred uncapped tg re-bench), then two
pre-registered gates that authorize zero, one, or two levers. A clean STOP
with a recorded finding is a successful outcome (M4b.4/M4b.6 precedent).

## Scope Decisions (M4b.11)

| Decision | Choice |
|---|---|
| Phase | **Decode only.** Prefill is closed (M4b.1 gate MET at 10.63x @ t=12, M4b.9) |
| Structure | **Attribution-first, gated.** Task 1 = fresh quiet-hw decode profile (t=1 + best-t) on the uncapped baseline; each lever ships only if its pre-registered gate fires |
| Lever 1 (gated) | **Head-sharded decode attention** — head-span kernels + a new pool dispatcher + a one-call codegen edit (decode `lower_attention` calls the dispatcher instead of the kernel). `HOST_ABI_VERSION` bump + recompile; bit-identical by construction |
| Lever 2 (gated) | **F16 KV** — interpreter and compiled path switch together; `HOST_ABI_VERSION` bump + cache-key change; documented tolerance re-derivation. Retires the M3 "KV stays f32" note deliberately |
| Fork arity | **Independent gates**, not M4b.2's "exactly one lever": the levers compose (F16 halves the bytes the attention loop streams; sharding divides its wall time) and each has its own governing measurement |
| Machines | 16c `d2.c1.medium` (primary — deepest recorded history) + 8c `s2.c2.medium` (check). Structural levers, not tuning constants; two microarchitectures showing the same shape suffice |
| Exit target | Set from measured headroom in the attribution amendment (the fork as written); tg ≥ 1x vs llama best-of recorded as v1 context, **never the gate** |
| By-product | The attribution session records the deferred **uncapped tg re-bench** (M4a §Amendments forward note, 2026-07-15) in M4a's Amendments |
| Tolerances | Untouched by Lever 1 (numerics-free by construction). Lever 2 re-derives `attn_rel_tol` / `logits_abs_tol` against observed distributions, documented here — never loosened-to-green |
| Measurement discipline | Quiet bare metal via `mise run metal`; interleaved reps; regret/ratios computed within-session (standing M4b discipline) |

**Explicitly out of scope:**

- **Flash-decoding position-split** (sharding the score/softmax/V loop over
  KV positions with an ordered online-softmax merge). It scales past the
  14-head cap, but it redefines the numeric contract and touches every
  differential, for headroom the 8c/16c boxes cannot use. Recorded as a
  **future escalation**: if, after this milestone's levers land, a
  many-core box's decode profile still blames serial attention, that is
  its own milestone with its own attribution.
- **NUMA-aware threading** — out of scope since M4b.1.
- **Prefill anything** — the prefill gate is MET and closed.
- **Any CI perf gate** — standing rule (AGENTS.md).
- **Any pos-/size-threshold heuristic in the decode dispatch.** M4b.10
  just buried one tuning constant; this milestone does not plant another.
  Short-context dispatch overhead is a recorded risk with a diagnostic
  data point, not a new heuristic.

## Attribution protocol (Task 1)

One quiet-hw session per machine, recording in this order:

1. **Uncapped tg re-bench** — the manual `inferno bench` protocol vs
   llama.cpp (pp=512 tg=128, reps=5), the first decode numbers on the
   de-capped default. Recorded in **M4a §Amendments** (standing protocol)
   and cross-referenced here. This is the baseline every lever's win is
   measured against, and it fixes each machine's **best decode thread
   count** `t_best`.
2. **Decode profile at t=1** — `inferno run --profile --threads 1`, the
   M4b.2 protocol on quiet hardware: directly comparable to the recorded
   81.5% baseline, isolates per-op cost.
3. **Decode profile at best-t** — same capture at `t_best`. Yields the two
   numbers the gates consume: **S** = attention's share of decode wall at
   `t_best`, and attention's **in-situ GB/s** read against that machine's
   recorded M4b.10 bandwidth-saturation curve (`gate-bw-curve.sh`).

The profiler is `rdtsc` self-measurement on the calling thread; attention
is serial today, so its best-t wall share is directly measurable. Shares
guide scoping and never gate CI (M4b.2 rule).

## The pre-registered gates

Written down **before** any sweep runs, per the standing discipline
(M4b.2, M4b.8, M4b.9, M4b.10). For the bench model
(`qwen2.5-0.5b-instruct-q8_0`), `n_heads = 14`.

**Gate 1 — head-sharded decode attention.** Ceiling projection of the
decode-wall reduction:

```
P1 = S × (1 − 1 / min(t_best, 14))
```

An upper bound: it ignores dispatch overhead (currently exactly zero —
decode codegen calls the kernel directly, no pool involved; the data
point, not the projection, pays that cost).

**Gate 2 — F16 KV.** Ceiling projection:

```
P2 = S′ × ½ × min(1, BW_insitu / BW_ceiling)
```

where `S′ = S / min(t_best, 14)` if Gate 1 authorized Lever 1 (a
both-machines decision, applied with each machine's own `S` and
`t_best`), else `S′ = S`. The `½` is the byte halving on the fully-bandwidth-bound
assumption; the bandwidth factor discounts the bet in proportion to how
far attention actually sits from the machine's measured ceiling. `S′`
estimates the post-parallelism share by pure division — real caches don't
divide that cleanly; the judgment-call band below absorbs the error.

**Thresholds (M4b.6's STOP gate, verbatim), applied per lever:**

- projected reduction **≥ 5% on both machines → lever authorized**;
- **< 3% on both machines → STOP** — the finding is recorded and the lever
  is closed as a diagnostic;
- **anything between, or the machines split → controller judgment call**,
  recorded as a spec amendment either way.

Gates are evaluated **once**, from the attribution session — no
re-attribution between levers. If both fire, **Lever 1 lands first**
(numerics-free, so its data point is clean), Lever 2 second (the numerics
change isolated in its own data point).

## Lever 1 — head-sharded decode attention

The decode attention kernel (`inferno-kernels/src/attention.rs`) is a
per-head loop: each head's scores → softmax → weighted-V pass is fully
independent, reading the KV cache read-only. That gives a sharding axis
with the same bit-safety argument as `par_gemv`'s row sharding: **every
output element is computed entirely within one lane.**

- **Kernels:** a head-span C-ABI entry per ISA,
  `inferno_attention_f32_{scalar,avx2}_hspan(…, h_start, h_end)` — the
  existing per-head loop restricted to `[h_start, h_end)`. The whole-call
  kernels become `hspan(0, n_heads)` delegates to the same core, so
  nothing forks. Per-head math is unchanged, so hspan output is
  bit-identical to the serial whole call by construction.
- **Pool:** a new host dispatcher symbol,
  `inferno_par_attention_heads(kernel_hspan, …)` (`inferno-pool/src/lib.rs`),
  with the same single-dispatcher CAS guard and serial-fallback structure
  as its three siblings: `shard_table` partitions `0..n_heads`, each lane
  runs the hspan kernel over its head span with lane-local scores
  scratch. The pool stays kernel-agnostic — the hspan kernel arrives as a
  fn pointer from generated code, like every other kernel; the pool never
  links `inferno-kernels`. The existing `inferno_par_attention` (prefill
  token-span) path is untouched, including its `m == 1` serial arm (T=1
  prefill tiles).
- **Codegen:** decode `lower_attention` replaces its direct kernel call
  with one call to the new dispatcher, passing the hspan kernel symbol
  for the module ISA. The KV-append half and all other lowerings are
  untouched.
- **Deployment class:** new host symbol + generated-code change →
  `HOST_ABI_VERSION` bump ("6" → "7") and cache-key change; existing
  cached artifacts recompile on first use (the M4b.8/M4b.9 dispatcher
  precedent, *not* the M4b.5/M4b.10 pool-only class).
  `INFERNO_DECODE_THREADS` continues to bound the lanes as the only
  override.

## Lever 2 — F16 KV

Both sides of the differential switch **together**, per the fork as
written — the interpreter remains the reference and compares like against
like.

- **Storage:** the KV cache becomes f16 (`u16` at the ABI). **Append**
  (generated code and the interpreter) converts f32 → f16
  round-to-nearest-even — `vcvtps2ph` on the SIMD path; the scalar
  conversion is asserted bit-identical to F16C semantics in the rig.
  **Read** widens f16 → f32 via `vcvtph2ps` (lossless); dots, softmax,
  scores, and outputs stay f32. The **only** new rounding term is the
  single quantization on append.
- **ABI:** attention-kernel signatures and the generated KV-append change
  → `HOST_ABI_VERSION` bump + cache-key change; old artifacts recompile
  and are never misread. This deliberately **retires the M3 note** ("KV
  stays f32 to keep the differential clean") rather than eroding it
  silently.
- **Tolerances:** `attn_rel_tol` and `logits_abs_tol` get a principled
  re-derivation for the one new rounding term, documented in this spec's
  Amendments against observed error distributions (the `gemv_rel_tol` /
  `LOGIT_TIE_EPSILON` precedent) — never loosened-to-green.
- **Composition:** landing second, the f16 read path goes into the hspan
  core, so the sharded and serial paths change together. Halved KV memory
  is recorded as a side benefit, not a goal.

## Error handling

No new failure surface beyond existing patterns: hspan inherits the
attention kernel's caller contract (generated code guarantees it by
construction, M3 trust model); the decode dispatch arm inherits
`par_gemv`'s guards — CAS race lost → serial full-range run, pool absent →
serial run. F16 KV's cache-key change means a stale artifact recompiles;
it does not error.

## Testing Strategy

- **Kernel rig:** hspan-vs-whole-call **exact bit-identity** over a
  head/pos/GQA grid including ragged head splits and `n_kv_heads` group
  boundaries; scalar↔AVX2 bit-identity extends to hspan; proptest shapes
  (M2 pattern). If Lever 2 fires: scalar↔F16C conversion bit-identity
  with RNE edge cases (subnormals, ties, ±inf, NaN) plus proptest.
- **Pool:** bit-identity **across thread counts** extends to `m == 1`
  attention (the cap-invariance analogue); the M4b.8 fallback tests
  (pool absent, dispatch race lost) extend to the decode arm.
- **Differentials:** Lever 1 — the `inferno-codegen`
  compiled-vs-interpreter differential and the `inferno-core` artifact
  differential pass **with zero tolerance change**. Lever 2 — both pass
  under the re-derived tolerances with both sides switched together; the
  `inferno-graph` fixture differential (rope-style coupling) is
  untouched.
- **Perf:** end-to-end numbers only via the manual quiet-hw protocol,
  recorded in Amendments. `bench-kernels` gains an hspan criterion bench
  for context. No CI perf gates.

## Exit criteria

1. **Attribution amendment recorded** (both machines): uncapped tg
   re-bench (cross-recorded in M4a), t=1 + best-t decode profiles, and
   both gate verdicts with the arithmetic shown.
2. **Each authorized lever lands with its own quiet-hw data point; each
   STOP records its finding.** Either is a successful outcome.
3. **Closing data point:** tg vs llama.cpp best-of on both machines,
   judged against the headroom-set target written in the attribution
   amendment. The v1 criterion (tg ≥ 1x) is recorded as context, not
   gated.
4. **Short-context diagnostic** (tg from a tiny prompt, pos ≈ 0 start)
   recorded if Lever 1 lands — the dispatch-overhead check, diagnostic
   only.

## Risks

- **Dispatch overhead at `m == 1` may eat the projected win**, especially
  on the 8c box: `P1` is a ceiling that ignores it. Mitigation: the
  lever's data point decides; reverting is restoring the direct kernel
  call in `lower_attention` (one codegen call site). No threshold
  heuristic will be added to mask it.
- **The F16 scalar↔F16C conversion identity is subtle** (RNE ties,
  subnormals). Mitigation: dedicated rig coverage before any adoption;
  the bar is exact bit-identity, same as every kernel pair.
- **`S′` is an estimate**, not a measurement — post-parallelism attention
  share won't divide as cleanly as `S / min(t_best, 14)`. Mitigation: the
  3–5% judgment band absorbs it, and Lever 2's own data point is the
  truth.
- **Two-machine evidence may split.** Pre-registered as a judgment call,
  recorded either way — no silent tie-breaking.
- **The tg win may still not close v1** even with both levers (pp has its
  own open lever — the M4b.2 prefill-attention follow-up). The
  headroom-set target keeps this milestone honest about what its levers
  can reach; the v1 gate stays with M4a's protocol.

## Amendments

*(Recorded protocol data points and scoped amendments land here; never
edit a recorded data point.)*
