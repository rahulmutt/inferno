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

### 2026-07-16 — attribution session A (d2.c1.medium / Xeon Gold 6336Y, 16c): decode profiles at t=1 and t_best=16

First of the two Task 2 sessions (16c primary). Quiet-hw pass: preflight FIT,
smoke then real, inferno @ `1f579ce` (`git_dirty: false`). `t_best` = 16 (the
protocol's physical-core count on this box). Attention decode share S: 33.7%
at t=1, 55.8% at t=16. The companion uncapped tg re-bench from the same
session is recorded in the M4a spec §Amendments (2026-07-16).

```
# gate-decode-attr (M4b.11 attribution: decode profile t=1 + best-t) — 2026-07-16T15:29:48Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-16

--- profile at --threads 1 ---
profile [prefill] 44.285s wall, 137281262842 cyc total
  op                                   cycles   share        GB/s
  attention                       44659891552   32.5%           -
  matmul:lm_head.weight           23314299614   17.0%        41.1
  matmul:layers.*.ffn.down_proj.weight    18609659284   13.6%        39.5
  matmul:layers.*.ffn.gate_proj.weight    17858460048   13.0%        41.2
  matmul:layers.*.ffn.up_proj.weight    17857131702   13.0%        41.2
  swiglu                           4257924198    3.1%           -
  matmul:layers.*.attn.o_proj.weight     3283783282    2.4%        41.3
  matmul:layers.*.attn.q_proj.weight     3282353836    2.4%        41.3
  rmsnorm                          1186978386    0.9%           -
  rope                              864016194    0.6%           -
  quantize                          530801038    0.4%           -
  matmul:layers.*.attn.k_proj.weight      471912076    0.3%        41.0
  matmul:layers.*.attn.v_proj.weight      471721634    0.3%        41.0
  add                               331653660    0.2%           -
  kv_append                         158436186    0.1%           -
  bias                              120927416    0.1%           -
  embed                              21312736    0.0%           -
profile [decode] 3.998s wall, 12297121076 cyc total
  op                                   cycles   share        GB/s
  attention                        4140932266   33.7%           -
  matmul:lm_head.weight            2171905292   17.7%        13.9
  matmul:layers.*.ffn.down_proj.weight     1678220536   13.6%        13.8
  matmul:layers.*.ffn.up_proj.weight     1669720890   13.6%        13.9
  matmul:layers.*.ffn.gate_proj.weight     1667523272   13.6%        13.9
  matmul:layers.*.attn.o_proj.weight      312427520    2.5%        13.7
  matmul:layers.*.attn.q_proj.weight      311677904    2.5%        13.7
  swiglu                            137023376    1.1%           -
  matmul:layers.*.attn.k_proj.weight       45572460    0.4%        13.4
  matmul:layers.*.attn.v_proj.weight       45554398    0.4%        13.4
  rope                               44521154    0.4%           -
  rmsnorm                            40555026    0.3%           -
  add                                18094438    0.1%           -
  bias                               12485866    0.1%           -
  embed                                505366    0.0%           -
  quantize                             401312    0.0%           -
  kv_append                                 0    0.0%           -

--- profile at --threads 16 ---
profile [prefill] 3.687s wall, 11427738740 cyc total
  op                                   cycles   share        GB/s
  attention                        3444655168   30.1%           -
  matmul:lm_head.weight            1806588282   15.8%       529.7
  matmul:layers.*.ffn.gate_proj.weight     1512402708   13.2%       486.1
  matmul:layers.*.ffn.up_proj.weight     1487605868   13.0%       494.2
  matmul:layers.*.ffn.down_proj.weight     1482294262   13.0%       496.0
  swiglu                            352399696    3.1%           -
  matmul:layers.*.attn.o_proj.weight      346622368    3.0%       390.7
  matmul:layers.*.attn.q_proj.weight      310304952    2.7%       436.4
  quantize                          176559194    1.5%           -
  rmsnorm                           113400044    1.0%           -
  rope                               87476734    0.8%           -
  matmul:layers.*.attn.v_proj.weight       80085500    0.7%       241.6
  matmul:layers.*.attn.k_proj.weight       79642870    0.7%       242.9
  add                                68260814    0.6%           -
  bias                               52188654    0.5%           -
  kv_append                          18827932    0.2%           -
  embed                               8423694    0.1%           -
profile [decode] 2.404s wall, 7346681636 cyc total
  op                                   cycles   share        GB/s
  attention                        4098136860   55.8%           -
  matmul:lm_head.weight             708111680    9.6%        42.3
  matmul:layers.*.ffn.gate_proj.weight      588411848    8.0%        39.1
  matmul:layers.*.ffn.up_proj.weight      585690312    8.0%        39.3
  matmul:layers.*.ffn.down_proj.weight      584340852    8.0%        39.4
  matmul:layers.*.attn.o_proj.weight      245359098    3.3%        17.3
  swiglu                            197202064    2.7%           -
  matmul:layers.*.attn.q_proj.weight      135213184    1.8%        31.4
  rope                               49573092    0.7%           -
  rmsnorm                            46708114    0.6%           -
  matmul:layers.*.attn.k_proj.weight       35469022    0.5%        17.1
  matmul:layers.*.attn.v_proj.weight       34453324    0.5%        17.6
  add                                21225638    0.3%           -
  bias                               15945380    0.2%           -
  embed                                500708    0.0%           -
  quantize                             340460    0.0%           -
  kv_append                                 0    0.0%           -

attention decode share (parsed):
  t=1: 33.7%
  t=16: 55.8%
```

### 2026-07-16 — attribution session B (s2.c2.medium / Xeon E-2388G, 8c): decode profiles at t=1 and t_best=8

Second Task 2 session (8c check). PHX had no stock (API 406, nothing
provisioned); ran in CHI. Quiet-hw pass: preflight FIT (psi 0.04,
throttled_delta 0), smoke then real, inferno @ `d5175c7` (`git_dirty:
false`). `t_best` = 8. Attention decode share S: 33.2% at t=1, 46.6% at
t=8. The companion uncapped tg re-bench from the same session is recorded
in the M4a spec §Amendments (2026-07-16, session B).

```
# gate-decode-attr (M4b.11 attribution: decode profile t=1 + best-t) — 2026-07-16T16:43:50Z
machine: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (GenuineIntel) | 16 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-16

--- profile at --threads 1 ---
profile [prefill] 31.374s wall, 100143936256 cyc total
  op                                   cycles   share        GB/s
  attention                       34396290323   34.3%           -
  matmul:lm_head.weight           16578214589   16.6%        59.3
  matmul:layers.*.ffn.down_proj.weight    13882013729   13.9%        54.4
  matmul:layers.*.ffn.up_proj.weight    12694606376   12.7%        59.5
  matmul:layers.*.ffn.gate_proj.weight    12690192472   12.7%        59.5
  swiglu                           2647831747    2.6%           -
  matmul:layers.*.attn.o_proj.weight     2342967758    2.3%        59.4
  matmul:layers.*.attn.q_proj.weight     2339391540    2.3%        59.4
  rmsnorm                           701176277    0.7%           -
  rope                              437722156    0.4%           -
  quantize                          386045603    0.4%           -
  matmul:layers.*.attn.v_proj.weight      334291166    0.3%        59.4
  matmul:layers.*.attn.k_proj.weight      333840150    0.3%        59.5
  add                               209395350    0.2%           -
  kv_append                          83371902    0.1%           -
  bias                               78702304    0.1%           -
  embed                               7882814    0.0%           -
profile [decode] 2.517s wall, 7993676254 cyc total
  op                                   cycles   share        GB/s
  attention                        2657726132   33.2%           -
  matmul:lm_head.weight            1423716425   17.8%        21.9
  matmul:layers.*.ffn.down_proj.weight     1100892073   13.8%        21.7
  matmul:layers.*.ffn.gate_proj.weight     1095520204   13.7%        21.8
  matmul:layers.*.ffn.up_proj.weight     1093835682   13.7%        21.9
  matmul:layers.*.attn.q_proj.weight      204240313    2.6%        21.6
  matmul:layers.*.attn.o_proj.weight      202913244    2.5%        21.7
  swiglu                             94895651    1.2%           -
  matmul:layers.*.attn.k_proj.weight       30420203    0.4%        20.7
  matmul:layers.*.attn.v_proj.weight       30403506    0.4%        20.7
  rope                               26352824    0.3%           -
  rmsnorm                            22380983    0.3%           -
  add                                 5931176    0.1%           -
  bias                                3973518    0.0%           -
  quantize                             314768    0.0%           -
  embed                                159552    0.0%           -
  kv_append                                 0    0.0%           -

--- profile at --threads 8 ---
profile [prefill] 5.033s wall, 16063063364 cyc total
  op                                   cycles   share        GB/s
  attention                        6487101054   40.4%           -
  matmul:lm_head.weight            2312199302   14.4%       424.9
  matmul:layers.*.ffn.down_proj.weight     1998697509   12.4%       377.7
  matmul:layers.*.ffn.gate_proj.weight     1850579127   11.5%       407.9
  matmul:layers.*.ffn.up_proj.weight     1846159703   11.5%       408.9
  swiglu                            389616087    2.4%           -
  matmul:layers.*.attn.o_proj.weight      363741443    2.3%       382.3
  matmul:layers.*.attn.q_proj.weight      352144524    2.2%       394.9
  rmsnorm                           102593675    0.6%           -
  quantize                           90677685    0.6%           -
  rope                               70469010    0.4%           -
  matmul:layers.*.attn.v_proj.weight       62041890    0.4%       320.2
  matmul:layers.*.attn.k_proj.weight       61361367    0.4%       323.8
  add                                38638990    0.2%           -
  bias                               20242324    0.1%           -
  kv_append                          13619195    0.1%           -
  embed                               3180479    0.0%           -
profile [decode] 1.778s wall, 5631359820 cyc total
  op                                   cycles   share        GB/s
  attention                        2621530483   46.6%           -
  matmul:lm_head.weight             775803599   13.8%        40.0
  matmul:layers.*.ffn.down_proj.weight      604902443   10.7%        39.4
  matmul:layers.*.ffn.up_proj.weight      597510724   10.6%        39.9
  matmul:layers.*.ffn.gate_proj.weight      594932208   10.6%        40.1
  matmul:layers.*.attn.o_proj.weight      114496034    2.0%        38.4
  matmul:layers.*.attn.q_proj.weight      113416584    2.0%        38.7
  swiglu                            103820681    1.8%           -
  rope                               27737772    0.5%           -
  rmsnorm                            24656424    0.4%           -
  matmul:layers.*.attn.k_proj.weight       20662880    0.4%        30.4
  matmul:layers.*.attn.v_proj.weight       19878495    0.4%        31.6
  add                                 6582875    0.1%           -
  bias                                4895906    0.1%           -
  quantize                             341604    0.0%           -
  embed                                191108    0.0%           -
  kv_append                                 0    0.0%           -

attention decode share (parsed):
  t=1: 33.2%
  t=8: 46.6%
```

### 2026-07-16 — gate verdicts (Task 3): Gate 1 AUTHORIZED, Gate 2 STOP

Computed from the two attribution sessions above, exactly as pre-registered
(§The pre-registered gates); the implementation tasks were not consulted.

**Parsed S (attention decode share, best-t profile tables above):**
16c 6336Y: S = 55.8% (t_best = 16, decode wall 2.404 s). 8c E-2388G:
S = 46.6% (t_best = 8, decode wall 1.778 s).

**Unique KV bytes (protocol arithmetic, identical for both machines):**
`n_layers = 24`, `kv_dim = 128` (2 kv_heads × head_dim 64, `inferno
inspect`), T = 64 generated tokens, prompt p0 = 2025 tokens. p0 is
estimated, not logged by the protocol: the prompt is random base64
(2048 bytes → 2732 chars); 20 samples through the devenv-pinned
`llama-tokenize` (same GGUF vocab) give mean 2025.25, sd ≈ 18 (range
1990–2057), i.e. ±0.9% on total bytes — far inside the gates' margins.

Σ(p+1) for p = 2025..2088 = 64×2025 + 64×65/2 = 131,680.
Per layer: 2 × 131,680 × 128 × 4 = 134,840,320 B. × 24 layers =
**3,236,167,680 B = 3.2362 GB** (decimal, matching gate-bw-curve's
bytes/sec/1e9).

**In-situ GB/s (attention wall = decode wall × S):**
- 16c: 2.404 × 0.558 = 1.3414 s → 3.2362 / 1.3414 = **2.41 GB/s**
- 8c: 1.778 × 0.466 = 0.8285 s → 3.2362 / 0.8285 = **3.91 GB/s**

**BW_ceiling provenance:** each machine's gate-bw-curve peak from the
M4b.10 spec §Amendments, 2026-07-15 session record: 6336Y **54.39 GB/s**
(2026-07-14T12:44:07Z, lanes = 8); E-2388G **45.95 GB/s**
(2026-07-14T18:14:55Z, lanes = 6).

**Gate 1 — P1 = S × (1 − 1/min(t_best, 14)):**
- 16c: 0.558 × (1 − 1/14) = **51.8%**
- 8c: 0.466 × (1 − 1/8) = **40.8%**

Both ≥ 5% → **Lever 1 AUTHORIZED** (Tasks 4–7 run).

**Gate 2 — P2 = S′ × ½ × min(1, BW_insitu / BW_ceiling), S′ = S/min(t_best, 14)
(Gate 1 authorized):**
- 16c: S′ = 0.558/14 = 0.03986; ratio = 2.41/54.39 = 0.0444 →
  P2 = 0.03986 × ½ × 0.0444 = **0.09%**
- 8c: S′ = 0.466/8 = 0.05825; ratio = 3.91/45.95 = 0.0850 →
  P2 = 0.05825 × ½ × 0.0850 = **0.25%**

Both < 3% → **Lever 2 (F16 KV) STOP**, closed as a diagnostic. **The
finding:** at best-t the serial decode attention pass streams unique KV
at 2.4–3.9 GB/s — 4–9% of each machine's measured bandwidth ceiling —
so decode attention is not bandwidth-bound at the operating point, and
halving KV bytes projects ≤ 0.25% of decode wall. The verdict is
insensitive to the p0 estimate: P2 scales linearly with BW_insitu, which
would have to exceed the ceiling itself before P2 crossed 3% on either
machine. Task 8 is skipped; F16 KV stays closed unless a future
attribution shows attention bandwidth-bound.

**Headroom-set tg target for the closing data point (Task 9):**
baseline tg (this session's uncapped re-bench, M4a §Amendments
2026-07-16) × (1 + authorized levers' combined projected reduction) —
Lever 1 only, so ×(1 + P1):

- 16c: 53.48 × 1.5181 = **81.2 tok/s**
- 8c: 56.74 × 1.4078 = **79.9 tok/s**

Conservatism, stated explicitly: (1 + P1) deliberately understates the
throughput a P1 wall-reduction would imply (1/(1−P1) = 2.08× / 1.69×),
but P1 itself is a ceiling that ignores dispatch overhead (§Risks) and
assumes perfect head-parallel scaling; the target nets those against
each other. The closing data point judges against these numbers; v1
(tg ≥ 1x vs llama best-of) is recorded as context only, never the gate.

### 2026-07-16 — Lever-1 data point (Task 7): decode tg +12.7% (16c) / +13.4% (8c); lever kept

Within-session A/B on both machines, parent `f74069e` (pre-Task-4) vs lever
`ebe83bc` (Tasks 4–6), interleaved invocations per the standing discipline;
preflight FIT on both; both boxes in CHI (PHX had no stock — different
physical instances than Task 2's session A box, which the within-session
design makes irrelevant). M4a protocol command per arm
(`bench --pp 512 --tg 128 --reps 5 --threads 0 --json`) plus the
short-context diagnostic (`--pp 16 --tg 32`).

**16c (Xeon Gold 6336Y, d2.c1.medium):**

```
# M4b.11 Lever-1 A/B — 2026-07-16T18:10:15Z
parent: f74069edb7ab935e0ae4dbab074e7c2b795ff445 | lever: ebe83bcfeb5839e42a9e786832c4032c48af3dbc
model name	: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
logical CPUs: 32
6.9.10+bpo-amd64

## parent-long.json
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "f74069e",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 5,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 798.6328922392393,
  "inferno_pp_stddev": 35.14910182966182,
  "inferno_tg_tok_s": 50.70993765871119,
  "inferno_tg_stddev": 0.5130204889396626,
  "llama_pp_tok_s": 1079.879018,
  "llama_pp_stddev": 282.208549,
  "llama_tg_tok_s": 60.107193,
  "llama_tg_stddev": 0.198032,
  "llama_t1_pp_tok_s": 117.922843,
  "llama_t1_pp_stddev": 0.045636,
  "llama_t1_tg_tok_s": 16.501481,
  "llama_t1_tg_stddev": 0.024568,
  "inferno_t1_pp_tok_s": 61.52930858195416,
  "inferno_t1_pp_stddev": 0.01297939025582972,
  "inferno_t1_tg_tok_s": 21.825334107656563,
  "inferno_t1_tg_stddev": 0.003631112457014242
}

## lever-long.json
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "ebe83bc",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 5,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 825.5882838046258,
  "inferno_pp_stddev": 1.368900128385791,
  "inferno_tg_tok_s": 57.144515715476494,
  "inferno_tg_stddev": 0.8598028864106598,
  "llama_pp_tok_s": 1152.475867,
  "llama_pp_stddev": 253.474047,
  "llama_tg_tok_s": 60.396291,
  "llama_tg_stddev": 0.440273,
  "llama_t1_pp_tok_s": 118.380995,
  "llama_t1_pp_stddev": 0.434384,
  "llama_t1_tg_tok_s": 22.667968,
  "llama_t1_tg_stddev": 0.010698,
  "inferno_t1_pp_tok_s": 61.372917574652924,
  "inferno_t1_pp_stddev": 0.143995986641353,
  "inferno_t1_tg_tok_s": 22.243378955892513,
  "inferno_t1_tg_stddev": 0.027464621580827724
}

## parent-short.json
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "f74069e",
  "llama_build_commit": "6f4f53f",
  "pp": 16,
  "tg": 32,
  "reps": 5,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 618.389012044565,
  "inferno_pp_stddev": 135.08682273852895,
  "inferno_tg_tok_s": 63.530573388403276,
  "inferno_tg_stddev": 0.37098307381862894,
  "llama_pp_tok_s": 647.881867,
  "llama_pp_stddev": 10.072625,
  "llama_tg_tok_s": 63.554477,
  "llama_tg_stddev": 2.637514,
  "llama_t1_pp_tok_s": 98.02317,
  "llama_t1_pp_stddev": 0.181475,
  "llama_t1_tg_tok_s": 22.831486,
  "llama_t1_tg_stddev": 0.007341,
  "inferno_t1_pp_tok_s": 70.9848009668085,
  "inferno_t1_pp_stddev": 0.08184955648835691,
  "inferno_t1_tg_tok_s": 23.95369644197977,
  "inferno_t1_tg_stddev": 0.27274227997289807
}

## lever-short.json
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz",
  "physical_cores": 16,
  "logical_cores": 32,
  "inferno_version": "0.1.0",
  "inferno_git": "ebe83bc",
  "llama_build_commit": "6f4f53f",
  "pp": 16,
  "tg": 32,
  "reps": 5,
  "inferno_threads": 16,
  "llama_threads": 16,
  "inferno_pp_tok_s": 574.0082344547969,
  "inferno_pp_stddev": 142.2490770577237,
  "inferno_tg_tok_s": 62.311857252695766,
  "inferno_tg_stddev": 1.5336768564031324,
  "llama_pp_tok_s": 598.713056,
  "llama_pp_stddev": 3.697379,
  "llama_tg_tok_s": 64.386323,
  "llama_tg_stddev": 1.6379,
  "llama_t1_pp_tok_s": 84.648605,
  "llama_t1_pp_stddev": 0.145328,
  "llama_t1_tg_tok_s": 16.473534,
  "llama_t1_tg_stddev": 0.074192,
  "inferno_t1_pp_tok_s": 71.26257243615689,
  "inferno_t1_pp_stddev": 0.17199813728394345,
  "inferno_t1_tg_tok_s": 23.266772036134295,
  "inferno_t1_tg_stddev": 0.19137532306713256
}

--- within-session ratios (lever/parent) ---
long: tg 50.70993765871119 -> 57.144515715476494 (x1.1269) | pp 798.6328922392393 -> 825.5882838046258 (x1.0338)
short: tg 63.530573388403276 -> 62.311857252695766 (x0.9808) | pp 618.389012044565 -> 574.0082344547969 (x0.9282)
```

**8c (Xeon E-2388G, s2.c2.medium):**

```
# M4b.11 Lever-1 A/B — 2026-07-16T18:40:10Z
parent: f74069edb7ab935e0ae4dbab074e7c2b795ff445 | lever: ebe83bcfeb5839e42a9e786832c4032c48af3dbc
model name	: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz
logical CPUs: 16
6.9.10+bpo-amd64

## parent-long.json
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "f74069e",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 5,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 561.7553606115027,
  "inferno_pp_stddev": 69.18354133887541,
  "inferno_tg_tok_s": 54.95298632751336,
  "inferno_tg_stddev": 0.5608284779582066,
  "llama_pp_tok_s": 1046.481488,
  "llama_pp_stddev": 1.974771,
  "llama_tg_tok_s": 72.676938,
  "llama_tg_stddev": 0.136844,
  "llama_t1_pp_tok_s": 165.197085,
  "llama_t1_pp_stddev": 0.032633,
  "llama_t1_tg_tok_s": 33.676624,
  "llama_t1_tg_stddev": 0.017472,
  "inferno_t1_pp_tok_s": 87.91627694295605,
  "inferno_t1_pp_stddev": 0.08436818720187408,
  "inferno_t1_tg_tok_s": 34.17909785553883,
  "inferno_t1_tg_stddev": 0.09810688797130884
}

## lever-long.json
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "ebe83bc",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 5,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 616.0170539070803,
  "inferno_pp_stddev": 14.17343068412102,
  "inferno_tg_tok_s": 62.29783192521944,
  "inferno_tg_stddev": 0.6613276998916865,
  "llama_pp_tok_s": 1045.051971,
  "llama_pp_stddev": 1.860512,
  "llama_tg_tok_s": 72.736329,
  "llama_tg_stddev": 0.037116,
  "llama_t1_pp_tok_s": 165.018451,
  "llama_t1_pp_stddev": 0.471274,
  "llama_t1_tg_tok_s": 33.395607,
  "llama_t1_tg_stddev": 0.026383,
  "inferno_t1_pp_tok_s": 89.12442616501292,
  "inferno_t1_pp_stddev": 0.06214856530277479,
  "inferno_t1_tg_tok_s": 34.56110229413846,
  "inferno_t1_tg_stddev": 0.11417049388125378
}

## parent-short.json
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "f74069e",
  "llama_build_commit": "6f4f53f",
  "pp": 16,
  "tg": 32,
  "reps": 5,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 660.5874677011891,
  "inferno_pp_stddev": 2.047906081336703,
  "inferno_tg_tok_s": 67.40417397744628,
  "inferno_tg_stddev": 0.009987423373478046,
  "llama_pp_tok_s": 695.567542,
  "llama_pp_stddev": 5.809217,
  "llama_tg_tok_s": 71.2929,
  "llama_tg_stddev": 2.690882,
  "llama_t1_pp_tok_s": 139.792715,
  "llama_t1_pp_stddev": 0.269516,
  "llama_t1_tg_tok_s": 33.70961,
  "llama_t1_tg_stddev": 0.22762,
  "inferno_t1_pp_tok_s": 101.22746440492234,
  "inferno_t1_pp_stddev": 0.020644634223130894,
  "inferno_t1_tg_tok_s": 37.77947822201564,
  "inferno_t1_tg_stddev": 0.03009997577874529
}

## lever-short.json
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz",
  "physical_cores": 8,
  "logical_cores": 16,
  "inferno_version": "0.1.0",
  "inferno_git": "ebe83bc",
  "llama_build_commit": "6f4f53f",
  "pp": 16,
  "tg": 32,
  "reps": 5,
  "inferno_threads": 8,
  "llama_threads": 8,
  "inferno_pp_tok_s": 660.2509426322911,
  "inferno_pp_stddev": 3.076636833628491,
  "inferno_tg_tok_s": 67.38537833543542,
  "inferno_tg_stddev": 0.409116869918936,
  "llama_pp_tok_s": 741.148084,
  "llama_pp_stddev": 0.832686,
  "llama_tg_tok_s": 73.231058,
  "llama_tg_stddev": 0.041178,
  "llama_t1_pp_tok_s": 139.279557,
  "llama_t1_pp_stddev": 0.193879,
  "llama_t1_tg_tok_s": 33.878735,
  "llama_t1_tg_stddev": 0.038399,
  "inferno_t1_pp_tok_s": 100.87022927632182,
  "inferno_t1_pp_stddev": 0.16164781412212373,
  "inferno_t1_tg_tok_s": 38.104628799731145,
  "inferno_t1_tg_stddev": 0.027262692004646982
}

--- within-session ratios (lever/parent) ---
long: tg 54.95298632751336 -> 62.29783192521944 (x1.1337) | pp 561.7553606115027 -> 616.0170539070803 (x1.0966)
short: tg 67.40417397744628 -> 67.38537833543542 (x0.9997) | pp 660.5874677011891 -> 660.2509426322911 (x0.9995)
```

**Against Gate 1's projection:** P1 was a ceiling of 51.8% (16c) / 40.8%
(8c) of decode wall; the conservative headroom targets implied tg ×1.518 /
×1.408. Observed: tg ×1.1269 (16c) and ×1.1337 (8c) — an implied decode-wall
reduction of 11.3% / 11.8%. **The projection did not hold at its ceiling;
the lever's win is real, positive on both machines, and far from P1** — the
per-head shards evidently do not scale the serial attention pass anywhere
near `min(t_best, 14)`-way. tg improved on both machines, so the
pre-registered revert mitigation is not triggered: **the lever is kept.**
The gap between +12% and the ceiling is a finding for any future decode-
attention work, not a defect of this one.

**Short-context diagnostic (dispatch overhead near pos ≈ 0, diagnostic
only):** 16c tg ×0.9808 / pp ×0.9282; 8c tg ×0.9997 / pp ×0.9995. A small
overhead is visible on the 16c box and none on the 8c box; per the spec, no
threshold heuristic is added.

pp at 512 moved ×1.0338 (16c) / ×1.0966 (8c); prefill lowering is untouched
by the lever, and the parent-arm pp stddevs (35.1 and 69.2 tok/s
respectively) put both movements within run-to-run spread — recorded as
context, not as a prefill effect.
