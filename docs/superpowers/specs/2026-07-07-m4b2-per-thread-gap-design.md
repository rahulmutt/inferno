# M4b.2 — Per-Thread Gap Design (Profiler, Prefill Tiles, Decode Lever)

**Date:** 2026-07-07
**Status:** Approved design, pre-implementation
**Milestone:** M4b.2 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones and the [M4b.1 spec](2026-07-06-m4b1-threading-design.md))

M4b.1 multi-threaded the generated code but its ≥6x scaling exit
criterion proved **unevaluable in this environment**: the dev machine is
a devpod container CPU-quota'd to 8 CPUs with proven mid-run throttling,
and llama.cpp's own prefill scales negatively there. The scaling
questions (exit-criterion evaluation, any parallel-attention amendment)
stay parked on M4b.1's required bare-metal rerun.

The **per-thread gap is measurable right now**: t=1-vs-t=1 comparisons
run both engines in the same environment and are not confounded by the
quota. Recorded baseline (M4b.1 data point, Q8_0 pinned model): inferno
t=1 pp 15.12 vs llama.cpp t=1 pp 54.60 (**0.28x**); tg 9.81–10.50 vs
23.42–23.93 (**0.42–0.44x**). M4b.2 attacks that gap, entirely at t=1.

What the evidence already rules in and out:

- **Prefill has a known structural cause.** Compiled prefill is the
  single-token forward wrapped in a token loop (`lower_prefill`,
  `inferno-codegen/src/llvm/ops.rs`) — every weight matrix is re-read
  from memory once *per token*, 512 full model passes for pp512.
  llama.cpp runs prefill as batched GEMM. Prefill tiles are the
  highest-confidence lever and this spec commits to them.
- **Decode is NOT losing in the GEMV kernel.** The M2 rig recorded the
  Q8_0 rs8 AVX2 GEMV at **1.09–1.72x ggml's throughput** on the hot
  shapes, yet end-to-end decode runs at 0.42–0.44x. The gap must live in
  generated attention/elementwise code, per-token overhead, activation
  quantize, or f32-KV memory traffic — and **no per-op profile exists
  anywhere in the project** (every attempt was blocked: no `perf` binary
  in the container, `perf_event_paranoid=3`). M4b.1's "~5–10% serial
  work" figure is an assertion, not a measurement. So decode levers are
  chosen from data this milestone first produces, not guessed.
- **AVX-512 kernels are out.** They chiefly fix the Q4_K microkernel
  (0.73–0.80x ggml, shuffle-port-bound, needs VNNI per the M2 spec), and
  the pinned nightly model is Q8_0. The `Isa::X86_64v4` / `Feature::Vnni`
  detection plumbed in M2 sits ready for whenever that work is scoped.

## Scope Decisions (M4b.2)

| Decision | Choice |
|---|---|
| Measurement basis | Everything t=1-vs-t=1 under the standard protocol (devenv shell, release, pinned Q8_0 model). Threading/scaling claims remain gated on M4b.1's bare-metal rerun |
| Attribution | **Profiler first.** A built-in, flag-gated per-op profiler lands before any optimization; the baseline decode+prefill profile is recorded in Amendments before tiles work starts |
| Exit criterion (prefill) | **Hard:** t=1 pp ≥ **0.7x** llama.cpp t=1, same protocol, data point recorded in Amendments |
| Exit criterion (decode) | **Scoped:** tg target set by explicit spec amendment *after* the profiler attributes the decode gap, against measured headroom; then met or explicitly re-scoped. No blind number — that is what made M4b.1's criterion unevaluable |
| Levers committed | Prefill/GEMM tiles (incremental: M-loop batch first, register-blocked tiles only by escalation amendment) |
| Levers contingent | Exactly one decode lever picked by the pre-registered attribution fork below: F16 KV, targeted fusion on the blamed ops, or a batched/AVX quantize path |
| Standing invariants | Bit-identity across thread counts (extends to `par_gemm`); scalar-vs-SIMD kernel bit-identity; compiled-vs-interpreter differential green with **no tolerance loosening** (an F16-KV adoption re-derives tolerances, documented, both sides switching together) |

**Explicitly out of scope:** AVX-512 kernel bodies (revisit when Q4_K
matters or bare metal exists), parallel attention (gated on the M4b.1
bare-metal rerun), batched *decode* / continuous batching (v2), NUMA and
thread affinity, in-memory LLJIT, prefix caching, any CI perf gate
(never — AGENTS.md).

## What M4b.2 Adds

- **`inferno-codegen`** — profile instrumentation variant (per-op rdtsc
  accumulation); `lower_prefill` restructured to tile-of-T batching.
- **`inferno-kernels`** — `gemm_*_rs8` batched siblings of the three
  GEMV kernels (f32, q8_0, q4_k), scalar + AVX2; rig extension.
- **`inferno-plan`** — tile size T as a planner constant; prefill
  activation-arena slots sized ×T; matmul byte counts exported for the
  profiler's GB/s column.
- **`inferno-pool`** — `inferno_par_gemm` dispatcher (row-partitioned,
  same shard rules as `inferno_par_gemv`).
- **`inferno-core`** — profile buffer ownership + symbol retention;
  artifact cache key grows the profile flag and T.
- **`cli`** — `--profile` on `inferno run` and `inferno bench`; profile
  table output.
- One decode lever (implementation scoped by amendment).

## The profiler

The compiled forward function is one generated entry point, so only
codegen can see op boundaries. When compiled with profiling enabled,
codegen wraps **each lowered op** — not just each island; GEMV must be
separable from elementwise inside the Ffn island — with `rdtsc` reads
and accumulates the delta into a per-op slot of a `u64` accumulator
array passed through the runtime context. No `perf`, no kernel
counters: it works in this container, which is the point.

**Granularity:** per op kind × site — embed; quantize; each matmul site
by name (qkv, o, gate/up, down, logits); rmsnorm; rope; swiglu; add;
attention. Not per layer (op-kind totals are enough to take the
attribution fork).

**Surface:** `inferno run --profile` and `inferno bench --profile`
print a table: per-op total time, share of wall clock, and — for matmul
sites — **achieved GB/s** computed from planner-known weight bytes. The
GB/s column is the load-bearing one: in-situ GEMV throughput vs the M2
microbench number (43.7 GiB/s for the hot Q8_0 shape) directly separates
"decode loses to per-op overhead" from "decode loses to memory
bandwidth."

**Constraints:**

- Profiling is a compile-time variant: the artifact cache key includes
  the profile flag; a profiled `model.so` is a distinct artifact.
- Instrumentation reads clocks and never touches math: logits must be
  **bit-identical** with and without `--profile` (asserted by test).
- Profiles are self-measurements that guide scoping; they never gate CI.
- rdtsc overhead (~tens of cycles) is negligible against µs-scale ops.

**Non-goals:** sampling, flamegraphs, per-layer breakdown.

## Prefill tiles

**Kernel layer.** Each rs8 GEMV kernel gains a batched sibling
`gemm_*_rs8(m, …)`: outer loop over weight strips (the strip/block inner
code is unchanged), inner loop over the `m` activation rows. Each weight
strip is read once per *batch* instead of once per token — the
structural win. Activations for the `m` tokens are a contiguous panel — quantized
row-by-row with the existing `quantize_row_*` for the quantized dtypes,
raw f32 rows for the f32 kernel (as today). `m` is a runtime argument.

**Determinism by construction.** Per-output-element dot order is
identical to GEMV (same strip/block k-order), so:

- `gemm(m=1)` must **bit-equal** `gemv` (new rig invariant);
- scalar and AVX2 GEMM must be bit-identical (existing rig rule);
- the compiled-vs-interpreter differential tolerances are untouched.

**Codegen.** `lower_prefill` moves from `for token { single-token
forward }` to a tile loop with compile-time tile size **T** (planner
constant, default 64; part of the artifact cache key). The final ragged
tile runs the same code with runtime `m ≤ T` — no second body. Within a
tile:

- embed and elementwise ops (rmsnorm, rope, swiglu, add, quantize) gain
  an outer m-loop over tile rows;
- matmuls call `gemm` with `m` = tile length via `inferno_par_gemm`;
- **attention stays a serial per-position loop**: the tile's K/V rows
  are all written first, then position `p` attends over `[0..p]`.
  Causality holds (K/V for positions ≤ p exist) and the math is
  unchanged from today;
- logits stay last-token-only (GEMV).

**Decode path untouched.** Decode keeps GEMV exactly as-is; zero
regression surface there.

**Memory plan.** The f32 intermediate arena already sizes every
`[Seq, ..]` value to `max_seq_len` rows (liveness-packed), so a tile of
T ≤ n ≤ max_seq_len tokens writes to already-allocated distinct rows —
**no f32-arena change is needed**. The one region that must grow is the
**quantized-activation scratch** (`act_scratch`, `inferno-plan/src/memory.rs`):
today it holds one quantized row; the GEMM activation panel needs T
contiguous quantized rows, so it is sized ×T. Decode is untouched. F32
weights need no panel at all — their per-token source rows are already a
contiguous `stride = k` panel in the arena, read straight by the GEMM.

**Pool.** `inferno_par_gemm` partitions rows across threads exactly like
`inferno_par_gemv`: each output row computed wholly by one thread with
the kernel's fixed combine order, so thread count never changes output
bits — same argument, same exact-equality assertion as M4b.1.

**Escalation clause (pre-registered).** If, with the M-loop GEMM landed,
t=1 pp is still < 0.7x, upgrading hot shapes to register-blocked tiles
(activation panel in registers, M×8 tiles, cache blocking over k)
becomes an **explicit scoped amendment** targeted at the shapes the
profiler blames — the same no-silent-scope rule as M4b.1's
parallel-attention clause.

## Decode attribution fork (pre-registered)

After the profiler lands, the baseline decode profile is recorded in
Amendments and picks **exactly one** lever via explicit amendment:

1. **Attention/KV dominates and in-situ GB/s sits near the memory
   ceiling → F16 KV.** The compiled path *and* the interpreter switch KV
   storage to F16 **together**, so the differential compares like
   against like. `logits_abs_tol` gets a principled re-derivation for
   the one new F16 rounding term in K/V — a documented derivation in
   this spec, never loosened-to-green. This deliberately retires the M3
   note ("KV stays f32 to keep the differential clean") rather than
   eroding it silently.
2. **Generated elementwise/norm/rope/overhead dominates → targeted
   fusion/codegen work** on exactly the blamed ops; the amendment lists
   the op set.
3. **Activation quantize dominates → batched/AVX quantize path.**

The tg exit target is set in the same amendment, against measured
headroom.

## Error handling

No new failure surface beyond existing patterns: `--profile` on a model
compiled without instrumentation recompiles (distinct cache key), it
does not error; `gemm` shares the GEMV kernels' contract (caller
validates shapes at plan time); `inferno_par_gemm` inherits
`inferno_par_gemv`'s dispatch guards.

## Testing Strategy

- **Kernel rig:** GEMM scalar-vs-AVX2 bit-identity over an m×rows×k
  grid including ragged shapes; `gemm(m=1) ≡ gemv` bitwise; proptest
  shapes (M2 pattern).
- **Differential gates (unchanged, must stay green):** the
  `inferno-codegen` compiled-vs-interpreter differential and the
  `inferno-core` artifact test pass with prefill tiles **without any
  tolerance change**.
- **Determinism:** bit-identity across thread counts extends to
  `par_gemm`; logits bit-identical with/without `--profile`; logits
  bit-identical across tile sizes T (new differential axis — tiling
  must not change math).
- **Perf context:** `bench-kernels` gains GEMM criterion benches vs
  ggml. End-to-end numbers only via the manual `mise run bench`
  protocol, recorded in Amendments. No CI perf gates.

## Implementation Phases

1. **Profiler** — codegen instrumentation, cache-key change, CLI
   `--profile`, table output; record the **baseline decode + prefill
   profile** in Amendments before any optimization.
2. **GEMM kernels** — M-loop `gemm_*_rs8` (scalar + AVX2, three
   dtypes), rig extension.
3. **Prefill tiling** — `lower_prefill` tile loop, arena ×T,
   `inferno_par_gemm`, differential + determinism tests.
4. **Prefill data point** — protocol run, recorded in Amendments;
   escalation amendment if pp < 0.7x.
5. **Decode amendment** — attribution fork taken from the recorded
   profile; tg target set; chosen lever implemented; tg data point
   recorded.
6. **Closing data point** — full protocol run recorded in Amendments.

## Risks

- **The M-loop GEMM may not reach 0.7x** if prefill is limited by more
  than weight re-reads (e.g. serial attention share grows as matmuls
  shrink). Mitigation: the escalation clause is pre-registered, and the
  profiler measures prefill too, so the amendment is scoped from data.
- **Decode attribution may be split** (no single dominant cost).
  Mitigation: the fork's amendment picks the largest measured item; the
  tg target is set against that item's headroom, not against a hope.
- **rdtsc-based self-measurement is noisy in a throttled container.**
  Mitigation: shares are meaningful even when absolute times wobble;
  profiles guide scoping and never gate anything.
- **F16 KV touches the differential's foundations.** Mitigation: both
  sides switch together; tolerance change is a documented re-derivation
  reviewed against observed error distributions (the `gemv_rel_tol` /
  `LOGIT_TIE_EPSILON` precedent).

## Amendments

*(Recorded protocol data points and scoped amendments land here; never
edit a recorded data point.)*

### 2026-07-07 — Pre-optimization baseline `--profile` capture (Phase 1)

- **Commit:** recorded by the `feat(cli,core): inferno run --profile —
  per-op cycle/GB-s tables; record baseline` commit on branch
  `m4b2-per-thread-gap` (built on parent `2fa1cd1`).
- **Model:** `qwen2.5-0.5b-instruct-q8_0.gguf`, `--threads 1`,
  `--max-tokens 64`, random base64 prompt (`head -c 2048 /dev/urandom |
  base64`, ~2.7 KB → ~1.3 K prompt tokens).
- **Status:** **pre-optimization / pre-tiling baseline.** Prefill is NOT
  tiled yet (that lands in Phase 3 / Task 9); this is the reference the
  GEMM + tiling work and the decode attribution fork are scoped against.
  Do not interpret it into an optimization lever here — that is the
  Phase-5 amendment.
- **Environment caveat:** 8-CPU cgroup-quota'd shared devpod. The run is
  `--threads 1`, so the quota does not distort this single-threaded
  profile. `rdtsc`-based self-measurement; shares are meaningful even
  where absolute wall-times wobble. The GB/s column is the diagnostic
  approximation described in `Engine::profile_matmul_bytes` (weight-image
  bytes per slot × per-token invocations; exact for decode, an
  over-count of calls for the batched prefill), not a contract.

Both tables verbatim from the run:

```
profile [prefill] 294.463s wall, 912763978121 cyc total
  op                                   cycles   share        GB/s
  attention                      624796451647   68.5%           -
  matmul:lm_head.weight           74244813963    8.1%        13.1
  matmul:layers.*.ffn.gate_proj.weight    60287936474    6.6%        12.4
  matmul:layers.*.ffn.up_proj.weight    59638351868    6.5%        12.6
  matmul:layers.*.ffn.down_proj.weight    58792975129    6.4%        12.7
  matmul:layers.*.attn.o_proj.weight    11677119697    1.3%        11.8
  matmul:layers.*.attn.q_proj.weight    10884470923    1.2%        12.7
  swiglu                           5081349250    0.6%           -
  matmul:layers.*.attn.k_proj.weight     1623296036    0.2%        12.2
  rope                             1596506924    0.2%           -
  matmul:layers.*.attn.v_proj.weight     1584005840    0.2%        12.5
  rmsnorm                          1409228632    0.2%           -
  add                               714746984    0.1%           -
  bias                              379261228    0.0%           -
  embed                              42510790    0.0%           -
  quantize                           10952736    0.0%           -
profile [decode] 17.149s wall, 53065140653 cyc total
  op                                   cycles   share        GB/s
  attention                       43236236049   81.5%           -
  matmul:lm_head.weight            2398434901    4.5%        12.6
  matmul:layers.*.ffn.gate_proj.weight     2137158496    4.0%        10.9
  matmul:layers.*.ffn.up_proj.weight     2079020006    3.9%        11.2
  matmul:layers.*.ffn.down_proj.weight     2013504139    3.8%        11.6
  matmul:layers.*.attn.o_proj.weight      432879623    0.8%         9.9
  matmul:layers.*.attn.q_proj.weight      375441139    0.7%        11.4
  swiglu                            157185878    0.3%           -
  matmul:layers.*.attn.v_proj.weight       53564531    0.1%        11.4
  matmul:layers.*.attn.k_proj.weight       52367101    0.1%        11.7
  rope                               51067684    0.1%           -
  rmsnorm                            43344252    0.1%           -
  add                                22557163    0.0%           -
  bias                               11541235    0.0%           -
  embed                                458304    0.0%           -
  quantize                             380152    0.0%           -
```

### 2026-07-07 — Batched GEMM kernel µbench (Phase 4, Task 10 Step 2)

- **Commit:** recorded by the `bench(kernels): batched GEMM criterion
  group; record M4b.2 prefill data point` commit on branch
  `m4b2-per-thread-gap` (parent `815c40c`, the prefill-tiling commit).
- **How:** `mise run bench-kernels` (`cargo bench -p inferno-kernels
  --features ggml-compare`) inside `devenv shell`; the new `gemm`
  criterion group drives `KernelSet::gemm` over the Llama-family shapes
  with `m ∈ {1, 16, 64}`. **Throughput unit is MACs (`m·rows·k`),
  reported by criterion as `Gelem/s`** — this is the compute-rate basis,
  *not* the GEMV group's weight-bytes-per-call basis, so the two groups'
  numbers are not directly comparable. ggml has no fused M-panel GEMM
  over the dlopen CPU ABI, so its baseline here is its per-row
  `vec_dot` called `m×rows` times (no weight reuse across tokens) — i.e.
  the honest "does our batched kernel beat naive per-row" comparison.
- **Environment caveat:** 8-CPU cgroup-quota'd shared devpod. These are
  single-threaded µbenches (one criterion thread), so the quota does not
  distort them; both inferno and ggml ran on the same box, so the
  **inferno/ggml ratio** is the meaningful figure, not the absolute
  `Gelem/s` (which wobbles with host contention).

Hot Q8_0 shapes — median throughput (`Gelem/s`), inferno-avx2 vs ggml:

| shape (rows×k)         | role (Qwen2.5-0.5B)      | m  | inferno-avx2 | ggml (per-row) | ratio |
|------------------------|--------------------------|----|--------------|----------------|-------|
| 896×896                | attn q/o_proj            | 1  | 33.5         | 23.1           | 1.45× |
| 896×896                |                          | 16 | 40.6         | 23.1           | 1.76× |
| 896×896                |                          | 64 | 40.0         | 22.8           | 1.75× |
| 4864×896               | ffn gate/up_proj         | 1  | 34.1         | 23.0           | 1.48× |
| 4864×896               |                          | 16 | 40.8         | 23.2           | 1.76× |
| 4864×896               |                          | 64 | 40.0         | 23.0           | 1.74× |
| 896×4864               | ffn down_proj            | 1  | 37.1         | 24.4           | 1.52× |
| 896×4864               |                          | 16 | 40.9         | 24.6           | 1.66× |
| 896×4864               |                          | 64 | 38.5         | 24.6           | 1.56× |
| 151936×896             | lm_head / embed          | 1  | 14.3         | 13.6           | 1.06× |
| 151936×896             |                          | 16 | 39.0         | 13.5           | 2.88× |
| 151936×896             |                          | 64 | 39.6         | 13.4           | 2.95× |

Llama-3-8B shapes (context, same run):

| shape (rows×k) | m  | inferno-avx2 | ggml | ratio |
|----------------|----|--------------|------|-------|
| 4096×4096      | 1  | 16.7         | 15.4 | 1.09× |
| 4096×4096      | 16 | 40.6         | 15.6 | 2.61× |
| 4096×4096      | 64 | 39.6         | 15.8 | 2.51× |
| 14336×4096     | 1  | 14.6         | 13.6 | 1.07× |
| 14336×4096     | 16 | 40.3         | 13.8 | 2.92× |
| 14336×4096     | 64 | 39.7         | 13.5 | 2.95× |

**Reading:** the M-loop win is exactly the pre-registered weight-reuse
effect. At `m=1` (GEMV-equivalent) inferno-avx2 is memory-bound
(~14–17 `Gelem/s` on the large-`k` shapes, ~1.0–1.5× ggml); at `m≥16`
the packed weights are re-read once and reused across the panel, so
inferno jumps to a compute-bound **~40 `Gelem/s` (2.5–3.0× ggml's flat
per-row rate)**. The batched kernel therefore reaches its intended
regime — the tiling *kernel* is not the problem (see the prefill data
point below for why the end-to-end bar is still missed). The small
896×896 shapes plateau lower at `m=64` (~40 vs the large shapes' ~40 —
scalar-tail / L2-residency limited) but still hold ~1.75× over ggml.

### 2026-07-07 — Prefill exit-criterion data point (Phase 4, Task 10 Step 3) — **CRITERION MISSED (0.55×)**

- **Commit:** recorded by the `bench(kernels): batched GEMM criterion
  group; record M4b.2 prefill data point` commit on branch
  `m4b2-per-thread-gap`. Binary built at inferno git `815c40c` (the
  prefill-tiling commit `feat(codegen): tile prefill into batched GEMM
  passes via inferno_par_gemm`); llama.cpp build `6f4f53f`.
- **Model:** `qwen2.5-0.5b-instruct-q8_0.gguf`, **`--threads 1`** (the
  per-thread criterion), pp=512 tg=128 reps=5.
- **Command:** `mise run bench -- <model> --threads 1 --json` (and the
  same without `--json` for the table). At `--threads 1` the CLI's
  `*_t1_*` diagnostic fields are `null` because the primary rows already
  *are* the t=1 measurement.
- **Environment caveat:** 8-CPU cgroup-quota'd shared devpod. The run is
  single-threaded, so the quota does not invalidate the measurement, but
  both inferno and llama.cpp were measured on the **same** quota'd box —
  so the **ratio** `inferno_t1_pp / llama_t1_pp` is the meaningful
  figure, not the absolute throughput.

Table run:

```
model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: BLAS, AMD Ryzen 9 3900 12-Core Processor (12 physical / 24 logical cores)
inferno 0.1.0 (815c40c) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)           1       25.64 ± 0.54         9.99 ± 0.18
llama.cpp                    1       46.08 ± 0.61        23.03 ± 0.19

ratio (inferno/llama.cpp): pp 0.56x | tg 0.43x
```

JSON run:

```json
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "BLAS, AMD Ryzen 9 3900 12-Core Processor",
  "physical_cores": 12,
  "logical_cores": 24,
  "inferno_version": "0.1.0",
  "inferno_git": "815c40c",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 5,
  "inferno_threads": 1,
  "llama_threads": 1,
  "inferno_pp_tok_s": 26.050713619604807,
  "inferno_pp_stddev": 0.6811958050660312,
  "inferno_tg_tok_s": 10.017551243569457,
  "inferno_tg_stddev": 0.49853228633740476,
  "llama_pp_tok_s": 47.522877,
  "llama_pp_stddev": 0.819074,
  "llama_tg_tok_s": 23.473513,
  "llama_tg_stddev": 1.177595,
  "llama_t1_pp_tok_s": null,
  "llama_t1_pp_stddev": null,
  "llama_t1_tg_tok_s": null,
  "llama_t1_tg_stddev": null,
  "inferno_t1_pp_tok_s": null,
  "inferno_t1_pp_stddev": null,
  "inferno_t1_tg_tok_s": null,
  "inferno_t1_tg_stddev": null
}
```

**Prefill ratio (the exit criterion):**
`inferno_t1_pp / llama_t1_pp = 26.05 / 47.52 =` **0.548×** (json) /
`25.64 / 46.08 =` **0.556×** (table). **Exit criterion is t=1 pp ≥
0.70× llama.cpp t=1 → NOT MET (0.55×).** Context: the pre-tiling M4b.1
t=1 baseline was pp ~14.5–15.1 tok/s; the tiling roughly **1.7×'d**
single-thread prefill (→ ~25.6–26.1), a real gain — but still short of
the bar. Decode (tg) is unchanged at ~0.43× and is out of scope for this
criterion (its lever is the pre-registered Phase-5 decode-attribution
fork).

Post-tiling **T=64 `--profile`** prefill capture (same model,
`--threads 1`, `--max-tokens 64`, random base64 prompt ~1.3 K tokens),
verbatim prefill table:

```
profile [prefill] 275.153s wall, 852911284860 cyc total
  op                                   cycles   share        GB/s
  attention                      602045421090   70.6%           -
  matmul:lm_head.weight           65031724814    7.6%        15.0
  matmul:layers.*.ffn.gate_proj.weight    51769603107    6.1%        14.4
  matmul:layers.*.ffn.up_proj.weight    51678143263    6.1%        14.5
  matmul:layers.*.ffn.down_proj.weight    51193302762    6.0%        14.6
  matmul:layers.*.attn.o_proj.weight     9646703975    1.1%        14.3
  matmul:layers.*.attn.q_proj.weight     9457546528    1.1%        14.5
  swiglu                           4806648981    0.6%           -
  rope                             1539885244    0.2%           -
  matmul:layers.*.attn.k_proj.weight     1407417006    0.2%        14.0
  matmul:layers.*.attn.v_proj.weight     1384686710    0.2%        14.2
  rmsnorm                          1375523429    0.2%           -
  add                               672059823    0.1%           -
  embed                             522807053    0.1%           -
  bias                              369345978    0.0%           -
  quantize                           10465097    0.0%           -
```

**Profile annotation (carry from Task 9 review):** in the tiled prefill
the `Step::Quantize` op is folded into the matmul lowering, so its cost
is attributed to the `matmul:*` slots and the standalone `quantize` slot
shows ~0 (10.5 M cyc, 0.0%). Quantization is **not** free — read it as
part of the matmul slots. Likewise the prefill `GB/s` column is the
documented over-count (weight bytes × per-token invocations; the batched
kernel actually re-reads each weight once per tile, not once per token),
so the ~14–15 `GB/s` figures are a *lower bound* on real matmul
efficiency, not the achieved bandwidth — they improved from the
pre-tiling baseline's ~12.5 (tiling is doing its job at the matmul).

**Attribution — why the bar is missed:** the profile blames
**attention, at 70.6% of prefill cycles** (up from the 68.5%
pre-tiling baseline — its *share* grew precisely because the GEMM
tiling shrank the matmuls, the pre-registered risk). The GEMM matmuls
are only **28.3% of prefill combined**; the tiling already moved them
into their compute-bound regime (µbench above; effective GB/s
12.5→~14.5). Arithmetic ceiling: even reducing the *entire* matmul cost
to zero caps t=1 prefill at `25.64 / (1 − 0.283) = 35.8 tok/s = 0.78×` —
and no realistic register-blocking removes all of it. **The dominant
prefill lever is the serial attention path, not the GEMM.**

#### Scoped follow-up amendment (register-blocked GEMM escalation — NOT done here)

Per the spec's pre-registered escalation clause (§"Escalation clause"),
the prefill miss triggers a **scoped follow-up task** (not started in
Task 10; no optimization work was begun):

1. **Register-blocked GEMM tiles** for the profile-blamed matmul shapes:
   the FFN projections **4864×896** (gate/up, `k=896`) and **896×4864**
   (down, `k=4864`), and **lm_head 151936×896** — together ~26% of
   prefill. Current `gemm_*_rs8` batches the M-loop but keeps a scalar
   accumulator tail per row-strip; a register-blocked micro-kernel
   (e.g. 4×4 or 8×4 `m×row` register tiles holding partial sums in
   YMM/`__m256`) would raise the large-`k` shapes from the µbench's
   ~40 `Gelem/s` toward the AVX2 FMA roofline and shave the matmul
   share. Bit-identity bar unchanged (scalar-vs-tiled and
   across-thread, per the existing rig).
2. **Bounded headroom, explicit:** by the arithmetic above this lever
   *alone* tops out at ~0.78× and realistically lands ~0.62–0.66×, so it
   is **necessary but not sufficient** to clear 0.70×. The larger lever
   is a **prefill-attention optimization** (the serial 70.6% slice).
   That touches the attention path — gated by the spec's out-of-scope
   note on parallel attention and the M4b.1 bare-metal re-measurement —
   and must be scoped as its own task against a fresh profile, not
   silently folded into the GEMM escalation.

This amendment records the fork; it does **not** authorize either lever.
The register-blocked GEMM task and the prefill-attention task are each
their own follow-up with their own plan.
