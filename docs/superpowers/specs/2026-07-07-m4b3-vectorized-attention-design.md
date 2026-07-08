# M4b.3 — Vectorized Attention Kernel Design

**Date:** 2026-07-07
**Status:** Approved design, pre-implementation
**Milestone:** M4b.3 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.2](2026-07-07-m4b2-per-thread-gap-design.md))

M4b.2 tiled prefill into batched GEMM and moved the matmuls into their
compute-bound regime (µbench 2.5–3.0× ggml; effective GB/s 12.5→~14.5),
but its t=1 prefill exit criterion **missed at 0.55×** (target ≥0.70×).
The recorded post-tiling profile reassigned the blame: with the matmuls
shrunk, **attention is 70.6% of prefill cycles** (up from 68.5%
pre-tiling — its *share* grew precisely because the GEMM tiling worked)
and **81.5% of decode cycles**. GEMM matmuls are only 28.3% of prefill
combined; the M4b.2 arithmetic ceiling shows that zeroing the *entire*
matmul cost still caps t=1 prefill at 0.78×. **The dominant lever for
both phases is the attention path, not the GEMM.**

The cause is structural and confirmed in the source. `lower_attention`
(`inferno-codegen/src/llvm/ops.rs`) emits attention as **fully scalar
inline LLVM IR** — a per-head scalar QK dot over `head_dim`, a scalar
max-subtraction softmax with a per-element `exp`, and a scalar AV
accumulate — run **once per query token** in prefill. It is the one hot
math path in the engine that never became a kernel: every matmul goes
through a tuned AVX2 `inferno-kernels` C-ABI symbol; attention is raw
scalar IR with no SIMD and no `KernelSet`. llama.cpp runs a vectorized
attention with a polynomial-exp softmax. M4b.3 closes that specific gap.

What this milestone rules in and out:

- **Vectorization is the lever; it is not gated.** The specs gate
  *parallel* (multi-thread) attention on the M4b.1 bare-metal rerun.
  Single-threaded SIMD vectorization is a different axis and is **not**
  gated — it lives entirely inside the t=1 per-thread criterion M4b.2
  already validated as meaningful on the CPU-quota'd devpod. M4b.3 stays
  single-threaded on purpose.
- **Attention is f32-only.** q/k/v activations and the KV cache are all
  f32 (KV stays f32 in M3+; see AGENTS.md). Unlike the matmul kernels,
  attention has **no dtype axis** — it is a single kernel family
  dispatched purely by ISA (`scalar` / `avx2`). This is simpler than the
  gemv/gemm path.
- **Prefill stays per-token.** The query-panel/flash-style blocking that
  would rewrite the causal-horizon loop is explicitly out of scope
  (below); it is a larger, riskier lever left for a later milestone.
  Vectorizing the per-token kernel already attacks the 70.6% / 81.5%
  slice and helps **both** phases with one piece of work.

## Scope Decisions (M4b.3)

| Decision | Choice |
|---|---|
| Measurement basis | t=1-vs-t=1 under the standard protocol (devenv shell, release, pinned Q8_0 model). Single-threaded throughout; no threading/scaling claim is made, so nothing here is gated on the M4b.1 bare-metal rerun |
| Kernel boundary | Attention becomes a C-ABI kernel family in `inferno-kernels` (`inferno_attention_f32_{scalar,avx2}`), f32-only, ISA-dispatched. `lower_attention` stops emitting the scalar triple-loop and instead declares + calls the symbol, the path `lower_gemv` already uses |
| `exp` implementation | **Vectorized polynomial `exp`** (minimax poly, ggml/SLEEF-style), evaluated **identically** in the scalar and AVX2 kernels. This is the source of the softmax speedup and is what llama.cpp does |
| Correctness — kernel vs kernel | **scalar-kernel vs AVX2-kernel bit-identical** (exact equality), asserted by the `inferno-kernels` rig — achievable because both share the poly `exp` and the same lane-partitioned reduction order. Extends the existing scalar-vs-SIMD invariant |
| Correctness — kernel vs interpreter | **Tolerance-bounded** by a new `attn_rel_tol` constant in `inferno-graph/src/tolerance.rs`, **derived from an observed error-distribution sweep** (an ignored `observed_error_*` diagnostic, same discipline as `gemv_rel_tol` / `LOGIT_TIE_EPSILON`) — never hand-set to make a test green |
| Exit criterion | **No hard 0.70× on attention alone** (M4b.2 showed attention + the deferred GEMM register-blocking are jointly necessary). Instead: (1) a fresh t=1 `--profile` capture showing attention is **no longer the majority** of prefill (target <45%, from 70.6%); (2) a recorded t=1 bench data point with **both** pp and tg improved. The 0.70× decision moves to a follow-up that re-evaluates the GEMM lever on top |
| Standing invariants | scalar-vs-SIMD kernel bit-identity (now extended to attention); compiled-vs-interpreter differential green with **no `logits_abs_tol` loosening**; the interpreter stays the independent std-`exp` oracle, unchanged |

**Explicitly out of scope:** parallel / multi-thread attention (gated on
the M4b.1 bare-metal rerun); prefill query-panel / flash-style blocking
(the causal-horizon rewrite — a later milestone); F16 KV cache (its own
decode-fork amendment, re-derives tolerances); AVX-512 attention bodies
(the `Isa::X86_64v4` detection stays ready); the register-blocked GEMM
escalation (M4b.2's separate follow-up); any change to the decode-fork
tolerances; any CI perf gate (never — AGENTS.md).

## What M4b.3 Adds

- **`inferno-kernels`** — new `inferno_attention_f32_scalar` /
  `inferno_attention_f32_avx2` C-ABI kernels (`#[unsafe(no_mangle)]
  extern "C"`); a shared vectorized polynomial-`exp` helper used by both;
  the `KernelSet`/registry selector extended so `reference_kernels` /
  `kernels_for` surface the attention fn pointer per ISA; rig coverage
  (scalar-vs-AVX2 bit-identity + kernel-vs-interpreter tolerance sweep).
- **`inferno-codegen`** — `lower_attention` rewritten to declare + call
  the ISA-selected symbol instead of emitting the scalar triple-loop; the
  existing `scores` `entry_alloca` passed through as the kernel's scratch
  pointer; the profiler's `attention` slot now times the call.
- **`inferno-graph`** — new `attn_rel_tol` tolerance constant (with its
  derivation note); the reference `ops::attention` is **unchanged**.

## Kernel Boundary & ABI

Attention today is inline IR that reads q/out via arena pointers, appends
this token's k/v into the f32 KV cache at `pos`, and reads a `scores`
stack buffer. As a kernel it takes those as explicit parameters. The
proposed C-ABI signature (final arg order firmed at plan time):

```
unsafe extern "C" fn inferno_attention_f32_{isa}(
    out:        *mut f32,    // this token's output rows [n_heads * head_dim]
    q:          *const f32,  // this token's query rows  [n_heads * head_dim]
    kv:         *mut f32,    // KV cache base pointer
    scores:     *mut f32,    // caller scratch, >= seq_len f32
    kv_base:    usize,       // per-layer K region offset (elements)
    v_off:      usize,       // V region offset from kv_base (= seq_len*kv_dim)
    pos:        usize,       // current position (causal horizon = pos+1)
    kv_dim:     usize,       // n_kv_heads * head_dim
    n_heads:    usize,
    n_kv_heads: usize,
    head_dim:   usize,
)
```

The kernel does the KV append (write this token's k/v at `pos`) and the
read (per-head scored softmax-weighted V sum), matching
`inferno_graph::ops::attention` op-for-op except for the vectorized `exp`
and lane-partitioned reductions. `scale = 1/sqrt(head_dim)` is derived
inside the kernel from `head_dim` (kept out of the ABI). `k`/`v` for this
token are read from the arena rows the codegen currently uses; the plan
step resolves whether they arrive as one fused `q|k|v` pointer or three.
Codegen selects the symbol from the same `Isa` it already uses for
gemv/gemm and calls it via `get_function(sym).build_call(...)`.

## Kernel Internals (AVX2)

Per query row, per head (`head_dim=64` on the pinned Qwen2.5-0.5B = eight
`__m256` lanes-of-8), the two-pass structure mirrors the interpreter so
the kernel is a faithful drop-in:

1. **QK-dot** — vectorized FMA accumulate of `q_head · kcache[t,g]` over
   `head_dim`, horizontal-reduce per visible position `t in 0..=pos`,
   scale → `scores[t]`.
2. **softmax** — vectorized max over `scores[..visible]`; then vectorized
   `(scores[t] − max)` fed to the **vectorized polynomial `exp`**;
   vectorized sum → `denom`.
3. **AV** — vectorized FMA accumulate of `(scores[t]/denom) · vcache[t,g]`
   into `out_head` over `head_dim`.

GQA mapping (`g = h / (n_heads/n_kv_heads)`) is unchanged. The **scalar
reference kernel** runs the identical polynomial `exp` and the identical
lane-partitioned reduction order, so the two kernels are bit-identical.

## Correctness Contract

Three layers, mirroring the gemm precedent:

1. **scalar-kernel vs AVX2-kernel → exact bit-identity**, asserted in the
   `inferno-kernels` rig over the property shape distribution. This is the
   flagship invariant (AGENTS.md); it holds here because both kernels
   share the poly `exp` and reduction tree.
2. **kernel vs interpreter (std-`exp` oracle) → `attn_rel_tol`**, a new
   constant in `inferno-graph/src/tolerance.rs` bounded by an ignored
   `observed_error_*` sweep across the shape distribution, exactly like
   `gemv_rel_tol`. The interpreter (`ops::attention`) is the ground truth
   and stays on `std::f32::exp` — it is **not** modified.
3. **End-to-end → `logits_abs_tol`** (unchanged: 1e-2 Q8_0). The
   compiled-vs-interpreter differential (`inferno-codegen --test
   differential`) and the artifact differential (`inferno-core --test
   artifact`) must stay green **with no loosening**. A degree-≈6 minimax
   `exp` is ~1 ULP (~1e-7 relative); propagated through a *normalizing*
   softmax and summed into logits, that sits orders of magnitude under
   the 1e-2 budget. If any differential goes red, the poly or reduction
   is wrong — the tolerance is never the fix.

## Measurement Protocol

All at t=1, devenv shell, release build, pinned
`qwen2.5-0.5b-instruct-q8_0.gguf`, results recorded in `## Amendments`
(never edited once recorded):

- **Kernel µbench** — an attention criterion group in
  `inferno-kernels/benches` over the pinned model's head geometry
  (`head_dim=64`, `n_heads=14`, `n_kv_heads=2`) at representative
  horizons, scalar vs AVX2, so the SIMD win is visible per-position.
- **`--profile` capture** — t=1 prefill + decode profile after the swap;
  exit-criterion check is attention share < 45% of prefill.
- **Bench data point** — `mise run bench -- <model> --threads 1 --json`;
  record the table + json and the pp/tg ratios vs llama.cpp t=1. Both pp
  and tg must improve over M4b.2's recorded 0.55× / 0.43×.

## Gated Follow-Ups (not tasks in this milestone)

- **Register-blocked GEMM escalation** (inherited from M4b.2). After
  M4b.3's profile exists, a follow-up decides whether register-blocked
  GEMM tiles on top of vectorized attention are needed to reach 0.70×
  prefill. Its own spec amendment / plan.
- **Prefill query-panel (flash-style) attention.** The causal-horizon
  rewrite that batches the query panel against shared K/V with online
  softmax — the bigger prefill lever, deferred here for risk. Own plan.
- **Parallel attention.** Multi-threading the per-head loop stays gated
  on the M4b.1 bare-metal rerun.
- **AVX-512 / F16 KV.** Unchanged from their existing gates.

## Amendments

*(Recorded protocol data points and scoped amendments land here; never
edit a recorded data point.)*

### 2026-07-08 — M4b.3 exit-criterion data point (Task 10) — **CRITERION MET**

- **Commit:** recorded by the `bench: record M4b.3 vectorized-attention
  data point + profile share` commit on branch `m4b3-vectorized-attention`,
  built at inferno git `7937b8d` (`bench(kernels): attention criterion
  group (scalar vs avx2, pinned geometry)`, the last Task-9 commit).
  llama.cpp build `6f4f53f` (same pin as M4b.2).
- **Model:** `qwen2.5-0.5b-instruct-q8_0.gguf`, **`--threads 1`** throughout
  (the per-thread protocol), release build, all runs inside `devenv shell`.
- **Environment caveat (read before the numbers):** 8-CPU cgroup-quota'd
  shared devpod, not quiet bare metal. All three measurements below are
  single-threaded, so the quota does not invalidate them structurally, but
  absolute throughput/cycles wobble run-to-run with host contention (the
  pp/tg bench below shows this directly — three reps of the same command
  gave `pp` 61.98–66.55 tok/s). Per AGENTS.md/M4a protocol: **ratios and
  cycle-shares are the meaningful figures**, not absolute numbers.
- **Cache-artifact pitfall found and worked around:** the very first
  `--profile` attempt (against a pre-existing, non-empty
  `~/.cache/inferno/` from earlier work on this devpod) produced a
  suspicious table — attention still 66.4% of prefill / 82.8% of decode,
  barely down from M4b.2's 70.6%/81.5%, which did not square with the
  µbench's 25–28× kernel speedup. Inspecting the cache directory's
  `meta.json` showed `"profile_slots": []` — an artifact whose profiler
  instrumentation never got populated, i.e. a stale/inconsistent cached
  compile (the on-disk cache key is `sha256(model bytes, target, max_seq_len,
  CARGO_PKG_VERSION, HOST_ABI_VERSION, profile flag, prefill_tile)` —
  content-addressed against source *inputs*, not against inferno's own
  source code, so nothing forces a rebuild when only `lower_attention`'s
  codegen changes and `HOST_ABI_VERSION` isn't bumped for it). Worked
  around by `rm -rf ~/.cache/inferno` and reproducing from a from-scratch
  compile; the fresh artifact's `meta.json` showed a correctly populated
  `profile_slots` list (16 named ops including `attention`), and the
  resulting profile matches the µbench's speedup story (below). All
  numbers in this amendment are from that from-scratch compile,
  cross-checked by two independent runs with different random prompts
  (26.4% and 26.8% prefill share — consistent). This is a real gap in the
  cache-key's source-sensitivity worth a follow-up note (see end of this
  amendment) but is **not** fixed here — Task 10 only measures.

#### 1. Kernel µbench (`cargo bench -p inferno-kernels --bench attention`)

Raw output (scalar vs avx2, `pos ∈ {15, 127, 511}`):

```
Benchmarking attention/scalar/15
attention/scalar/15    time:   [69.999 µs 70.280 µs 70.642 µs]
                        thrpt:  [12.684 Melem/s 12.749 Melem/s 12.800 Melem/s]
Benchmarking attention/avx2/15
attention/avx2/15      time:   [2.7285 µs 2.7551 µs 2.7842 µs]
                        thrpt:  [321.81 Melem/s 325.22 Melem/s 328.38 Melem/s]
Benchmarking attention/scalar/127
attention/scalar/127   time:   [560.03 µs 562.34 µs 564.89 µs]
                        thrpt:  [1.5861 Melem/s 1.5933 Melem/s 1.5999 Melem/s]
Benchmarking attention/avx2/127
attention/avx2/127     time:   [20.015 µs 20.212 µs 20.451 µs]
                        thrpt:  [43.813 Melem/s 44.329 Melem/s 44.766 Melem/s]
Benchmarking attention/scalar/511
attention/scalar/511   time:   [2.2431 ms 2.2546 ms 2.2679 ms]
                        thrpt:  [395.07 Kelem/s 397.41 Kelem/s 399.44 Kelem/s]
Benchmarking attention/avx2/511
attention/avx2/511     time:   [87.212 µs 88.358 µs 89.622 µs]
                        thrpt:  [9.9975 Melem/s 10.141 Melem/s 10.274 Melem/s]
```

Scalar-vs-AVX2 ratio (median thrpt, the meaningful figure on this box):

| pos | scalar (Melem/s) | avx2 (Melem/s) | ratio |
|-----|-------------------|-----------------|-------|
| 15  | 12.749            | 325.22          | 25.5× |
| 127 | 1.5933            | 44.329          | 27.8× |
| 511 | 0.39741           | 10.141          | 25.5× |

Consistent **~25–28× AVX2 speedup** across the horizon range — the
vectorized polynomial-`exp` softmax plus lane-partitioned QK/AV FMA
accumulation is doing real work, not just noise.

#### 2. `--profile` capture (the exit-gate signal)

Command: `cargo run --release -p inferno -- run --profile --threads 1
--max-tokens 64 --prompt <random ~1.3K-token base64 prompt>
/home/dev/.cache/inferno-tests/qwen2.5-0.5b-instruct-q8_0.gguf` (prompt
generated the same way as M4b.2: `head -c 2048 /dev/urandom | base64`).
Run twice with independent random prompts against the same verified
from-scratch artifact.

**Run A** — verbatim:

```
profile [prefill] 39.014s wall, 120938497358 cyc total
  op                                   cycles   share        GB/s
  attention                       31930759863   26.4%           -
  matmul:lm_head.weight           22155216397   18.3%        44.2
  matmul:layers.*.ffn.down_proj.weight    17713019873   14.6%        42.5
  matmul:layers.*.ffn.up_proj.weight    16986408894   14.0%        44.3
  matmul:layers.*.ffn.gate_proj.weight    16979192560   14.0%        44.3
  swiglu                           4384715267    3.6%           -
  matmul:layers.*.attn.o_proj.weight     3147187730    2.6%        44.0
  matmul:layers.*.attn.q_proj.weight     3140224467    2.6%        44.1
  rmsnorm                          1330122902    1.1%           -
  rope                             1222671928    1.0%           -
  add                               600482513    0.5%           -
  matmul:layers.*.attn.v_proj.weight      485855767    0.4%        40.7
  matmul:layers.*.attn.k_proj.weight      485732598    0.4%        40.7
  bias                              355568958    0.3%           -
  embed                              21337641    0.0%           -
  quantize                                  0    0.0%           -
profile [decode] 3.141s wall, 9645616510 cyc total
  op                                   cycles   share        GB/s
  attention                        2553998046   26.5%           -
  matmul:lm_head.weight            1870098835   19.4%        16.1
  matmul:layers.*.ffn.gate_proj.weight     1454995702   15.1%        15.9
  matmul:layers.*.ffn.up_proj.weight     1450813158   15.0%        15.9
  matmul:layers.*.ffn.down_proj.weight     1448748058   15.0%        16.0
  matmul:layers.*.attn.q_proj.weight      268726767    2.8%        15.9
  matmul:layers.*.attn.o_proj.weight      267562473    2.8%        15.9
  swiglu                            134905899    1.4%           -
  rope                               45011490    0.5%           -
  rmsnorm                            41130410    0.4%           -
  matmul:layers.*.attn.v_proj.weight       40167940    0.4%        15.1
  matmul:layers.*.attn.k_proj.weight       39639494    0.4%        15.4
  add                                17801531    0.2%           -
  bias                               11217314    0.1%           -
  embed                                453933    0.0%           -
  quantize                             345460    0.0%           -
```

**Run B** — independent random prompt, same artifact, verbatim:

```
profile [prefill] 42.728s wall, 132451277301 cyc total
  op                                   cycles   share        GB/s
  attention                       35509922638   26.8%           -
  matmul:lm_head.weight           24144491639   18.2%        41.1
  matmul:layers.*.ffn.down_proj.weight    19393555764   14.6%        39.3
  matmul:layers.*.ffn.up_proj.weight    18495215567   14.0%        41.2
  matmul:layers.*.ffn.gate_proj.weight    18404963894   13.9%        41.4
  swiglu                           4803399437    3.6%           -
  matmul:layers.*.attn.o_proj.weight     3471641946    2.6%        40.5
  matmul:layers.*.attn.q_proj.weight     3396003927    2.6%        41.4
  rmsnorm                          1418167686    1.1%           -
  rope                             1316987718    1.0%           -
  add                               641728638    0.5%           -
  matmul:layers.*.attn.k_proj.weight      530468806    0.4%        37.8
  matmul:layers.*.attn.v_proj.weight      525283443    0.4%        38.2
  bias                              375827267    0.3%           -
  embed                              23618931    0.0%           -
  quantize                                  0    0.0%           -
profile [decode] 3.200s wall, 9825591252 cyc total
  op                                   cycles   share        GB/s
  attention                        2653183758   27.0%           -
  matmul:lm_head.weight            1917319579   19.5%        15.7
  matmul:layers.*.ffn.gate_proj.weight     1462239969   14.9%        15.8
  matmul:layers.*.ffn.up_proj.weight     1459017970   14.8%        15.8
  matmul:layers.*.ffn.down_proj.weight     1458736134   14.8%        15.9
  matmul:layers.*.attn.q_proj.weight      270633156    2.8%        15.7
  matmul:layers.*.attn.o_proj.weight      268707368    2.7%        15.9
  swiglu                            138173473    1.4%           -
  rope                               44914936    0.5%           -
  rmsnorm                            41627110    0.4%           -
  matmul:layers.*.attn.v_proj.weight       40011576    0.4%        15.2
  matmul:layers.*.attn.k_proj.weight       39590813    0.4%        15.4
  add                                18309357    0.2%           -
  bias                               11253155    0.1%           -
  embed                                1530439    0.0%           -
  quantize                             342459    0.0%           -
```

**Attention prefill cycle share: 26.4% (Run A) / 26.8% (Run B) — down
from the pre-M4b.3 baseline's 70.6%. Exit criterion is < 45% → MET, with
a wide margin.** Decode share also fell, from 81.5%/82.8%(stale) to
26.5%/27.0% — attention is no longer the majority of either phase.
GEMM/matmul now dominates prefill instead (Run B: 18.2 + 14.6 + 14.0 +
13.9 + 2.6 + 2.6 + 0.4 + 0.4 = **66.7% combined**, up from M4b.2's 28.3%
— the expected mirror-image of attention's collapse, and exactly the
"passes to the GEMM follow-up" scenario the spec anticipated). The
`GB/s` figures also roughly tripled (11–15 → 37–44) versus the M4b.2
baseline profile; per the M4b.2 amendment's own caveat this column is a
diagnostic over-count (weight bytes × per-token invocations, not exact
achieved bandwidth for tiled prefill) and the absolute jump is likely
dominated by the ~8× total-cycle drop (less host contention during this
shorter run) rather than a GEMM change — M4b.3 touched no GEMM code.

#### 3. `mise run bench -- <model> --threads 1 --json` (t=1 vs llama.cpp)

Three independent reps (same command, run separately to characterize
the box's noise — see environment caveat above):

**Rep 1 (JSON):**

```json
{
  "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
  "model_type": "qwen2 1B Q8_0",
  "cpu_info": "BLAS, AMD Ryzen 9 3900 12-Core Processor",
  "physical_cores": 12,
  "logical_cores": 24,
  "inferno_version": "0.1.0",
  "inferno_git": "7937b8d",
  "llama_build_commit": "6f4f53f",
  "pp": 512,
  "tg": 128,
  "reps": 5,
  "inferno_threads": 1,
  "llama_threads": 1,
  "inferno_pp_tok_s": 66.54984327875374,
  "inferno_pp_stddev": 0.21623910731322116,
  "inferno_tg_tok_s": 25.284298806979884,
  "inferno_tg_stddev": 0.29310806660907834,
  "llama_pp_tok_s": 50.002651,
  "llama_pp_stddev": 0.95185,
  "llama_tg_tok_s": 23.256662,
  "llama_tg_stddev": 0.839261,
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

Ratio: pp `66.550 / 50.003 =` **1.331×**; tg `25.284 / 23.257 =` **1.087×**.

**Rep 2 (table form):**

```
model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: BLAS, AMD Ryzen 9 3900 12-Core Processor (12 physical / 24 logical cores)
inferno 0.1.0 (7937b8d) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)           1       62.57 ± 3.71        21.74 ± 4.50
llama.cpp                    1       49.67 ± 1.44        24.66 ± 0.33

ratio (inferno/llama.cpp): pp 1.26x | tg 0.88x
```

**Rep 3 (JSON):**

```json
{
  "inferno_pp_tok_s": 61.97661005254555,
  "inferno_pp_stddev": 0.2943279500988874,
  "inferno_tg_tok_s": 20.958590880816054,
  "inferno_tg_stddev": 0.16060095940288494,
  "llama_pp_tok_s": 46.282899,
  "llama_pp_stddev": 1.204776,
  "llama_tg_tok_s": 22.523049,
  "llama_tg_stddev": 0.251782
}
```
(full JSON identical shape to Rep 1, other fields unchanged; only the
throughput fields shown here for brevity — the complete Rep-1 JSON above
records every field this schema has.) Ratio: pp `61.977 / 46.283 =`
**1.339×**; tg `20.959 / 22.523 =` **0.931×**.

**pp ratio across reps: 1.26×–1.34× (inferno now faster than llama.cpp
t=1 prefill). tg ratio across reps: 0.88×–1.09× (roughly parity, noisy
around 1.0×).** Both ranges sit **far above** M4b.2's recorded 0.55× pp /
0.43× tg — pp more than doubled its ratio and now exceeds 1×; tg roughly
doubled its ratio too. **Exit criterion "both pp and tg improve over
0.55×/0.43×" → MET, unambiguously** (the tg range's low end, 0.88×, is
still more than double 0.43×; the noise band does not touch the old
baseline at any rep).

#### Verdict

**Both M4b.3 exit-criterion legs are MET:**

1. Attention prefill cycle share: **26.4–26.8%**, target < 45% — met with
   a wide margin (not a borderline pass).
2. pp/tg ratio vs llama.cpp t=1: **pp 1.26×–1.34×, tg 0.88×–1.09×**, both
   far above M4b.2's 0.55×/0.43× — met.

Per the spec's exit-criterion clause: the residual prefill cost is no
longer attention-dominated. GEMM/matmul is now ~66.7% of prefill cycles
(up from 28.3% pre-M4b.3), so the **register-blocked GEMM escalation**
already gated as a separate follow-up in the M4b.2 amendment is the
indicated next lever *if* further prefill gains are wanted — but pp is
already 1.26×–1.34× llama.cpp t=1, well past M4b.2's original 0.70×
target, so that follow-up is now a "push further" optimization, not a
gap-closing one. **This amendment does not start that work** — it is
recorded here as context for whoever picks up the GEMM follow-up's own
plan, per the spec's "no silent scope" rule.

**Secondary observation (not a new task, just recorded):** the artifact
cache-key pitfall found above (source-insensitive `HOST_ABI_VERSION`
across `lower_attention`'s codegen rewrite) is a latent trap for anyone
re-running `--profile`/`bench` against a devpod with a pre-existing
`~/.cache/inferno/` from before this milestone's code landed — it will
silently `dlopen` a stale artifact instead of recompiling, with no error.
Worth a scoped follow-up (e.g. bump `HOST_ABI_VERSION` per milestone, or
fold a codegen source hash into the cache key) but that is a cache/build
hygiene fix, out of scope for this measurement task.
