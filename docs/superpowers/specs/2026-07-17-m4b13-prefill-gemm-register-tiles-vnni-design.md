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

## Amendments

### 2026-07-17 — Lever 1 dev data point (Task 3 local gate)

Dev box (AMD Ryzen 9 3900, Zen 2, 12c/24t, shared devpod host), not quiet
hardware — local gate only, never the exit criterion.

**Methodology note:** the devpod's host carried heavy neighbor load during
the session (host loadavg 35–44 on 24 cores, no load inside this
container), so single-shot criterion runs swung up to ±40% on the largest
shape. The sweep below therefore used four pinned bench binaries
(pre-tiling baseline @ 5deafb7 and the tiled kernel @ 930ef87 built at
INFERNO_GEMM_MR=2/4/8) executed **interleaved, 3 rounds, round-robin**, and
scores each variant by min-of-medians across rounds (least-perturbed run).
Same machine, same session, same binaries throughout.

µbench `gemm/Q8_0/inferno-avx2/*/m64`, min-of-medians, Gelem/s (Δ vs base):

| shape (m=64)   | base @5deafb7 | MR=2           | MR=4               | MR=8           |
|----------------|---------------|----------------|--------------------|----------------|
| 4864x896       | 35.14         | 33.15 (−5.7%)  | **37.64 (+7.1%)**  | 35.39 (+0.7%)  |
| 896x4864       | 29.24         | 33.39 (+14.2%) | **38.17 (+30.5%)** | 38.94 (+33.2%) |
| 151936x896     | 34.51         | 34.87 (+1.0%)  | **34.58 (+0.2%)**  | 33.58 (−2.7%)  |
| geomean Δ      | —             | +2.8%          | **+11.9%**         | +9.3%          |

ggml reference rows (same session, standalone baseline run, m=64):
4864x896 18.75 Gelem/s · 896x4864 19.45 Gelem/s · 151936x896 5.71 Gelem/s.

**Chosen MR = 4** (best geomean; also the shipped default — no source
change). Caveat recorded honestly: on 151936x896 (the LM head) the tiled
kernel only ties baseline (+0.2%, within noise) — that shape is
DRAM-bandwidth-bound at m=64 (weights ≈ 145 MB stream once regardless of
tiling), so the tile win is confined to the cache-resident shapes. The
quiet boxes have different bandwidth-per-core; Task 5 judges the real
effect.

Dev t=1 pp, `mise run bench -- models/qwen2.5-0.5b-instruct-q8_0.gguf`,
`inferno (t=1 diag)` pp512 rows, two interleaved rounds each:

| round | base @5deafb7 | tiled @930ef87 (MR=4) |
|-------|---------------|------------------------|
| 1     | 59.27 ± 1.03  | 63.30 ± 1.39 (+6.8%)  |
| 2     | 58.80 ± 1.95  | 61.73 ± 3.24 (+5.0%)  |

**Local gate verdict: PASS.** Tiled µbench ≥ baseline on all three blamed
shapes at MR=4 (two decisive, one within-noise tie as caveated) AND t=1 pp
improves in both rounds. Metal spend for Task 5 is unblocked.

### 2026-07-17 — mid-milestone gate sessions (Lever 1 on both boxes)

Both sessions bench main `3183e29` (Tasks 1–4 merged, PR #30). Protocol:
`mise run metal` per `docs/runbooks/metal.md`; on-box smoke `verify.sh
--smoke` first, then the real pass; preflight FIT both boxes. Session A
raw: `target/metal/d2.c1.medium-20260717T064329Z` (real pass
`.../quiet-hw/20260717T071428Z`). Session B: PHX 406 no-stock on first
launch (nothing billed; post-failure gc also caught session A's server
whose trap delete had not stuck — deleted, zero confirmed), CHI retry
succeeded; raw: `target/metal/s2.c2.medium-20260717T082513Z` (real pass
`.../quiet-hw/20260717T085021Z`). Zero servers confirmed after both.

#### Session A — d2.c1.medium (Xeon Gold 6336Y, 16c, PHX): gate-bench-protocol.out

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-17T07:56:44Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (16 physical / 32 logical cores)
inferno 0.1.0 (3183e29) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      929.60 ± 10.14       59.78 ± 0.08 
inferno (t=1 diag)           1       73.87 ± 0.04        16.20 ± 0.01 
llama.cpp                   16     1235.24 ± 237.90       62.42 ± 0.31 
llama.cpp (t=1 diag)         1      117.92 ± 0.04        17.59 ± 0.02 

ratio (inferno/llama.cpp): pp 0.75x | tg 0.96x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 515.72 | tg 62.50 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.74x | tg 0.94x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

#### Session A — t=1 prefill profile (gate-prefill-attr.out)

```
# gate-prefill-attr (M4b.13: split-bracket t=1 prefill profile) — 2026-07-17T08:03:09Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

--- t=1 prefill profile (split brackets) ---
profile [prefill] 39.822s wall, 123445100224 cyc total
  op                                   cycles   share        GB/s
  attention                       43763672022   35.5%           -
  matmul:lm_head.weight           19953000980   16.2%        47.6
  matmul:layers.*.ffn.gate_proj.weight    15310093992   12.4%        47.7
  matmul:layers.*.ffn.up_proj.weight    15302367844   12.4%        47.7
  matmul:layers.*.ffn.down_proj.weight    15287421952   12.4%        47.7
  swiglu                           4224858674    3.4%           -
  matmul:layers.*.attn.q_proj.weight     2819448996    2.3%        47.7
  matmul:layers.*.attn.o_proj.weight     2818763890    2.3%        47.7
  rmsnorm                          1178469826    1.0%           -
  rope                              854864282    0.7%           -
  quantize                          526598206    0.4%           -
  matmul:layers.*.attn.v_proj.weight      404686292    0.3%        47.4
  matmul:layers.*.attn.k_proj.weight      404288584    0.3%        47.5
  add                               329769542    0.3%           -
  bias                              133610524    0.1%           -
  kv_append                         113963352    0.1%           -
  embed                              19221266    0.0%           -
profile [decode] 2.539s wall, 7820769910 cyc total
  op                                   cycles   share        GB/s
  attention                        2037731668   26.1%           -
  matmul:lm_head.weight            1553511692   19.9%         9.7
  matmul:layers.*.ffn.down_proj.weight     1199280242   15.3%         9.7
  matmul:layers.*.ffn.up_proj.weight     1196093716   15.3%         9.7
  matmul:layers.*.ffn.gate_proj.weight     1196067018   15.3%         9.7
  matmul:layers.*.attn.o_proj.weight      224876702    2.9%         9.5
  matmul:layers.*.attn.q_proj.weight      222585308    2.8%         9.6
  swiglu                             68002870    0.9%           -
  matmul:layers.*.attn.v_proj.weight       32319758    0.4%         9.4
  matmul:layers.*.attn.k_proj.weight       32144906    0.4%         9.5
  rope                               22017010    0.3%           -
  rmsnorm                            20075134    0.3%           -
  add                                 9105892    0.1%           -
  bias                                6083350    0.1%           -
  embed                                616488    0.0%           -
  quantize                             258156    0.0%           -
  kv_append                                 0    0.0%           -
```

#### Session B — s2.c2.medium (Xeon E-2388G, 8c, CHI): gate-bench-protocol.out

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-17T09:24:02Z
machine: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (GenuineIntel) | 16 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (8 physical / 16 logical cores)
inferno 0.1.0 (3183e29) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)           8      733.34 ± 2.36        62.72 ± 0.05 
inferno (t=1 diag)           1      106.45 ± 0.49        34.97 ± 0.05 
llama.cpp                    8     1040.84 ± 5.69        72.70 ± 0.04 
llama.cpp (t=1 diag)         1      162.62 ± 1.49        33.18 ± 0.14 

ratio (inferno/llama.cpp): pp 0.70x | tg 0.86x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 629.04 | tg 72.64 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.70x | tg 0.86x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

#### Session B — t=1 prefill profile (gate-prefill-attr.out)

```
# gate-prefill-attr (M4b.13: split-bracket t=1 prefill profile) — 2026-07-17T09:28:14Z
machine: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (GenuineIntel) | 16 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

--- t=1 prefill profile (split brackets) ---
profile [prefill] 28.440s wall, 90777941966 cyc total
  op                                   cycles   share        GB/s
  attention                       33091451358   36.5%           -
  matmul:lm_head.weight           14712405365   16.2%        68.1
  matmul:layers.*.ffn.up_proj.weight    11212453126   12.4%        68.7
  matmul:layers.*.ffn.gate_proj.weight    11189064422   12.3%        68.8
  matmul:layers.*.ffn.down_proj.weight    11106343462   12.2%        69.4
  swiglu                           2810755635    3.1%           -
  matmul:layers.*.attn.q_proj.weight     2069805400    2.3%        68.6
  matmul:layers.*.attn.o_proj.weight     2063833665    2.3%        68.8
  rmsnorm                           712850186    0.8%           -
  rope                              443101159    0.5%           -
  quantize                          388824608    0.4%           -
  matmul:layers.*.attn.v_proj.weight      296418339    0.3%        68.4
  matmul:layers.*.attn.k_proj.weight      295105993    0.3%        68.7
  add                               212686943    0.2%           -
  kv_append                          84748951    0.1%           -
  bias                               80347096    0.1%           -
  embed                               7746258    0.0%           -
profile [decode] 1.225s wall, 3889342480 cyc total
  op                                   cycles   share        GB/s
  attention                        1243763524   32.0%           -
  matmul:lm_head.weight             704675845   18.1%        22.1
  matmul:layers.*.ffn.down_proj.weight      544723368   14.0%        21.9
  matmul:layers.*.ffn.gate_proj.weight      543648899   14.0%        22.0
  matmul:layers.*.ffn.up_proj.weight      543064689   14.0%        22.0
  matmul:layers.*.attn.q_proj.weight      100963222    2.6%        21.8
  matmul:layers.*.attn.o_proj.weight      100465006    2.6%        21.9
  swiglu                             47691940    1.2%           -
  matmul:layers.*.attn.k_proj.weight       15256692    0.4%        20.6
  matmul:layers.*.attn.v_proj.weight       15109387    0.4%        20.8
  rope                               13485582    0.3%           -
  rmsnorm                            11227775    0.3%           -
  add                                 3035064    0.1%           -
  bias                                1962121    0.1%           -
  quantize                             178605    0.0%           -
  embed                                 90761    0.0%           -
  kv_append                                 0    0.0%           -
```

### 2026-07-17 — pre-registered ladder verdict (arithmetic shown)

`matmul_share` = Σ `matmul:*` prefill cycles ÷ prefill total, from the
fresh split-bracket t=1 profiles above. pp ratio = inferno vs llama
best-of-builds, from the same sessions' gate-bench-protocol runs.

Session A (d2.c1.medium, 6336Y 16c):
- Σ matmul:* = 19,953,000,980 + 15,310,093,992 + 15,302,367,844
  + 15,287,421,952 + 2,819,448,996 + 2,818,763,890 + 404,686,292
  + 404,288,584 = 72,300,072,530 cyc
- prefill total = 123,445,100,224 cyc → matmul_share = 58.57%
- pp ratio = 0.74x
- ceiling check: 0.74 / (1 − 0.5857 × 0.5) = 0.74 / 0.7071 = **1.046 ≥ 1.0** ✓

Session B (s2.c2.medium, E-2388G 8c):
- Σ matmul:* = 14,712,405,365 + 11,212,453,126 + 11,189,064,422
  + 11,106,343,462 + 2,069,805,400 + 2,063,833,665 + 296,418,339
  + 295,105,993 = 52,945,429,772 cyc
- prefill total = 90,777,941,966 cyc → matmul_share = 58.32%
- pp ratio = 0.70x
- ceiling check: 0.70 / (1 − 0.5832 × 0.5) = 0.70 / 0.7084 = **0.988 < 1.0** ✗

Walking the pre-registered rule (½ factor fixed by this spec, not
adjustable at gate time):

1. pp ≥ 1.0x on both boxes? **No** (0.74x / 0.70x) → rule 1 does not apply.
2. `pp_ratio / (1 − matmul_share × 0.5) ≥ 1.0` on every box still under
   1.0x? **No** — Session A passes (1.046) but Session B fails (0.988).
3. → **STOP-out.** Even a perfect VNNI outcome at the spec's ½-ceiling
   cannot lift the 8-core box to pp 1.0x. **Lever 2 (Tasks 6–7) SKIPPED.**

**Finding (where the residual lives, from the fresh profiles):** the t=1
prefill residual is attention-shaped, not matmul-shaped. Attention is the
single largest prefill bracket on both boxes (35.5% / 36.5%), larger than
any matmul row; the matmul rows run at memory-stream rates (47.6 GB/s on
A, ~68 GB/s on B) with lm_head the biggest single matmul (16.2% both).
Lever 1 (register tiles) shipped and is in these numbers; the remaining
prefill gap is dominated by the attention kernel plus the ~41% non-matmul
tail, which no GEMM-side lever can close. This finding scopes a future
attention-side prefill milestone; per the ladder discipline, closing
diagnostic follows in the exit-criteria walk.
