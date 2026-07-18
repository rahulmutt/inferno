# M4a — Bench Protocol + Sampling Suite Design

**Date:** 2026-07-06
**Status:** Approved design, pre-implementation
**Milestone:** M4a (see [inferno v1 design](2026-07-04-inferno-v1-design.md) §Milestones)

The v1 design's M4 ("Runtime polish + bench") is decomposed into two
milestones. **M4a builds the measurement and the product surface**: the full
sampling suite and a real `inferno bench` that produces an apples-to-apples
inferno-vs-llama.cpp comparison under a documented protocol. **M4b is the
perf work** (multi-threaded generated code, GEMM prefill tiles, AVX-512
kernels, F16 KV) — planned separately, driven by what M4a's bench shows.
You can't tune what you can't measure.

## Scope Decisions (M4a)

| Decision | Choice |
|---|---|
| Milestone split | M4a = sampling suite + `inferno bench` + protocol; M4b = perf work until the v1 win criterion ("beat llama.cpp prefill **and** decode tok/s") is met |
| llama.cpp comparison | **Manual protocol + recorded data points.** `inferno bench` runs locally on quiet hardware and emits a comparison report; numbers are recorded in this spec's Amendments section, like the M3 speedup data point. No CI perf gate against llama.cpp (CI runners are noise — see AGENTS.md) |
| llama.cpp measurement source | **Drive `llama-bench -o json`** (the devenv-pinned binary) with matched parameters and parse its JSON — apples-to-apples by construction, version pinned by devenv's nixpkgs lock |
| Sampler RNG | Hand-rolled **xoshiro256\*\*** (~20 lines, `u64` seed). Zero new dependencies; output can never shift under a crate bump, which the exact-pick unit tests depend on |
| Sampler chain order | llama.cpp's conventional order for comparability: repeat penalty → top-k → top-p → min-p → temperature → seeded draw |
| Default sampling | `temperature = 0` short-circuits to argmax. `inferno run` defaults stay bit-identical to today's greedy behavior; differential gates are untouched |
| Streaming CLI | Already done (M1's `Utf8Buffer` + callback streaming in `run`); M4a adds only a post-generation stats line |

**Explicitly out of scope for M4a** (deferred to M4b unless noted):
multi-threaded generated code, GEMM/prefill tiles, AVX-512 kernels, F16 KV,
aggressive/cost-model fusion, in-memory LLJIT, any CI perf gate vs llama.cpp,
exotic samplers beyond the v1 list (mirostat, typical-p, XTC — not v1 at all).

## What M4a Adds

No new crates. Two work items in existing crates:

- **`inferno-runtime`** — `ChainSampler` behind the existing `Sampler` trait;
  one trait extension (`accept`).
- **`cli`** — `inferno bench` subcommand + sampling flags on `inferno run`.
  `bench-compiled` (the M3 nightly interpreter-speedup gate) is unchanged.

## Sampling suite (`inferno-runtime`)

### Trait extension

`Sampler` gains one default-no-op method:

```rust
pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> u32;
    fn accept(&mut self, _token: u32) {}   // new; default no-op
}
```

The generator calls `accept` for every prompt token during prefill and for
every sampled token — that is how repeat penalty sees context without
changing `sample()`'s signature. `Greedy` is untouched.

### `ChainSampler`

Configured by `SamplerConfig { temperature, top_k, top_p, min_p,
repeat_penalty, repeat_last_n, seed }`. Stages apply in order:

1. **Repeat penalty** over the last `repeat_last_n` accepted tokens (ring
   buffer): llama.cpp's sign convention — positive logits divided by the
   penalty, negative logits multiplied. `penalty = 1.0` is a no-op.
2. **Top-k**: keep the k highest logits (`k = 0` disables).
3. **Top-p**: softmax, keep the smallest prefix of descending-probability
   tokens whose cumulative mass ≥ p (`p = 1.0` disables).
4. **Min-p**: drop tokens whose probability < `min_p × max_prob`
   (`min_p = 0` disables).
5. **Temperature**: divide surviving logits by `temperature`;
   `temperature = 0` short-circuits the whole chain to argmax (greedy
   semantics, lowest-index tie-break, matching `Greedy`).
6. **Draw** from the renormalized distribution using the internal
   xoshiro256\*\* RNG seeded from `seed`.

Neutral defaults (`temperature = 0`, `top_k = 0`, `top_p = 1.0`,
`min_p = 0.0`, `repeat_penalty = 1.0`, `repeat_last_n = 64`, `seed = 0`)
make `ChainSampler::default()` behave exactly like `Greedy`.

### RNG

xoshiro256\*\* implemented in-repo (private module in `inferno-runtime`),
seeded from the `u64` seed via splitmix64 (the reference seeding procedure).
Rationale: `rand`'s `SmallRng` documents that its algorithm may change
between versions, which would silently break the "seeded RNG, exact expected
picks" tests the v1 testing strategy requires; a pinned reference algorithm
cannot.

### CLI flags (`inferno run`)

`--temperature` (default 0), `--top-k`, `--top-p`, `--min-p`,
`--repeat-penalty`, `--repeat-last-n`, `--seed`. Validation rejects
out-of-range values (`temperature ≥ 0`, `top_p ∈ (0, 1]`, `min_p ∈ [0, 1)`,
`repeat_penalty > 0`). After generation, `run` prints a one-line stats
summary to **stderr** (prompt tokens, generated tokens, prefill tok/s,
decode tok/s from `GenStats`) so piped stdout stays clean token text.

## `inferno bench` (CLI)

### Inferno-side measurement

- **Prefill (pp):** a synthetic prompt of exactly `--pp` token ids
  (default 512; valid vocab ids, content irrelevant to speed — mirrors
  llama-bench), fed directly to the compiled backend's prefill, bypassing
  the tokenizer. Reported as tok/s from `GenStats::prefill_secs`.
- **Decode (tg):** `--tg` greedy decode steps (default 128) from that state,
  tok/s from `GenStats::decode_secs`.
- `--reps` repeats (default 5, matching llama-bench's default), reported as
  mean ± stddev. First compile happens before timing begins (warm artifact
  cache), so LLVM cost never lands in a measured run.

The synthetic-prompt path needs one small runtime addition: a way to prefill
from raw token ids instead of a string prompt (used only by bench).

### llama.cpp-side measurement

Invoke `llama-bench` (found on `PATH` — the devenv shell provides the pinned
build; overridable via `--llama-bench <path>`) with the **same model file**,
`-p <pp> -n <tg>`, `-o json`, and `-t <threads>` where `--threads` defaults
to physical cores from `TargetDesc`. Parse the JSON for pp/tg avg ± stddev
and build info.

While inferno's generated code is single-threaded (until M4b), the protocol
also records a `-t 1` llama-bench row as a per-thread-parity diagnostic —
but the honest headline number is the full-thread one; the v1 win criterion
is measured against llama.cpp at its best.

### Report

Human-readable table by default; `--json` emits the machine-readable data
point. Both include the fingerprint: CPU model + physical/logical cores
(from `TargetDesc` detection), inferno version + git hash, llama.cpp build
info (from llama-bench's JSON), model file name, quant, pp/tg/reps/threads.

### Protocol (recorded here, executed manually)

1. Quiet machine, devenv shell, release build (`mise run bench` wraps
   `cargo run --release -p inferno -- bench …`).
2. Pinned nightly model (Qwen2.5-0.5B-Instruct Q8_0 GGUF) unless the data
   point says otherwise.
3. Defaults: pp=512, tg=128, reps=5; llama-bench at `-t <physical cores>`
   plus the `-t 1` diagnostic row.
4. Record the full report (both engines, fingerprint) in this spec's
   Amendments section. Never edit a recorded data point.

### Error handling

Actionable failures, never a half-report: missing `llama-bench` binary
("run inside `devenv shell`…"), non-zero llama-bench exit (stderr passed
through), unparseable JSON (schema drift → error citing the pinned version).
CLI flag validation as above.

## Testing Strategy

- **Sampler units (blocking tier):** seeded exact-pick tests (fixed logits +
  seed → exact token); per-stage boundary tests (top-k=1 ≡ greedy; top-p
  cumulative-mass edge; min-p relative cutoff; repeat-penalty sign
  convention; temperature=0 ≡ `Greedy` including tie-break); neutral-config
  ≡ `Greedy` over random logit vectors; xoshiro256\*\* + splitmix64 against
  the published reference vectors.
- **Determinism (blocking tier):** same seed → same token sequence on the
  tiny fixture model via the interpreter backend; different seeds diverge at
  temperature > 0.
- **llama-bench JSON parsing (blocking tier):** golden fixture captured from
  the pinned llama-bench — parsing tested with no llama.cpp at test time.
- **Bench E2E:** manual per the protocol (needs a real model + quiet
  hardware). Not in CI; the existing nightly `speedup` gate already covers
  "compiled path still fast" regression detection.
- **Flag validation units** for `run` and `bench`.

## Implementation Phases

1. **Sampling** — RNG module + `ChainSampler` + trait `accept` + generator
   wiring + `run` flags + stats line. Blocking-tier tests throughout.
2. **Bench** — raw-token prefill path, inferno-side measurement, llama-bench
   invocation + JSON parsing, report + `--json`, `mise run bench` task.
3. **Protocol + first data point** — run the protocol on the dev machine,
   record the first inferno-vs-llama.cpp numbers in Amendments; that data
   point seeds M4b's plan.

## Risks

- **The first data point will likely show inferno losing** (single-threaded
  generated code vs llama.cpp at full threads). That is the point: M4a's
  deliverable is the true gap, quantified, so M4b attacks the right
  bottlenecks. The `-t 1` diagnostic row tells us whether per-thread perf is
  already competitive.
- **llama-bench JSON schema drift.** Mitigated by the devenv pin (schema
  can't change under us) and the golden-fixture parse test; a future
  llama.cpp bump that changes the schema fails loudly in the blocking tier.
- **Sampling correctness is subtle** (ordering, ties, renormalization).
  Mitigated by exact-pick seeded tests and the greedy-equivalence property;
  the differential gates stay pinned to greedy and are unaffected by
  construction.

## Amendments

*(Recorded data points and post-approval changes land here.)*

- **First inferno-vs-llama.cpp data point (Task 9, dev machine, 2026-07-06):**
  Ran the protocol inside `devenv shell` (release build via `mise run
  bench`) on the quiet dev machine — **AMD Ryzen 9 3900 12-Core Processor,
  12 physical / 24 logical cores** — against the pinned nightly model
  `qwen2.5-0.5b-instruct-q8_0.gguf` (qwen2 1B Q8_0,
  `/home/dev/.cache/inferno-tests/qwen2.5-0.5b-instruct-q8_0.gguf`),
  defaults pp=512, tg=128, reps=5. inferno commit `af53098`, llama.cpp build
  `6f4f53f` (devenv-pinned, BLAS + CPU-haswell backends).

  Real llama-bench JSON flowed through the Task 6 parser **unchanged** — no
  fixture correction was needed; both the table run and the `--json` run
  parsed and reported cleanly on the first try.

  Table run (`mise run bench -- <MODEL>`):

  ```
  model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
  cpu: BLAS, AMD Ryzen 9 3900 12-Core Processor (12 physical / 24 logical cores)
  inferno 0.1.0 (af53098) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

  engine                 threads        pp512 tok/s        tg128 tok/s
  inferno (compiled)           1       16.18 ± 0.10        10.51 ± 0.07 
  llama.cpp                   12       45.56 ± 0.74        26.78 ± 0.66 
  llama.cpp (t=1 diag)         1       53.71 ± 1.04        23.76 ± 0.16 

  ratio (inferno/llama.cpp): pp 0.36x | tg 0.39x
  ```

  `--json` run (independent invocation immediately after, same protocol):

  ```json
  {
    "model": "qwen2.5-0.5b-instruct-q8_0.gguf",
    "model_type": "qwen2 1B Q8_0",
    "cpu_info": "BLAS, AMD Ryzen 9 3900 12-Core Processor",
    "physical_cores": 12,
    "logical_cores": 24,
    "inferno_version": "0.1.0",
    "inferno_git": "af53098",
    "llama_build_commit": "6f4f53f",
    "pp": 512,
    "tg": 128,
    "reps": 5,
    "inferno_threads": 1,
    "llama_threads": 12,
    "inferno_pp_tok_s": 16.213095083530817,
    "inferno_pp_stddev": 0.0872368152354205,
    "inferno_tg_tok_s": 10.55102694029267,
    "inferno_tg_stddev": 0.027926234244083543,
    "llama_pp_tok_s": 43.374793,
    "llama_pp_stddev": 1.357777,
    "llama_tg_tok_s": 25.028042,
    "llama_tg_stddev": 0.61535,
    "llama_t1_pp_tok_s": 31.301321,
    "llama_t1_pp_stddev": 12.139925,
    "llama_t1_tg_tok_s": 15.160113,
    "llama_t1_tg_stddev": 0.219028
  }
  ```

  **inferno loses both pp and tg against llama.cpp at full threads** — the
  expected M4a outcome (single-threaded generated code vs llama.cpp at 12
  physical cores): table run ratio **pp 0.36x, tg 0.39x**; the `--json` run
  agrees (pp 16.21/43.37 ≈ 0.37x, tg 10.55/25.03 ≈ 0.42x). The `-t 1`
  diagnostic row is noisier between the two runs (pp 53.71±1.04 vs
  31.30±12.14, tg 23.76±0.16 vs 15.16±0.22) — the second run's `t=1` pp
  stddev (12.14 tok/s on a 31.3 tok/s mean) suggests some scheduling/thermal
  contention during just that measurement, most likely from the `-t 1`
  single-core run being more sensitive to a noisy neighbor than the
  full-thread rows (whose numbers stayed consistent across both
  invocations). inferno's own pp/tg numbers were stable and consistent
  between runs (16.18 vs 16.21 tok/s pp; 10.51 vs 10.55 tok/s tg), so the
  headline pp/tg ratios above are trustworthy; treat the `t=1` diagnostic
  as directional rather than precise. This data point seeds M4b's plan:
  the gap is real and, per the design's risk note, expected — multi-threaded
  generated code is M4b's first lever.

  **Per-thread read (follow-up note, same data):**

  The headline ratios (pp 0.36x, tg 0.39x) compare inferno at 1 thread to
  llama.cpp at 12 threads. The `-t 1` diagnostic row shows single-thread
  parity:

  - **Prefill (pp):** inferno ~16.2 tok/s vs llama.cpp t=1 pp 31.30–53.71 tok/s
    across the two runs → inferno at roughly 0.30–0.52x of llama.cpp's
    single-thread prefill rate.
  - **Decode (tg):** inferno ~10.5 tok/s vs llama.cpp t=1 tg 15.16–23.76 tok/s
    → roughly 0.44–0.70x single-thread decode.
  - **Conclusion:** Even per-thread, inferno is behind — the full-thread gap is
    NOT explained by thread count alone; part of it is per-thread kernel/codegen
    quality. Both per-thread parity and M4b threading are needed to close the
    headline 0.36x/0.39x gap. (The t=1 range is wide because of the scheduling
    noise already noted above.)

### 2026-07-11 — quiet-hw verdict (M4b.7 gate-bench-protocol, bare metal): v1 win criterion NOT MET

First run of the protocol on genuinely quiet hardware — bare metal via
`mise run metal` (d2.c1.medium, Xeon Gold 6336Y, 16 physical / 32
logical, PREFLIGHT FIT), inferno 0.1.0 @ 6b0df49 vs llama.cpp 6f4f53f,
pp=512 tg=128 reps=5. **pp 0.84x | tg 0.87x (independent --json run: pp
0.61x | tg 0.86x) → v1 win criterion (pp > 1x AND tg > 1x) NOT MET.**

Caveat recorded with the number: the llama.cpp side reports a **BLAS cpu
backend**, and its t=1 diagnostic (pp 403.53) *exceeds* its t=16 run
(pp 310.30) — BLAS almost certainly multithreads internally regardless
of -t, so the pp ratio compares inferno@16 threads against an
effectively-all-cores llama, and the t=1 "per-thread parity" diagnostic
is meaningless for pp. tg is barely affected (22.03 @ t=1 vs 56.39 @
t=16 scales plausibly). Follow-up: pin the llama-bench build to the
pure-CPU backend (no BLAS) so pp ratios compare like-for-like; the tg
0.87x reading stands meanwhile — decode is within 13% of llama.cpp on
quiet Intel, per-thread tg 0.73x (16.14 vs 22.03).

*Follow-up done same day:* devenv now pins the pure-CPU ggml build as
the PATH comparator (`pkgs.llama-cpp.override { blasSupport = false; }`
— verified: no `libggml-blas.so` in the package, `backends: CPU` at
runtime), and `gate-bench-protocol.sh` additionally records the stock
BLAS build as a "llama at its best" reference row (via
`INFERNO_LLAMA_BENCH_BLAS`, kept off PATH), judging the v1 criterion
against the per-metric max of the two builds. The pp numbers above
predate this and keep their confound caveat; the next quiet-hw pass
supersedes them.

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-11T13:10:00Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-11

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: BLAS, Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (16 physical / 32 logical cores)
inferno 0.1.0 (6b0df49) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      260.72 ± 8.24        49.17 ± 0.11 
inferno (t=1 diag)           1       63.24 ± 0.02        16.14 ± 0.01 
llama.cpp                   16      310.30 ± 22.05       56.39 ± 0.07 
llama.cpp (t=1 diag)         1      403.53 ± 2.22        22.03 ± 2.36 

ratio (inferno/llama.cpp): pp 0.84x | tg 0.87x

ratios (inferno/llama.cpp, from the independent --json run): pp 0.61x | tg 0.86x
gate: v1 win criterion (pp > 1x AND tg > 1x) -> NOT MET
```

### 2026-07-11 — second quiet-hw session, fixed comparator: first valid ratios — pp 0.23x | tg 0.79x, NOT MET (supersedes the morning's 0.84x/0.61x)

Same box type, inferno @ 1804d9f, protocol now judged against llama at
its genuine best (pure-CPU ggml build on PATH honoring the -t pin, plus
a BLAS-build reference row, criterion vs per-metric max — see the
follow-up note above). **pp 0.23x | tg 0.79x vs best-of → v1 win
criterion NOT MET.** The morning's pp numbers are formally superseded:
the BLAS-confounded llama measured ~310 pp; the pure-CPU build does
**1187.7 ± 242.2** on the same silicon (~4x stronger opponent), and the
BLAS reference row (685.0 / 61.9) confirms BLAS is not llama's best
here — the fix strengthened the opponent, exactly as the confound
predicted. tg is stable across builds (61.3 vs 61.9) and inferno's tg
0.79–0.80x is consistent with every prior reading. Caveats carried: the
llama pp stddev is large (±242, ~20%) and its thread-scaling mildly
superlinear — worth re-measuring on a second machine class before
treating 0.23x as precise; the direction (deep pp deficit, threading
AND per-thread — see the M4b.1 same-day fork) is not in doubt.

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-11T21:05:24Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-11

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (16 physical / 32 logical cores)
inferno 0.1.0 (1804d9f) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      248.18 ± 14.74       49.27 ± 0.06 
inferno (t=1 diag)           1       63.22 ± 0.01        16.16 ± 0.01 
llama.cpp                   16     1187.74 ± 242.19       61.25 ± 0.07 
llama.cpp (t=1 diag)         1      118.03 ± 0.18        22.86 ± 0.00 

ratio (inferno/llama.cpp): pp 0.21x | tg 0.80x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 684.97 | tg 61.86 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.23x | tg 0.79x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

### 2026-07-12 — third quiet-hw session, post-M4b.8: pp 0.32x | tg 0.79x — NOT MET, pp materially improved

Same box type and protocol (pure-CPU comparator + BLAS reference row,
judged vs per-metric best-of), inferno @ 823437f — first reading with
M4b.8's parallel prefill attention. **pp 0.32x | tg 0.79x vs llama
best-of → v1 win criterion NOT MET.** pp moved 0.23x → 0.32x (+39%,
entirely the M4b.8 prefill gain — inferno pp512 285→398 tok/s at t=16);
tg 0.79x unchanged, consistent with every prior reading. Caveat carried
from the second session: llama's pp stddev remains large (±272 here,
~24%); the pure-CPU build (1143.5) again beats the BLAS reference
(698.2), confirming the comparator choice.

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-12T08:50:44Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-12

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (16 physical / 32 logical cores)
inferno 0.1.0 (823437f) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      398.01 ± 3.88        48.87 ± 0.04 
inferno (t=1 diag)           1       63.44 ± 0.02        17.31 ± 0.01 
llama.cpp                   16     1143.51 ± 272.44       61.35 ± 0.26 
llama.cpp (t=1 diag)         1      118.15 ± 0.04        17.96 ± 0.01 

ratio (inferno/llama.cpp): pp 0.35x | tg 0.80x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 698.20 | tg 61.53 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.32x | tg 0.79x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

### 2026-07-12 — fourth quiet-hw session, post-M4b.9: pp 0.69x | tg 0.84x — NOT MET; pp doubled again

Same box type and protocol (pure-CPU comparator + BLAS reference row, judged
vs per-metric best-of), inferno @ 2387266 — first reading with M4b.9's
parallel serial tail. **pp 0.69x | tg 0.84x vs llama best-of → v1 win
criterion NOT MET.** pp moved 0.32x → 0.69x (+116%; inferno pp512 398 → 738
tok/s at t=16 — the M4b.9 prefill gain, consistent with the +88% seen on the
scaling gate's t=12 row). The pp gap to llama is now 1.46x, down from 3.1x
one milestone ago.

Two honest caveats on this row:

- **The tg ratio improvement (0.79x → 0.84x) is a comparator artifact, not an
  inferno gain.** Decode is untouched by M4b.9 by construction and inferno's
  tg128 is flat (48.87 → 48.18); the ratio moved because llama's tg fell
  61.35 → 58.09 this session. Read tg as unchanged.
- **Inferno's pp variance grew sharply: ±122.86 (~17%), up from ±3.88 (~1%)
  in the M4b.8 session.** M4b.9 replaced tight serial per-token loops with
  ~8–14 pool dispatches per layer per tile, so run-to-run scheduler noise now
  has a surface it did not have before. It does not threaten a gate (the
  scaling gate's medians are clean and bit-identity is CI-enforced), but the
  headline pp number is now a noisier estimator than it was, and a future
  reading should not treat a ±15% swing as signal. llama's pp stddev remains
  large as always (±253.62, ~24%); the pure-CPU build (1076.0) again beats
  the BLAS reference (675.4), confirming the comparator choice.

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-12T15:20:39Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-12

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (16 physical / 32 logical cores)
inferno 0.1.0 (2387266) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      738.47 ± 122.86       48.18 ± 0.50
inferno (t=1 diag)           1       63.57 ± 0.05        22.90 ± 0.00
llama.cpp                   16     1075.97 ± 253.62       58.09 ± 0.18
llama.cpp (t=1 diag)         1      117.69 ± 0.14        16.32 ± 0.08

ratio (inferno/llama.cpp): pp 0.69x | tg 0.83x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 675.38 | tg 58.29 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.69x | tg 0.84x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

### 2026-07-15 — fifth & sixth quiet-hw sessions (M4b.10 by-product): v1 NOT MET on two machines — and the last capped-baseline tg numbers

Two more machines (bench-protocol is a by-product of the M4b.10 `verify.sh`
runs). v1 win criterion NOT MET on both, consistent with the four prior
sessions.

- **6336Y, 16c:** pp 0.67x | tg 0.85x vs llama best-of → **NOT MET.**
- **E-2388G, 8c:** pp 0.60x | tg 0.74x vs llama best-of → **NOT MET.**

**Forward note — these are the last decode numbers on the capped baseline.**
Both runs predate M4b.10's cap removal (inferno @ 72677d7 and 25bade4, before
`4425e1f`), so tg was still throttled by `clamp(active/3)`. M4b.10 measured that
default leaving 7–12% of decode on the table on these smaller boxes; expect tg
to rise by roughly that much on the next quiet-hw bench pass, now that decode
uses full active threads. pp is unaffected (prefill was never capped).

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-14T13:06:00Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-14

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (16 physical / 32 logical cores)
inferno 0.1.0 (72677d7) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      797.77 ± 80.89       49.31 ± 0.06 
inferno (t=1 diag)           1       63.34 ± 0.03        16.17 ± 0.09 
llama.cpp                   16     1137.27 ± 259.83       57.71 ± 0.29 
llama.cpp (t=1 diag)         1      118.42 ± 0.10        23.06 ± 0.04 

ratio (inferno/llama.cpp): pp 0.70x | tg 0.85x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 683.39 | tg 57.33 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.67x | tg 0.85x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET

# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-14T18:36:24Z
machine: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (GenuineIntel) | 16 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-14

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (8 physical / 16 logical cores)
inferno 0.1.0 (25bade4) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)           8      637.44 ± 0.85        56.94 ± 0.01 
inferno (t=1 diag)           1       90.43 ± 0.28        36.24 ± 0.26 
llama.cpp                    8     1061.47 ± 1.87        77.35 ± 0.10 
llama.cpp (t=1 diag)         1      165.21 ± 0.08        35.20 ± 0.06 

ratio (inferno/llama.cpp): pp 0.60x | tg 0.74x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 654.35 | tg 77.29 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.60x | tg 0.74x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

### 2026-07-16 — seventh quiet-hw session (M4b.11 session A): the deferred uncapped tg re-bench (d2.c1.medium, 16c)

This is the uncapped tg re-bench the 2026-07-15 forward note deferred: first
bench-protocol run on the 6336Y with the decode-thread cap removed (M4b.10,
`4425e1f`; inferno @ `1f579ce`). tg rose 49.31 → 53.48 tok/s (+8.5%), inside
the forecast 7–12% band; tg ratio vs llama best-of moved 0.85x → 0.89x. pp
unchanged within noise, as predicted. v1 win criterion still NOT MET.

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-16T15:23:25Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-16

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (16 physical / 32 logical cores)
inferno 0.1.0 (1f579ce) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      849.09 ± 0.44        53.48 ± 0.09 
inferno (t=1 diag)           1       63.39 ± 0.05        22.39 ± 0.02 
llama.cpp                   16     1180.92 ± 290.11       60.45 ± 0.19 
llama.cpp (t=1 diag)         1      117.54 ± 0.26        16.39 ± 0.02 

ratio (inferno/llama.cpp): pp 0.72x | tg 0.88x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 628.79 | tg 61.51 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.73x | tg 0.89x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

### 2026-07-16 — eighth quiet-hw session (M4b.11 session B): the deferred uncapped tg re-bench (s2.c2.medium, 8c)

Second half of the uncapped re-bench the 2026-07-15 forward note deferred
(E-2388G; cap removed in M4b.10 `4425e1f`; inferno @ `d5175c7`). Read with
care: this is a **different physical instance** (CHI) than the 2026-07-15
session's box (PHX), and the whole box runs slower — llama.cpp tg dropped
77.35 → 72.82 (−5.9%) on identical engine code. Raw inferno tg is flat
across the two boxes (56.94 → 56.74), so the forecast +7% from cap removal
is not directly observable cross-instance; the within-run tg ratio vs llama
best-of, which cancels box speed, moved 0.74x → 0.79x, consistent with the
forecast's direction and size. v1 win criterion still NOT MET.

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-16T16:39:24Z
machine: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (GenuineIntel) | 16 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-16

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (8 physical / 16 logical cores)
inferno 0.1.0 (d5175c7) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)           8      637.75 ± 3.33        56.74 ± 0.08 
inferno (t=1 diag)           1       90.44 ± 0.24        34.85 ± 0.16 
llama.cpp                    8     1042.23 ± 22.30       72.82 ± 0.02 
llama.cpp (t=1 diag)         1      163.93 ± 0.16        33.15 ± 0.14 

ratio (inferno/llama.cpp): pp 0.61x | tg 0.78x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 657.33 | tg 72.67 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.61x | tg 0.79x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

### 2026-07-16 — ninth & tenth quiet-hw sessions: M4b.11 closing data point (both machines)

The M4b.11 closing protocol run at the milestone's final commit (`332130b`,
Lever 1 = head-sharded decode attention landed; preflight FIT on both, CHI
instances). Judged in the M4b.11 spec against its headroom-set targets; the
v1 criterion lines below are context, never the gate (M4b.11 discipline).
Versus the same-day Task 2 baselines, the within-run tg ratio vs llama
best-of moved 0.89x → 0.96x (16c) and 0.79x → 0.86x (8c).

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-16T19:14:49Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-16

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (16 physical / 32 logical cores)
inferno 0.1.0 (332130b) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      793.71 ± 76.42       57.20 ± 0.55 
inferno (t=1 diag)           1       61.59 ± 0.12        22.22 ± 0.00 
llama.cpp                   16     1290.07 ± 241.84       61.00 ± 0.31 
llama.cpp (t=1 diag)         1      117.72 ± 0.06        16.64 ± 0.03 

ratio (inferno/llama.cpp): pp 0.62x | tg 0.94x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 570.39 | tg 60.88 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.74x | tg 0.96x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET

# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-16T19:42:57Z
machine: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (GenuineIntel) | 16 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-16

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (8 physical / 16 logical cores)
inferno 0.1.0 (332130b) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)           8      592.11 ± 55.55       62.49 ± 0.30 
inferno (t=1 diag)           1       88.98 ± 0.05        34.21 ± 0.29 
llama.cpp                    8     1053.28 ± 1.69        72.69 ± 0.12 
llama.cpp (t=1 diag)         1      165.10 ± 0.19        33.42 ± 0.01 

ratio (inferno/llama.cpp): pp 0.56x | tg 0.86x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 648.69 | tg 72.68 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.60x | tg 0.86x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

### 2026-07-17 — M4b.12 closing benches (both machines, no lever landed)

M4b.12 closed all-STOP (no lever landed), so the closing tg context comes
from the attribution sessions' own protocol runs — the code benched IS the
closing state (main `783b453`; session B ran `c2b8e04`, docs-only over it).
Recorded here verbatim per the standing protocol; verdict cross-referenced
from the M4b.12 spec §Amendments closing verdict.

#### d2.c1.medium (6336Y 16c, PHX) — gate-bench-protocol.out (verbatim)

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-17T01:33:20Z
machine: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (GenuineIntel) | 32 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz (16 physical / 32 logical cores)
inferno 0.1.0 (783b453) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      836.84 ± 19.71       58.78 ± 0.27 
inferno (t=1 diag)           1       63.35 ± 0.02        16.17 ± 0.01 
llama.cpp                   16     1232.83 ± 220.71       62.04 ± 0.04 
llama.cpp (t=1 diag)         1      117.77 ± 0.10        16.44 ± 0.01 

ratio (inferno/llama.cpp): pp 0.68x | tg 0.95x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 532.35 | tg 63.00 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.75x | tg 0.91x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

#### s2.c2.medium (E-2388G 8c, CHI) — gate-bench-protocol.out (verbatim)

```
# gate-bench-protocol (M4a protocol / v1 win criterion) — 2026-07-17T03:35:26Z
machine: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (GenuineIntel) | 16 logical CPUs | kernel 6.9.10+bpo-amd64 | 2026-07-17

model: qwen2.5-0.5b-instruct-q8_0.gguf (qwen2 1B Q8_0)
cpu: Intel(R) Xeon(R) E-2388G CPU @ 3.20GHz (8 physical / 16 logical cores)
inferno 0.1.0 (c2b8e04) vs llama.cpp 6f4f53f | pp=512 tg=128 reps=5

engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)           8      622.35 ± 11.92       62.66 ± 0.03 
inferno (t=1 diag)           1       91.09 ± 0.09        35.27 ± 0.12 
llama.cpp                    8     1041.01 ± 2.08        72.68 ± 0.05 
llama.cpp (t=1 diag)         1      164.89 ± 0.61        33.29 ± 0.05 

ratio (inferno/llama.cpp): pp 0.60x | tg 0.86x

llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 638.64 | tg 72.66 tok/s

ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.60x | tg 0.86x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

### 2026-07-17 — M4b.13 closing benches (both machines, Lever 1 landed, Lever 2 gated out)

M4b.13 mid-milestone gate sessions double as the closing data: the ladder
verdict was STOP-out at rule 3 (see the M4b.13 spec Amendments for the
arithmetic), so no lever landed after these runs — the benched code IS
the closing state (main `3183e29`, register-tiled Q8_0 GEMM shipped in
PR #30). Verbatim `gate-bench-protocol.out` from each box:

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

### 2026-07-17 — M4b.14 closing bench (query-blocked prefill attention, both quiet-hw boxes)

M4b.14 closing data point (Lever 1 = query-blocked prefill attention kernel,
branch `m4b14-qblock` @ `83c183f`; Lever 2 gated out all-STOP — see the
M4b.14 spec §Amendments for the sessions and ladder arithmetic). The
mid-milestone sessions double as the closing protocol runs (M4b.13
precedent — same commit, same day, no new provision). v1 win criterion
still NOT MET on either box; pp best-of 0.79x (16c, was 0.74x) / 0.70x
(8c, unchanged).

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

### 2026-07-18 — M4b.15 session A protocol data points (d2.c1.medium, 6336Y, quiet-hw FIT)

Baseline binary 9883086 (2026-07-18T00:34:27Z):

```
engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      856.95 ± 85.20       57.26 ± 0.64
inferno (t=1 diag)           1       75.58 ± 0.09        22.62 ± 0.01
llama.cpp                   16     1206.84 ± 243.87      58.33 ± 0.27
llama.cpp (t=1 diag)         1      118.02 ± 0.33        22.88 ± 0.00
ratio (inferno/llama.cpp): pp 0.71x | tg 0.98x
llama.cpp BLAS-build reference: pp 528.13 | tg 63.12 tok/s
ratios (inferno vs llama best-of-builds): pp 0.79x | tg 0.93x
```

M4b.15 branch binary 066bd45, same kernel — no lever shipped
(2026-07-18T00:52:15Z):

```
engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      924.67 ± 17.31       59.12 ± 0.53
inferno (t=1 diag)           1       69.42 ± 0.01        15.78 ± 0.02
llama.cpp                   16     1252.59 ± 279.69      63.69 ± 0.14
llama.cpp (t=1 diag)         1      117.71 ± 0.07        16.45 ± 0.03
ratio (inferno/llama.cpp): pp 0.74x | tg 0.93x
llama.cpp BLAS-build reference: pp 414.62 | tg 62.60 tok/s
ratios (inferno vs llama best-of-builds): pp 0.78x | tg 0.91x
```

Caveat (recorded, not edited): between the two runs the t=1 diag rows
dropped for BOTH engines symmetrically (inferno 22.62→15.78, llama
22.88→16.45 tg) — box-side drift affecting the single-thread diag only;
t=16 rows are stable and are the protocol quantity.

### 2026-07-18 — M4b.15 session B protocol data points (s2.c2.medium, E-2388G, CHI, quiet-hw FIT)

Baseline binary 9883086 (2026-07-18T01:24:53Z):

```
engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)           8      736.13 ± 2.25        62.40 ± 0.14
inferno (t=1 diag)           1      106.31 ± 0.09        34.26 ± 0.17
llama.cpp                    8     1053.54 ± 1.30        72.84 ± 0.02
llama.cpp (t=1 diag)         1      165.01 ± 0.47        33.15 ± 0.02
ratio (inferno/llama.cpp): pp 0.70x | tg 0.86x
llama.cpp BLAS-build reference: pp 660.47 | tg 72.56 tok/s
ratios (inferno vs llama best-of-builds): pp 0.70x | tg 0.87x
```

M4b.15 branch binary 066bd45, same kernel — no lever shipped
(2026-07-18T01:39:50Z):

```
engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)           8      737.23 ± 4.00        62.52 ± 0.10
inferno (t=1 diag)           1      106.30 ± 0.06        35.19 ± 0.02
llama.cpp                    8     1029.03 ± 53.75       72.71 ± 0.04
llama.cpp (t=1 diag)         1      164.35 ± 0.52        33.56 ± 0.03
ratio (inferno/llama.cpp): pp 0.72x | tg 0.86x
llama.cpp BLAS-build reference: pp 650.10 | tg 72.66 tok/s
ratios (inferno vs llama best-of-builds): pp 0.71x | tg 0.87x
```

### 2026-07-18 — Erratum note (provenance, M4b.15 session entries above)

The "ratios (inferno vs llama best-of-builds)" lines in the two M4b.15
session entries are machine-generated by gate-bench-protocol.sh from an
independent --json run with its own absolutes (the script's full line
reads "…, from the independent --json run"); they are not recomputable
from the adjacent reps=5 tables. Entries unedited.

### 2026-07-18 — M4b.16 session A protocol data point (d2.c1.medium, 6336Y, CHI, quiet-hw FIT)

Fresh llama.cpp baseline, mandatory this cycle (toolchain moved to
inkwell 0.9 / LLVM 22 in ee03def). Branch binary 1bd3838
(2026-07-18T07:41:22Z):

```
engine                 threads        pp512 tok/s        tg128 tok/s
inferno (compiled)          16      873.69 ± 123.66       58.06 ± 0.33
inferno (t=1 diag)           1       75.19 ± 0.08        21.67 ± 0.01
llama.cpp                   16     1209.34 ± 243.15       58.05 ± 0.46
llama.cpp (t=1 diag)         1      118.37 ± 0.14        22.73 ± 0.03
ratio (inferno/llama.cpp): pp 0.72x | tg 1.00x
llama.cpp BLAS-build reference (t pin not honored by BLAS): pp 531.23 | tg 59.91 tok/s
ratios (inferno vs llama best-of-builds, from the independent --json run): pp 0.83x | tg 0.96x
gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET
```

Context: this run is the artifact-cache default path
(INFERNO_EMITTED_ATTN unset → runtime-symbol attention); the M4b.16
lever-vs-baseline comparison lives in the M4b.16 spec §Amendments
(same box, same session).
