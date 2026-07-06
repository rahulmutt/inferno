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
