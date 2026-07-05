# M1 — Graph IR + Interpreter Design

**Date:** 2026-07-05
**Status:** Approved design, pre-implementation
**Parent:** [Inferno v1 design](2026-07-04-inferno-v1-design.md), milestone M1

## Goal

First tokens out. A Llama-family graph builder lowers `ModelDesc` to a small
graph IR; a slow, scalar, obviously-correct interpreter executes it; a native
tokenizer and greedy sampler close the loop. `inferno run <model> -p <prompt>`
streams real (slow) tokens from real quantized models in both GGUF and MLX
formats. The interpreter is the correctness oracle for every later milestone.

## Scope decisions

| Decision | Choice |
|---|---|
| Crates created | `inferno-graph` (IR + builder + interpreter) **and** a minimal `inferno-runtime` (tokenizer, sampling, generation loop) — matches the final architecture; M3 swaps the interpreter for compiled entry points without moving code |
| Tokenizer | Full story in M1: native execution of GGUF-embedded vocab (BPE **and** SentencePiece-style) plus `tokenizer.json` via the `tokenizers` crate for MLX, unified under one trait |
| Dtypes | All five v1 dtypes interpreted: F32, F16, BF16, Q8_0, Q4_K (scalar dequant with pack/unpack property tests) |
| Sampling | Greedy only (argmax, lowest-index tie-break); `Sampler` trait exists, other strategies are M4 |
| CLI | Real `inferno run`, interpreter-backed, streaming; M3 swaps the backend in place |
| Validation | Nightly teacher-forced differential against pinned llama.cpp on a real small quantized model, with tie tolerance |
| Out of scope | Chat templates, non-greedy sampling, engine/session API split (`inferno-core`), mmap weight access, any performance work |

The interpreter targets small models (≤ ~1B params): weights are held
dequantized as f32 in memory (~4 bytes/param). Documented limitation, not
enforced.

## Structural approach

Explicit graph IR, built by a data-driven Llama-family builder, executed by a
separate graph-walking interpreter. Alternatives rejected:

- *Hand-rolled forward pass without IR:* fastest to tokens but builds a
  throwaway oracle — M3 compiles the graph, so an oracle that bypasses the
  graph cannot catch builder bugs.
- *Eager op library without a materialized graph:* the planned IR-dump
  snapshots and M3's planner both need a materialized, printable,
  partitionable graph; deferring it buys almost nothing.

## Extensions to `inferno-formats`

Three prerequisites, all preserving the format-agnostic boundary and
`#![forbid(unsafe_code)]`:

1. **Canonical tensor naming.** Parsers map format-specific names (GGUF
   `blk.0.attn_q.weight`, MLX `model.layers.0.self_attn.q_proj.weight`) to one
   canonical scheme at the edge — e.g. `layers.{i}.attn.q_proj.weight`,
   `layers.{i}.ffn.gate.weight`, `token_embed.weight`, `output_norm.weight`,
   `lm_head.weight`. Without this the builder would need per-format name
   tables, breaking the "downstream can't tell the format" rule. Unmapped
   tensors keep their raw name (the builder ignores them). Existing
   `ModelDesc` snapshots change and are reviewed, never blind-accepted.
2. **`TokenizerSpec` in `ModelDesc`.** Two variants:
   `Embedded { kind: Bpe | Spm, vocab, merges, scores, token_types, special_ids (bos/eos), add_bos }`
   (from GGUF metadata) and `HfJson { path }` (MLX's `tokenizer.json`). This
   names the tokenizer *kind*, not the file format, so the boundary holds.
3. **Tensor data access + scalar quant codecs.** A reader that returns a
   tensor's raw bytes given `ModelDesc` + `TensorDesc` (bounds-checked against
   `data_len`; plain file reads — mmap arrives with the M3 artifact work). A
   `quant` module owning scalar `dequant` (and `pack`, used by
   `gen_fixtures`) for F16/BF16/Q8_0/Q4_K; it lives here because this crate
   already owns `DType` block layouts.

## `inferno-graph`

### Graph IR

A small, closed, printable op set — exactly what the Llama family needs:

- **Values:** tensor IDs with shape and dtype. Dims are `Const(u64)` or `Seq`
  (the single symbolic dimension; batch = 1 is implicit). Weight values carry
  their quant dtype; activations are f32.
- **Ops:** `Embed`, `MatMul` (weight × f32 activation, optional bias — Qwen2
  has attention biases), `RmsNorm` (eps, weight), `Rope` (theta), `Attention`
  (single fused GQA-aware causal op over Q/K/V, carrying its layer index for
  KV-cache identity), `SwiGlu`, `Add` (residual). The final
  norm + lm-head projection reuses `RmsNorm` + `MatMul`; a dedicated `Logits`
  op is added only if implementation shows it earns its keep.
- One flat SSA-style node list per model, built once. Prefill and decode are
  the **same graph** executed with `Seq = prompt_len` or `Seq = 1`. KV state
  is not a graph value: `Attention` reads/appends the executor-supplied KV
  cache keyed by layer index — the graph stays functional everywhere except
  the one genuinely stateful op.
- **Printable:** a stable text dump for `insta` snapshots.

### Builder

`build_graph(&ModelDesc) -> Result<Graph>` — one data-driven builder for the
Llama family (Llama 3.x, Qwen 2.5/3, Mistral). Hyperparams drive shapes;
presence/absence of canonical tensors drives structure:

- attention biases (Qwen2) — present iff the bias tensors exist;
- per-head Q/K norms (Qwen3);
- **tied embeddings** — no `lm_head.weight` means reuse `token_embed.weight`
  (Qwen2.5-0.5B does this).

Unknown architecture, missing required tensors, or shape mismatches are typed
errors, never panics.

### Scalar interpreter

- Walks the node list over `Tensor { shape: Vec<usize>, data: Vec<f32> }`
  with obviously-correct scalar loops. No rayon, no SIMD, no cleverness —
  this is the oracle.
- Weights dequantize to f32 **lazily on first use**, cached per tensor.
- KV cache: plain per-layer f32 buffers allocated up front from
  `max_seq_len`, appended each step.
- Plain f32 accumulation throughout.
- **Per-dtype comparison tolerances are defined here, once**, and imported by
  every later test layer (M2 kernel properties, M3 compiled-vs-reference
  differential).

## `inferno-runtime`

### Tokenizer

```rust
trait Tokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>>;
    fn decode_token(&self, id: u32) -> Vec<u8>; // bytes, for streaming
}
```

- **`NativeTokenizer`** executes GGUF-embedded vocab natively, selected by
  `TokenizerSpec::Embedded.kind`: byte-level **BPE** (merge-rank driven,
  GPT-2 byte↔unicode table, per-model regex pre-tokenization) and
  **SentencePiece-style** (score-driven greedy merge over `▁`-space
  encoding). Special-token handling (BOS/EOS, added-token splitting before
  encoding) is shared plumbing above both.
- **`HfTokenizer`** wraps the `tokenizers` crate over `tokenizer.json`
  (`TokenizerSpec::HfJson`). New pinned dependency with default features
  trimmed — no HTTP/hub features. It doubles as the property-test reference
  for the native implementation.
- `decode_token` returns **bytes** because BPE tokens can split UTF-8
  codepoints; the generation loop buffers and emits only valid UTF-8
  prefixes.

### Sampling

`Sampler` trait; the only M1 implementation is greedy argmax with
lowest-index tie-break (fully deterministic). Temperature/top-k/top-p/min-p
land in M4 without touching the loop.

### Generation loop

```
load ModelDesc → build graph → tokenize(prompt)
→ interpret(graph, Seq = prompt_len)          # prefill, fills KV cache
→ loop: argmax → interpret(graph, Seq = 1)    # decode, appends KV
→ stream bytes until EOS or max_tokens
```

Owned by a single `Generator` struct (the engine/session split is deferred to
`inferno-core` in M4). Stop conditions: EOS token id from `TokenizerSpec`, or
`--max-tokens`. No chat templates — raw completion, matching what the
llama.cpp differential compares against.

## CLI

`inferno run <model> --prompt <text> [--max-tokens N] [--max-seq-len N]` —
streams tokens to stdout as they decode; prints a tok/s summary to stderr.
Help text notes this is the reference interpreter path (slow by design until
M3). Accepts GGUF files and MLX directories, same as `inspect`.

## Error handling

- Each new crate gets a typed `thiserror` enum mirroring `inferno-formats`'
  taxonomy: `inferno-graph::Error` (unsupported architecture,
  missing/mismatched tensors, shape errors), `inferno-runtime::Error`
  (invalid tokenizer spec, prompt exceeds `max_seq_len`, wrapped graph
  errors).
- Model files remain **untrusted input past parsing**: anything reachable
  from file contents surfaces as a typed error, and arena/KV allocation
  sizes derived from `HyperParams` are limit-checked in the same spirit as
  the parser limits. The interpreter may `assert!` purely internal
  invariants — it is the oracle; loud is good.
- The `tokenizers` crate joins the supply-chain surface, covered by the
  existing `cargo audit` / `cargo deny` gates.

## Testing

### Blocking tier (fixtures only; ≤5-minute budget holds)

- **Unit:** each interpreter op vs hand-computed values (RmsNorm, Rope, GQA
  attention with `n_heads ≠ n_kv_heads`, SwiGlu); greedy tie-break; KV
  append correctness across the prefill/decode boundary.
- **Property (`proptest`):** quant `pack → dequant` round-trips within the
  per-dtype tolerance for Q8_0/Q4_K/F16/BF16; native tokenizer encode/decode
  round-trips **and encode-equivalence against the `tokenizers` crate on the
  same vocab** (the fixture vocab ships in both embedded and
  `tokenizer.json` forms).
- **Golden (`insta`):** graph-IR text dumps for the GGUF and MLX fixture
  models; updated `ModelDesc` snapshots for canonical naming.
- **End-to-end fixture differential:** `gen_fixtures` is extended to emit
  tiny models with deterministic pseudorandom real-valued weights in all
  five dtypes plus tokenizer metadata (embedded and `tokenizer.json`). A
  blocking test decodes a few tokens from the GGUF and MLX builds of the
  same fixture and asserts identical output — the two-formats boundary test,
  executable.

### Nightly tier (real model)

**Teacher-forced llama.cpp cross-check** on Qwen2.5-0.5B-Instruct Q8_0
(GGUF; exercises attention biases and tied embeddings) plus its MLX sibling
for the `tokenizer.json` path.

Protocol, designed for determinism:

1. Run the devenv-pinned llama.cpp single-threaded (`-t 1`), greedy, on a
   fixed prompt for N = 64 tokens; capture its token sequence. Greedy
   decoding has no RNG, and the pinned version + fixed thread count make the
   reference sequence reproducible.
2. Replay that sequence **teacher-forced** through the interpreter: at each
   position, feed llama.cpp's history and compare only the next-token
   prediction. Cross-implementation numeric drift therefore cannot compound.
3. A position passes if `our_argmax == llama_token` **or** our top-2 logit
   gap is below epsilon (a genuine near-tie is visible from our own logits —
   no llama.cpp logit extraction needed). The epsilon is a named constant
   defined alongside the per-dtype tolerances in `inferno-graph`.
4. Failures report the divergent position and both top-5 logit sets.

Model files are downloaded and cached by the nightly job, never committed;
the blocking tier never touches the network.

## Milestone acceptance

1. `inferno run <qwen2.5-0.5b-q8_0.gguf> -p "…"` streams coherent text
   (slowly).
2. The MLX build of the same model runs identically.
3. The nightly teacher-forced differential passes at N = 64.
4. Blocking tier green within budget; GGUF/safetensors fuzz targets re-run
   after the parser changes (`mise run fuzz`).

## Risks

- **Native tokenizer fidelity** is the highest-variance item (pre-tokenizer
  regexes, added tokens, byte fallback). Mitigation: encode-equivalence
  property tests against the `tokenizers` crate, plus the real-model
  differential which fails loudly on any tokenization mismatch at position 0.
- **Interpreter throughput** on a 0.5B model may make the nightly check slow
  (tokens/minute). Teacher forcing needs exactly N = 64 forward passes plus
  one prefill; if that exceeds the nightly budget, shrink N — the check's
  power is per-position, not per-length.
- **Canonical naming churn** touches existing parsers and snapshots.
  Mitigation: mechanical mapping tables at the parser edge, snapshot review,
  fuzz re-run.
