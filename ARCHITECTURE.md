# Architecture

Data flow (v1): model file → `ModelDesc` → graph IR → target-aware plan →
LLVM codegen → cached native artifact → runtime executes it.

## Crates

Present (M0–M1):

- `crates/inferno-formats` — GGUF + MLX/safetensors parsing into a
  format-agnostic `ModelDesc`. Deliberately dumb: no graph knowledge, no
  `unsafe`, every read bounded (untrusted input). Downstream code must not
  be able to tell which format a model came from — that's why hyperparams
  are normalized here and not in the graph builder.
- `crates/inferno-graph` — IR + builder + scalar oracle; tolerances live
  here and nowhere else.
- `crates/inferno-runtime` — tokenizer/sampling/generation; drives the
  interpreter in M1, compiled entry points from M3.
- `cli` — the `inferno` binary. Thin; all real logic lives in library crates.

Planned (M2–M4, see the spec for details):

- `inferno-target` — `TargetDesc` (ISA, caches, topology): always an explicit
  input to planning/codegen, never re-probed downstream. A detected target
  and a named-profile target are the same struct — that equivalence is the
  future cross-compile interface.
- `inferno-plan` — fusion islands, weight-layout repacking, static memory plan.
- `inferno-kernels` — hand-tuned matmul microkernels behind a fixed C ABI,
  selected by symbol from generated code.
- `inferno-codegen` — loop IR → LLVM IR (inkwell); JIT + artifact cache.
  The only crate that links LLVM.
- `inferno-core` — the embeddable public API.

## Boundary rules that aren't visible in the code

- Quantization formats are dtypes, not ops: dequant is always fused into the
  consuming kernel, so no crate ever materializes a dequantized weight tensor.
- Shapes are row-major outermost-first everywhere; only the GGUF parser knows
  GGUF stores them reversed.
- `fuzz/` is excluded from the workspace so nightly-only deps can't infect
  the stable build.
