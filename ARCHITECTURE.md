# Architecture

Data flow (v1): model file → `ModelDesc` → graph IR → target-aware plan →
LLVM codegen → cached native artifact → runtime executes it.

## Crates

Present (M0–M3):

- `crates/inferno-formats` — GGUF + MLX/safetensors parsing into a
  format-agnostic `ModelDesc`. Deliberately dumb: no graph knowledge, no
  `unsafe`, every read bounded (untrusted input). Downstream code must not
  be able to tell which format a model came from — that's why hyperparams
  are normalized here and not in the graph builder.
- `crates/inferno-graph` — IR + builder + scalar oracle; tolerances live
  here and nowhere else.
- `crates/inferno-runtime` — tokenizer/sampling/generation; drives the
  interpreter in M1, compiled entry points from M3.
- `crates/inferno-target` — `TargetDesc` (ISA, caches, topology): always an
  explicit input to planning/codegen, never re-probed downstream. A detected
  target and a named-profile target are the same struct — that equivalence
  is the future cross-compile interface.
- `crates/inferno-kernels` — hand-tuned matmul microkernels behind a fixed
  C ABI, selected by symbol from generated code.
- `crates/inferno-pool` — persistent fork-join thread pool + the
  `inferno_par_{gemv,gemm,attention}` dispatchers generated code calls for
  every GEMV, GEMM, and attention kernel. This crate is "the caller" the
  kernel boundary rule refers to: it partitions row ranges into 8-row-aligned
  shards and calls the unchanged single-threaded kernels. Third sanctioned
  `unsafe` crate.
- `crates/inferno-plan` — fusion islands, weight-layout repacking, static
  memory plan. Pure data: no LLVM, no codegen, just `Plan`/`Island`/layout
  structs consumed by `inferno-codegen`.
- `crates/inferno-codegen` — loop IR → LLVM IR (inkwell); object emit + link
  to a native artifact. The only crate that links LLVM.
- `crates/inferno-core` — the embeddable public API: mmap's + `dlopen`s the
  cached artifact and calls its compiled entry points. The second sanctioned
  `unsafe` crate (after `inferno-kernels`).
- `cli` — the `inferno` binary. Thin; all real logic lives in library crates.

## Boundary rules that aren't visible in the code

- Quantization formats are dtypes, not ops: dequant is always fused into the
  consuming kernel, so no crate ever materializes a dequantized weight tensor.
- Shapes are row-major outermost-first everywhere; only the GGUF parser knows
  GGUF stores them reversed.
- `fuzz/` is excluded from the workspace so nightly-only deps can't infect
  the stable build.
- Activation-side quant formats (q8a/q8k) are kernel implementation details:
  they live in `inferno-kernels` and never appear in `inferno_formats::DType`.
- Kernel ISA variants are bit-identical by construction (exact integer block
  dots, fixed f32 combine order); the rig asserts exact equality, so any
  "harmless" reassociation in a kernel is a contract break, not an optimization.
- Kernels are single-threaded and row-range partitioned; parallelism lives
  in `inferno-pool`'s `inferno_par_gemv` dispatcher, which generated code
  calls with the full range (M4b.1). Shards are 8-row-aligned, so each
  output row is computed entirely by one thread with the kernel's fixed
  combine order — **thread count never changes output bits**, and the
  tests assert exact equality. A host that never initializes the pool runs
  serially (the dispatcher falls back to one direct kernel call).
