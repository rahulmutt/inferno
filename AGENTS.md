# Agent instructions — inferno

Everything derivable from code is not repeated here. Read
[ARCHITECTURE.md](ARCHITECTURE.md) for the crate map and
[docs/superpowers/specs/2026-07-04-inferno-v1-design.md](docs/superpowers/specs/2026-07-04-inferno-v1-design.md)
for the v1 design.

## Non-obvious constraints

- **Workflows are mise tasks** (`mise tasks`): use `mise run test` / `lint` /
  `audit` / `fuzz` — CI runs the same names. Don't hand-roll cargo invocations
  in docs or CI.
- **Toolchain:** rust + dev tools are mise-pinned (`mise.toml`); native deps
  (LLVM, llama.cpp) come ONLY from `devenv.nix`. The devenv LLVM
  major.minor there must match the `inkwell` feature flag in
  `inferno-codegen` (M3+) exactly — currently `llvm18-1` (`Cargo.toml`)
  against LLVM 18.1.8 (`devenv.nix`'s `pkgs.llvmPackages_18`,
  `LLVM_SYS_180_PREFIX`). Bumping one without the other breaks the build.
- **`inferno-formats` must stay `#![forbid(unsafe_code)]`** and every parser
  read bounded — model files are untrusted input (see
  [docs/threat-model.md](docs/threat-model.md)). Touching parser code means
  running `mise run fuzz -- gguf_parse` / `-- safetensors_parse` locally.
- **`ModelDesc` is format-agnostic:** never let a downstream crate learn
  which file format a model came from.
- **Tensor shapes are row-major, outermost first** everywhere in inferno;
  GGUF stores dims reversed and the GGUF parser normalizes them on ingest.
- **Snapshots (insta):** review with `cargo insta review`; never blind-accept.
- Fixture files under `tests/fixtures/` and `fuzz/corpus/` are generated —
  regenerate with `cargo run -p inferno-formats --example gen_fixtures`,
  don't hand-edit.
- **Rope style is coupled to weight layout:** GGUF llama-arch files carry
  *row-permuted* Q/K weights (Interleaved rope); MLX/HF files are unpermuted
  (HalfSplit). `HyperParams::rope_style` records which; the fixture
  differential (`inferno-graph/tests/differential.rs`) guards the coupling.
  Never "simplify" one side without the other.
- **Embedded and JSON tokenizer fixtures must stay equivalent:**
  `fixtures::tiny_vocab()` feeds both the GGUF metadata and
  `mlx/tokenizer.json`; the BPE equivalence property tests depend on it.
- **`LOGIT_TIE_EPSILON`** (`inferno-graph/src/tolerance.rs`) is tuned against
  the gap distributions printed by `mise run differential` — adjust it with
  observed data, never to make a red nightly green without understanding the
  divergence.
- **`inferno-kernels` and `inferno-core` are the only crates allowed
  `unsafe`** (`inferno-kernels`: intrinsics + the C ABI; `inferno-core`,
  M3+: mmap + `dlopen` + calling compiled entry points by raw fn pointer).
  Both opt out of the workspace's `unsafe_code = "deny"` lint with their own
  `[lints.rust]` table. Scalar and SIMD kernel variants must stay
  bit-identical — the rig asserts exact equality.
- **Kernel perf numbers come only from `mise run bench-kernels`** inside the
  devenv shell on quiet hardware; CI runners are noise. Record data points in
  the M2 spec's amendments section.
- **`gemv_rel_tol`** follows the same rule as `LOGIT_TIE_EPSILON`: tuned
  against observed error distributions (the rig's ignored `observed_error_*`
  diagnostics), never to make a red test green.
- **Object-emit + `dlopen` is the only compile path (M3+):** `inferno
  compile`/`run` emit an LLVM module to a native object, link it to
  `model.so`, and `dlopen` it from the on-disk artifact cache — there is no
  in-memory JIT path.
- **KV is stored as `f32` in M3** (not F16), specifically to keep the
  compiled-vs-interpreter differential free of an extra rounding term; don't
  "optimize" it to F16 without re-deriving tolerances (M4 work).
- **After touching codegen op lowerings**, run
  `cargo test -p inferno-codegen --test differential` (the
  compiled-vs-interpreter correctness gate) and
  `cargo test -p inferno-core --test artifact` (the artifact-level
  differential); never loosen `logits_abs_tol` to make either green.
