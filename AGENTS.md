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
  (LLVM, llama.cpp) come ONLY from `devenv.nix`. The LLVM major version there
  (18) must match the `inkwell` feature flag in `inferno-codegen` (M3+).
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
