# inferno

CPU-first LLM inference engine. Inferno compiles each model for the exact
machine it runs on — hardware-detected code generation via LLVM, with memory
layout, quantization format, and thread partitioning specialized at
setup time — then caches the compiled artifact for instant reloads.

Loads GGUF and MLX (safetensors) models. No GPU: the goal is maximum speed on
commodity hardware, laptops to phones. Written in Rust.

**Status:** pre-release, milestone M2 (targets + AVX2 quantized GEMV kernels).

## Quickstart

Requires [devenv](https://devenv.sh) (native deps: LLVM, llama.cpp) and
[mise](https://mise.jdx.dev) (Rust toolchain + dev tools; task runner):

    devenv shell        # native deps
    mise install        # pinned toolchain
    mise run test       # fast test suite
    lefthook install    # pre-commit hooks (gitleaks, fmt)

Try it:

    cargo run -p inferno -- inspect crates/inferno-formats/tests/fixtures/tiny.gguf
    cargo run -p inferno -- run crates/inferno-formats/tests/fixtures/tiny.gguf --prompt "the" --max-tokens 4

## Common tasks

Run `mise tasks` for the authoritative list — `test`, `test-full`, `lint`,
`fmt`, `audit`, `fuzz`, `differential`.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — crate map and why the boundaries fall
  where they do
- [docs/superpowers/specs/](docs/superpowers/specs/) — design specs
- [docs/threat-model.md](docs/threat-model.md) — what we defend against
