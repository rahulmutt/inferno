# inferno

CPU-first LLM inference engine. Inferno compiles each model for the exact
machine it runs on — hardware-detected code generation via LLVM, with memory
layout, quantization format, and thread partitioning specialized at
setup time — then caches the compiled artifact for instant reloads.

Loads GGUF and MLX (safetensors) models. No GPU: the goal is maximum speed on
commodity hardware, laptops to phones. Written in Rust.

**Status:** v1 complete (2026-07-18). Inferno loads GGUF and MLX models,
compiles them through LLVM to a cached native artifact, and runs a threaded
compiled inference path end to end. The v1 goal was to beat llama.cpp on
both prefill and decode throughput — **that goal was not met**. On the
recorded quiet-hardware benches (Qwen2.5-0.5B-Instruct Q8_0, pp512/tg128,
full-thread, vs llama.cpp's best builds) inferno reaches pp 0.83x / tg 0.96x
on a 16-core Xeon Gold 6336Y and pp 0.69x / tg 0.86x on an 8-core Xeon
E-2388G. The remaining gap is attributed to measured ceilings rather than
unexplored headroom — see the
[v1 close record](docs/superpowers/specs/2026-07-18-v1-close-design.md).

## Quickstart

Requires [devenv](https://devenv.sh) (native deps: LLVM 22.1.8 + a C toolchain
for linking, llama.cpp) and [mise](https://mise.jdx.dev) (Rust toolchain +
dev tools; task runner):

    devenv shell        # native deps
    mise install        # pinned toolchain
    mise run test       # fast test suite
    lefthook install    # pre-commit hooks (gitleaks, fmt)

Try it:

    cargo run -p inferno -- inspect crates/inferno-formats/tests/fixtures/tiny.gguf
    cargo run -p inferno -- run crates/inferno-formats/tests/fixtures/tiny.gguf --prompt "the" --max-tokens 4

The `run` command above compiles the model first (LLVM codegen + link to a
cached `model.so` — needs the devenv shell's LLVM 22.1.8 + linker the first
time; later runs reuse the cached artifact) and runs the compiled path by
default. Force a compile without generating, or inspect where the artifact
landed, with `inferno compile`:

    cargo run -p inferno -- compile crates/inferno-formats/tests/fixtures/tiny.gguf

Pass `--interp` to `run` to use the M1 scalar interpreter instead (slow by
design; a cross-check against the compiled path, not for everyday use):

    cargo run -p inferno -- run crates/inferno-formats/tests/fixtures/tiny.gguf --prompt "the" --max-tokens 4 --interp

## Run options

`INFERNO_DECODE_THREADS=N` caps the number of threads the *decode* phase
shards across (prefill still uses all `--threads`). Decode is
memory-bandwidth-bound, so more threads than saturate DRAM bandwidth only
add overhead; the default is a fraction of cores. Output is identical for
any value.

## Common tasks

Run `mise tasks` for the authoritative list — `test`, `test-full`, `lint`,
`fmt`, `audit`, `fuzz`, `differential`.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — crate map and why the boundaries fall
  where they do
- [docs/superpowers/specs/](docs/superpowers/specs/) — design specs
- [docs/threat-model.md](docs/threat-model.md) — what we defend against
