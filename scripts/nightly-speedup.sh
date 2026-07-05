#!/usr/bin/env bash
# Nightly M3 "faster than the interpreter" gate (spec §Testing, Task 17):
# runs `inferno bench-compiled` on the pinned Qwen2.5-0.5B GGUF, which
# generates the same tokens with the compiled backend and with the M1
# interpreter and asserts compiled decode tok/s clears `cli/src/bench.rs`'s
# `MARGIN * interpreter decode tok/s` (exit code carries pass/fail).
# Requires: cc/ld + LLVM 18 (devenv shell — first compile links them), curl.
set -euo pipefail

command -v cargo >/dev/null || { echo "missing tool: cargo (run inside 'devenv shell')" >&2; exit 2; }

GGUF="$(bash "$(dirname "$0")/fetch-qwen-gguf.sh")"

echo "=== M3 speedup gate: compiled vs interpreter decode tok/s ==="
cargo run --release -p inferno -- bench-compiled "$GGUF" \
  --prompt "The capital of France is" --max-tokens 48
