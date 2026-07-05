#!/usr/bin/env bash
# Downloads (and caches) the pinned Qwen2.5-0.5B-Instruct Q8_0 GGUF shared by
# the nightly differential (scripts/nightly-differential.sh) and the M3
# speedup gate (scripts/nightly-speedup.sh) — same cache dir/key
# (`inferno-test-models-qwen25-05b-v1` in nightly.yml) either script is run
# from. Prints the resulting file path on stdout; callers capture it with
# `$(...)`.
set -euo pipefail

command -v curl >/dev/null || { echo "missing tool: curl (run inside 'devenv shell')" >&2; exit 2; }

CACHE="${INFERNO_TEST_MODEL_DIR:-$HOME/.cache/inferno-tests}"
mkdir -p "$CACHE"
HF="https://huggingface.co"

GGUF="$CACHE/qwen2.5-0.5b-instruct-q8_0.gguf"
[ -f "$GGUF" ] || { curl -fL --retry 3 -o "$GGUF.tmp" \
  "$HF/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf" \
  && mv "$GGUF.tmp" "$GGUF"; }

echo "$GGUF"
