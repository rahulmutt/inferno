#!/usr/bin/env bash
# Nightly teacher-forced differential vs llama.cpp (spec §Nightly tier).
# Requires: llama-server on PATH (devenv shell), curl, jq, cargo.
set -euo pipefail

for tool in llama-server curl jq cargo; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool (run inside 'devenv shell')" >&2; exit 2; }
done

CACHE="${INFERNO_TEST_MODEL_DIR:-$HOME/.cache/inferno-tests}"
mkdir -p "$CACHE/qwen2.5-0.5b-mlx"
HF="https://huggingface.co"

GGUF="$(bash "$(dirname "$0")/fetch-qwen-gguf.sh")"

MLX="$CACHE/qwen2.5-0.5b-mlx"
for f in config.json model.safetensors tokenizer.json; do
  [ -f "$MLX/$f" ] || { curl -fL --retry 3 -o "$MLX/$f.tmp" \
    "$HF/mlx-community/Qwen2.5-0.5B-Instruct-bf16/resolve/main/$f" \
    && mv "$MLX/$f.tmp" "$MLX/$f"; }
done

PROMPT="The capital of France is"
N_TOKENS=64
PORT=18080
printf '%s' "$PROMPT" > "$CACHE/prompt.txt"

# Single-threaded, no warmup randomness: greedy decoding is deterministic
# for a pinned llama.cpp build + fixed thread count.
llama-server -m "$GGUF" -t 1 --port "$PORT" --host 127.0.0.1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
healthy=0
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    healthy=1
    break
  fi
  sleep 1
done
if [ "$healthy" -ne 1 ]; then
  echo "llama-server never became healthy on port $PORT after 60s" >&2
  exit 1
fi

PROMPT_TOKENS=$(curl -sf "http://127.0.0.1:$PORT/tokenize" \
  -d "$(jq -n --arg c "$PROMPT" '{content: $c}')" | jq -c '.tokens')
GENERATED=$(curl -sf "http://127.0.0.1:$PORT/completion" \
  -d "$(jq -n --arg p "$PROMPT" --argjson n "$N_TOKENS" \
        '{prompt: $p, n_predict: $n, temperature: 0, top_k: 1, samplers: ["top_k"],
          return_tokens: true, cache_prompt: false}')" | jq -c '.tokens')
kill "$SERVER_PID" || true; trap - EXIT

jq -n --argjson p "$PROMPT_TOKENS" --argjson g "$GENERATED" \
  '{prompt_tokens: $p, generated_tokens: $g}' > "$CACHE/tokens.json"
echo "llama.cpp: $(jq length <<<"$PROMPT_TOKENS") prompt + $(jq length <<<"$GENERATED") generated tokens"

echo "=== teacher-forced differential (GGUF) ==="
cargo run --release -p inferno -- diff \
  --model "$GGUF" --prompt-file "$CACHE/prompt.txt" --tokens-file "$CACHE/tokens.json"

echo "=== MLX smoke run ==="
cargo run --release -p inferno -- run "$MLX" --prompt "$PROMPT" --max-tokens 16

echo "differential: PASS"
