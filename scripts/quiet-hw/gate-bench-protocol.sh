#!/usr/bin/env bash
# M4b.7 gate 4 — the official M4a comparison protocol; the ONLY place the
# v1 win criterion ("beat llama.cpp prefill AND decode tok/s at its best
# thread count") can be judged. Runs the table form for the human record
# and the --json form for the evaluation, defaults pp=512 tg=128 reps=5
# threads=0 (physical cores), per the M4a spec. Verdict destination: M4a
# spec §Amendments (docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md).
# Usage: gate-bench-protocol.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }
command -v jq >/dev/null || { echo "missing jq" >&2; exit 2; }
command -v llama-bench >/dev/null || { echo "missing llama-bench (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-bench-protocol.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PP=32; TG=8; REPS=1; else PP=512; TG=128; REPS=5; fi

smoke_header "gate-bench-protocol (M4a protocol / v1 win criterion)"
machine_block
echo

cargo run --release -q -p inferno -- bench "$MODEL" \
  --pp "$PP" --tg "$TG" --reps "$REPS" --threads 0 \
  | tee "$OUT/bench-table.txt"
cargo run --release -q -p inferno -- bench "$MODEL" \
  --pp "$PP" --tg "$TG" --reps "$REPS" --threads 0 --json \
  > "$OUT/bench.json"

rpp=$(jq -r '.inferno_pp_tok_s / .llama_pp_tok_s' "$OUT/bench.json")
rtg=$(jq -r '.inferno_tg_tok_s / .llama_tg_tok_s' "$OUT/bench.json")
echo
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: evaluation skipped"
else
  printf "ratios (inferno/llama.cpp, from the independent --json run): pp %.2fx | tg %.2fx\n" "$rpp" "$rtg"
  if awk -v a="$rpp" -v b="$rtg" 'BEGIN { exit !(a > 1.0 && b > 1.0) }'; then
    echo "gate: v1 win criterion (pp > 1x AND tg > 1x) -> MET"
  else
    echo "gate: v1 win criterion (pp > 1x AND tg > 1x) -> NOT MET"
  fi
fi
