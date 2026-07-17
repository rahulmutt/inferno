#!/usr/bin/env bash
# M4b.13 mid-milestone attribution gate — the fresh split-bracket t=1
# prefill profile the pre-registered ladder rule consumes (spec §Mid-
# Milestone Gate). Prints the t=1 prefill op table verbatim; the pp ratios
# come from gate-bench-protocol.sh in the same session. VERDICTS ARE
# HUMAN: paste into the M4b.13 spec §Amendments and compute there
# matmul_share (sum of the prefill table's matmul:* rows / prefill total)
# and the ceiling check pp_ratio / (1 - matmul_share * 0.5) >= 1.0, per
# the spec's pre-registered rule (docs/runbooks/quiet-hw-verification.md).
# Usage: gate-prefill-attr.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-prefill-attr.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PROMPT_BYTES=256; MAXTOK=8; else PROMPT_BYTES=2048; MAXTOK=32; fi

smoke_header "gate-prefill-attr (M4b.13: split-bracket t=1 prefill profile)"
machine_block
echo

PROMPT="$(head -c "$PROMPT_BYTES" /dev/urandom | base64 | tr -d '\n')"

echo "--- t=1 prefill profile (split brackets) ---"
cargo run --release -q -p inferno -- run "$MODEL" \
  --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads 1 --profile \
  > "$OUT/prefill-attr-t1.txt" 2>&1
sed -n '/^profile \[/,$p' "$OUT/prefill-attr-t1.txt"
