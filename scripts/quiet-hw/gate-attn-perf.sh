#!/usr/bin/env bash
# M4b.12 rider — perf-counter capture on the SHIPPING build: topdown (or
# -d fallback) + scheduler events around a decode-dominant run. This is
# the worker-side view calling-thread self-measurement can't see, and the
# escalation evidence if the menu guard fires. Whole-process counters
# (prefill included) — the workload is shaped decode-heavy (short prompt,
# long generation); interpretation is controller work. VERDICTS ARE HUMAN.
# Exit: 0 completed, 3 SKIPPED (no perf), else failure.
# Usage: gate-attn-perf.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }
command -v perf >/dev/null || { echo "SKIPPED: perf not on PATH"; exit 3; }

MODEL="${1:?usage: gate-attn-perf.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then MAXTOK=8; else MAXTOK=256; fi

smoke_header "gate-attn-perf (M4b.12 rider: topdown + scheduler counters, shipping build)"
machine_block
echo

REPO=$(git rev-parse --show-toplevel)
cargo build --release -q -p inferno
BIN="$REPO/target/release/inferno"
PROMPT="$(head -c 256 /dev/urandom | base64 | tr -d '\n')"

if perf stat --topdown -- true >/dev/null 2>&1; then TD=(--topdown); else TD=(-d); fi
echo "--- perf stat ${TD[*]} ---"
perf stat "${TD[@]}" -o "$OUT/attn-perf-topdown.txt" -- \
  "$BIN" run "$MODEL" --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads 0 >/dev/null
cat "$OUT/attn-perf-topdown.txt"
echo
echo "--- perf stat scheduler events ---"
perf stat -e task-clock,context-switches,cpu-migrations -o "$OUT/attn-perf-sched.txt" -- \
  "$BIN" run "$MODEL" --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads 0 >/dev/null
cat "$OUT/attn-perf-sched.txt"
