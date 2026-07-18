#!/usr/bin/env bash
# M4b.16 gate — emitted-attention lever vs runtime-symbol baseline, same
# binary, same box. Prints both bench tables; the gate arithmetic
# (tg_lever/tg_base per the M4b.11 thresholds: >=5% both boxes ship,
# <3% both STOP) is HUMAN — record tables and verdict in the M4b.16 spec
# §Amendments. Fresh llama.cpp baselines are gate-bench-protocol.sh's
# job (run it in the same session; toolchain changed, they are mandatory).
# Usage: gate-emitted-attn.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-emitted-attn.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PP=32; TG=8; REPS=1; else PP=512; TG=128; REPS=5; fi

smoke_header "gate-emitted-attn (M4b.16 lever vs baseline)"
machine_block
echo

cargo build --release -q -p inferno

echo "== baseline (INFERNO_EMITTED_ATTN=0) =="
INFERNO_EMITTED_ATTN=0 target/release/inferno bench "$MODEL" \
  --pp "$PP" --tg "$TG" --reps "$REPS" --threads 0 \
  | tee "$OUT/bench-baseline.txt"

echo "== lever (INFERNO_EMITTED_ATTN=1) =="
INFERNO_EMITTED_ATTN=1 target/release/inferno bench "$MODEL" \
  --pp "$PP" --tg "$TG" --reps "$REPS" --threads 0 \
  | tee "$OUT/bench-lever.txt"

echo
echo "artifacts: $OUT (bench-baseline.txt, bench-lever.txt)"
echo "VERDICT IS HUMAN: compute tg_lever/tg_base; thresholds per M4b.16 spec."
