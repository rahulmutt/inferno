#!/usr/bin/env bash
# M4b.12 admissibility check #2 — instrument perturbation: shipping build
# vs pool-profile build with recording ON (INFERNO_POOL_PROF=1), the M4a
# bench protocol, interleaved rep pairs in one session. VERDICTS ARE
# HUMAN: paste into the M4b.12 spec §Amendments; if the within-session tg
# ratio moves more than 1%, the instrumentation is reworked before any
# attribution is trusted (spec §The dispatch-split instrument).
# Usage: gate-attn-perturb.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-attn-perturb.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PP=16; TG=8; REPS=1; else PP=512; TG=128; REPS=5; fi

smoke_header "gate-attn-perturb (M4b.12 admissibility: ship vs pool-profile-recording A/B)"
machine_block
echo

REPO=$(git rev-parse --show-toplevel)
cargo build --release -q -p inferno
cp "$REPO/target/release/inferno" "$OUT/inferno-ship"
cargo build --release -q -p inferno --features pool-profile
cp "$REPO/target/release/inferno" "$OUT/inferno-prof"

: > "$OUT/perturb-ship.jsonl"; : > "$OUT/perturb-prof.jsonl"
for r in $(seq "$REPS"); do
  echo "--- rep $r: ship ---"
  "$OUT/inferno-ship" bench "$MODEL" --pp "$PP" --tg "$TG" --reps 1 --threads 0 --json \
    | tee -a "$OUT/perturb-ship.jsonl"
  echo "--- rep $r: prof (recording on) ---"
  INFERNO_POOL_PROF=1 "$OUT/inferno-prof" bench "$MODEL" --pp "$PP" --tg "$TG" --reps 1 --threads 0 --json \
    | tee -a "$OUT/perturb-prof.jsonl"
done

echo
echo "inferno tg per interleaved rep (ship | prof-recording):"
paste \
  <(grep -o '"inferno_tg_tok_s": *[0-9.]*' "$OUT/perturb-ship.jsonl" | grep -o '[0-9.]*$') \
  <(grep -o '"inferno_tg_tok_s": *[0-9.]*' "$OUT/perturb-prof.jsonl" | grep -o '[0-9.]*$')
