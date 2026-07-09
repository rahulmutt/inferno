#!/usr/bin/env bash
# M4b.7 gate 1 — M4b.1 exit criterion: prefill scaling ≥6x at t=12 vs t=1.
# `inferno bench` always emits llama-bench rows; they are recorded as
# environment corroboration (M4b.1-amendment style) — the evaluation uses
# the inferno rows only. Verdict destination: M4b.1 spec §Amendments
# (docs/superpowers/specs/2026-07-06-m4b1-threading-design.md).
# Usage: gate-prefill-scaling.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }
command -v jq >/dev/null || { echo "missing jq" >&2; exit 2; }

MODEL="${1:?usage: gate-prefill-scaling.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  THREADS="1 2"; PP=32; TG=8; REPS=1
else
  THREADS="1 2 4 8 12"; PP=512; TG=128; REPS=5
fi

smoke_header "gate-prefill-scaling (M4b.1 ≥6x @ t=12)"
machine_block
echo

for t in $THREADS; do
  cargo run --release -q -p inferno -- bench "$MODEL" \
    --pp "$PP" --tg "$TG" --reps "$REPS" --threads "$t" --json \
    > "$OUT/prefill-t$t.json"
done

ipp1=$(jq -r .inferno_pp_tok_s "$OUT/prefill-t1.json")
itg1=$(jq -r .inferno_tg_tok_s "$OUT/prefill-t1.json")
case "$ipp1$itg1" in *null*|"") ipp1=0 ;; esac
awk -v a="$ipp1" -v b="$itg1" 'BEGIN { exit !(a + 0 > 0 && b + 0 > 0) }' \
  || { echo "FATAL: t=1 baseline missing/zero in $OUT/prefill-t1.json" >&2; exit 1; }
echo "| t | pp tok/s | pp scale | tg tok/s | tg scale | llama pp (corrob.) | llama tg (corrob.) |"
echo "|---|---|---|---|---|---|---|"
scale12=""
for t in $THREADS; do
  j="$OUT/prefill-t$t.json"
  ipp=$(jq -r .inferno_pp_tok_s "$j"); itg=$(jq -r .inferno_tg_tok_s "$j")
  lpp=$(jq -r .llama_pp_tok_s "$j");   ltg=$(jq -r .llama_tg_tok_s "$j")
  spp=$(awk -v a="$ipp" -v b="$ipp1" 'BEGIN { printf "%.2f", a / b }')
  stg=$(awk -v a="$itg" -v b="$itg1" 'BEGIN { printf "%.2f", a / b }')
  echo "| $t | $ipp | ${spp}x | $itg | ${stg}x | $lpp | $ltg |"
  [ "$t" = 12 ] && scale12="$spp"
done
echo
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: evaluation skipped"
elif awk -v s="$scale12" 'BEGIN { exit !(s + 0 >= 6.0) }'; then
  echo "gate: prefill scale @ t=12 = ${scale12}x (target ≥6x) -> MET"
else
  echo "gate: prefill scale @ t=12 = ${scale12}x (target ≥6x) -> NOT MET"
  echo "note: on a MET=no result, take the M4b.1 spec's attribution fork (serial attention vs memory bandwidth) — see its Amendments."
fi
