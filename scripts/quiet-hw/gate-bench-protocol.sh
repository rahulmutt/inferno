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

# "At its best" needs a second llama build: the PATH llama-bench is the
# pure-CPU ggml build (the only one that respects the -t pin — see
# devenv.nix and the M4a 2026-07-11 amendment on the BLAS confound), but
# the stock BLAS build can win pp on some boxes by throwing its own
# all-core pool at GEMM. Measure it as a reference row and judge the
# criterion against the per-metric max of the two builds.
lpp_blas=0; ltg_blas=0
if [ -x "${INFERNO_LLAMA_BENCH_BLAS:-}" ]; then
  "$INFERNO_LLAMA_BENCH_BLAS" -m "$MODEL" -p "$PP" -n "$TG" \
    -t "$(phys_cores)" -r "$REPS" -o json \
    > "$OUT/llama-blas.json" 2> "$OUT/llama-blas.log"
  # Plain assignment first: a jq failure inside a <<<"$(...)" redirection is
  # discarded under set -e; as an assignment it aborts loudly on schema drift.
  blas_row=$(llama_bench_pp_tg "$OUT/llama-blas.json")
  read -r lpp_blas ltg_blas <<<"$blas_row"
  echo
  printf "llama.cpp BLAS-build reference (t pin not honored by BLAS): pp %.2f | tg %.2f tok/s\n" \
    "$lpp_blas" "$ltg_blas"
else
  echo
  echo "llama.cpp BLAS-build reference: UNAVAILABLE (INFERNO_LLAMA_BENCH_BLAS unset/missing — criterion judged on the CPU build alone)"
fi

ipp=$(jq -r '.inferno_pp_tok_s' "$OUT/bench.json")
itg=$(jq -r '.inferno_tg_tok_s' "$OUT/bench.json")
lpp_best=$(fmax "$(jq -r '.llama_pp_tok_s' "$OUT/bench.json")" "$lpp_blas")
ltg_best=$(fmax "$(jq -r '.llama_tg_tok_s' "$OUT/bench.json")" "$ltg_blas")
rpp=$(awk -v a="$ipp" -v b="$lpp_best" 'BEGIN { print a / b }')
rtg=$(awk -v a="$itg" -v b="$ltg_best" 'BEGIN { print a / b }')
echo
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: evaluation skipped"
else
  printf "ratios (inferno vs llama best-of-builds, from the independent --json run): pp %.2fx | tg %.2fx\n" "$rpp" "$rtg"
  if awk -v a="$rpp" -v b="$rtg" 'BEGIN { exit !(a > 1.0 && b > 1.0) }'; then
    echo "gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> MET"
  else
    echo "gate: v1 win criterion (pp > 1x AND tg > 1x vs llama at its best) -> NOT MET"
  fi
fi
