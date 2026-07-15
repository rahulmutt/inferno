#!/usr/bin/env bash
# M4b.7 gate 2 — M4b.5 exit-criterion leg 2: decode-thread sweep. The
# shipped default cap clamp(active/3, 2, active) must meet-or-beat the best
# fixed cap, remove the high-thread regression, and leave t=1 decode
# unchanged. Sweeps INFERNO_DECODE_THREADS with rounds interleaved (rep-
# outer, cap-inner) per the standing M4b discipline. Verdict destination:
# M4b.5 spec §Amendments
# (docs/superpowers/specs/2026-07-08-m4b5-phase-aware-decode-threading-design.md).
# Usage: gate-decode-cap.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-decode-cap.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
PROMPT="The capital of France is"
PHYS=$(phys_cores)
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  CAPS="1 2"; REPS=1; MAXTOK=8
else
  CAPS=$(cap_grid "$PHYS"); REPS=3; MAXTOK=128
fi

smoke_header "gate-decode-cap (M4b.5 default-vs-best sweep)"
machine_block
echo "sweep: caps={$CAPS} + default + t1 | reps=$REPS (interleaved rounds) | max-tokens=$MAXTOK"
numa_require   # a pinned session that cannot pin must die, not measure unpinned
[ -n "${QHW_NUMA_NODE:-}" ] && echo "numa: pinned to node ${QHW_NUMA_NODE} (cpubind+membind); phys_cores=$PHYS"
echo

one_run() { # <cap: number|default|t1> -> decode tok/s on stdout
  local threads=0 envset=()
  case "$1" in
    default) ;;                       # heuristic path: env unset
    t1)      threads=1 ;;             # t=1 decode-unchanged row
    *)       envset=(INFERNO_DECODE_THREADS="$1") ;;
  esac
  # numa_wrap is empty unless QHW_NUMA_NODE is set; unquoted on purpose so it
  # expands to zero words in the common case.
  env "${envset[@]}" $(numa_wrap) cargo run --release -q -p inferno -- run "$MODEL" \
    -p "$PROMPT" --max-tokens "$MAXTOK" --threads "$threads" 2>&1 \
    | tee -a "$OUT/decode-cap-runs.log" | decode_toks -
}

declare -A samples
for rep in $(seq "$REPS"); do
  for cap in $CAPS default t1; do
    tgs=$(one_run "$cap")
    [ -n "$tgs" ] || { echo "FATAL: no decode tok/s parsed (rep $rep cap $cap) — see $OUT/decode-cap-runs.log" >&2; exit 1; }
    samples[$cap]="${samples[$cap]:-} $tgs"
  done
done

echo "| cap | decode tok/s (median of $REPS) | per-rep |"
echo "|---|---|---|"
best_cap=""; best=0
for cap in $CAPS default t1; do
  med=$(median ${samples[$cap]})
  echo "| $cap | $med |${samples[$cap]} |"
  case "$cap" in default|t1) ;; *)
    if awk -v m="$med" -v b="$best" 'BEGIN { exit !(m > b) }'; then
      best="$med"; best_cap="$cap"
    fi ;;
  esac
done
def_med=$(median ${samples[default]})
echo
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: evaluation skipped"
else
  echo "knee (best fixed cap): $best_cap ($best tok/s median)"
  # Discipline: per-rep default/best ratios (same interleaved round), THEN median.
  read -ra def_arr <<< "${samples[default]}"
  read -ra best_arr <<< "${samples[$best_cap]}"
  ratios=""
  for i in $(seq 0 $((REPS - 1))); do
    ratios="$ratios $(pct "${def_arr[$i]}" "${best_arr[$i]}")"
  done
  echo "default clamp(active/3,2,active): $def_med tok/s median -> $(median $ratios)% vs best fixed (median of per-rep ratios)"
  echo "gate inputs (human verdict to M4b.5 Amendments): default meets-or-beats"
  echo "best-fixed? high-thread regression gone (compare cap=$PHYS row vs knee)?"
  echo "t=1 decode unchanged (t1 row vs prior recorded t=1)?"
fi
