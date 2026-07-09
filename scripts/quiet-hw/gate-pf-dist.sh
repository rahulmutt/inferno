#!/usr/bin/env bash
# M4b.7 gate 3 — M4b.4 deferred verdicts: PF_DIST keep/revert (0 vs 4) and
# sweep {2,4,8} for the q8_0 and q4_k AVX2 GEMV prefetch. Builds one bench
# binary per value up front (INFERNO_PF_DIST is a compile-time input), then
# interleaves runs rep-outer/value-inner so A/B pairs sit close in time.
# Ratios are per-rep vs the shipped v=4 binary, medianed across reps.
# Verdict destination: M4b.4 spec §Amendments
# (docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md) — the
# keep/revert call and the Task-3 (interleave) go/no-go stay human.
# Usage: gate-pf-dist.sh   (no model needed; env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }
command -v jq >/dev/null || { echo "missing jq" >&2; exit 2; }

OUT="${QHW_OUT:-$(mktemp -d)}"
VALUES="0 2 4 8"
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  REPS=1; FILTER='gemv/Q8_0/inferno-avx2/896x896$'; EXTRA=(--quick)
else
  REPS=3; FILTER='gemv/(Q8_0|Q4_K)/inferno-avx2/'; EXTRA=()
fi

smoke_header "gate-pf-dist (M4b.4 keep/revert + {2,4,8} sweep)"
machine_block
echo "values={$VALUES} reps=$REPS (interleaved; per-rep ratios vs v=4) filter=$FILTER"
echo

for v in $VALUES; do
  bin=$(INFERNO_PF_DIST=$v cargo bench -p inferno-kernels --bench gemv \
          --no-run --message-format=json 2>"$OUT/pf-build-$v.log" \
        | jq -r 'select(.reason == "compiler-artifact"
                        and .target.name == "gemv") | .executable' | tail -1)
  [ -n "$bin" ] && [ "$bin" != null ] || { echo "FATAL: no bench binary for INFERNO_PF_DIST=$v — see $OUT/pf-build-$v.log" >&2; exit 1; }
  cp "$bin" "$OUT/gemv-pf$v"
done

for rep in $(seq "$REPS"); do
  for v in $VALUES; do
    "$OUT/gemv-pf$v" --bench "${EXTRA[@]}" "$FILTER" \
      > "$OUT/pf$v-rep$rep.out" 2>&1
  done
done

shapes=$(crit_mid_ns "$OUT/pf4-rep1.out" 'inferno-avx2/' | awk '{ print $1 }')
[ -n "$shapes" ] || { echo "FATAL: no criterion times parsed — see $OUT/pf4-rep1.out" >&2; exit 1; }

echo "| bench id | v=0 vs 4 (median %) | v=2 vs 4 | v=8 vs 4 | (negative = faster than shipped v=4) |"
echo "|---|---|---|---|---|"
for id in $shapes; do
  row="| $id |"
  for v in 0 2 8; do
    diffs=""
    for rep in $(seq "$REPS"); do
      t4=$(crit_mid_ns "$OUT/pf4-rep$rep.out" "^${id}$" | awk '{ print $2 }')
      tv=$(crit_mid_ns "$OUT/pf$v-rep$rep.out" "^${id}$" | awk '{ print $2 }')
      [ -n "$t4" ] && [ -n "$tv" ] || { echo "FATAL: missing time for $id v=$v rep=$rep" >&2; exit 1; }
      diffs="$diffs $(pct "$tv" "$t4")"
    done
    row="$row $(median $diffs)% |"
  done
  echo "$row (n/a) |"
done
echo
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: evaluation skipped"
else
  echo "gate inputs (human verdict to M4b.4 Amendments): v=0 column decides"
  echo "keep/revert (v=0 faster => revert prefetch); best of {2,4,8} decides the"
  echo "distance; a >=5%-class win on any DRAM-bound shape is the signal that"
  echo "would authorize M4b.4 Task 3 (interleave)."
fi
