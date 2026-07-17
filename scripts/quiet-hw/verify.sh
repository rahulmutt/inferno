#!/usr/bin/env bash
# M4b.7 orchestrator: fitness preflight, then the five deferred-verdict
# gates, everything tee'd under one timestamped results dir. UNFIT preflight
# is a hard stop unless --smoke (which stamps every output NON-RECORDABLE).
# A gate that fails is recorded FAILED and the pass continues — on rare
# quiet hardware, partial data beats an aborted run. Scripts never write to
# docs/; paste verdicts into the owning spec's Amendments by hand (see
# docs/runbooks/quiet-hw-verification.md).
# Usage: verify.sh <model.gguf> [--smoke] [--out-dir D] [--force-vendor]
#                                [--i-know-what-im-doing]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (run inside 'devenv shell')" >&2; exit 2; }

MODEL="${1:?usage: verify.sh <model.gguf> [--smoke] [--out-dir D] [--force-vendor] [--i-know-what-im-doing]}"
shift
SMOKE=0; OUTDIR=""; OVERRIDE=0; AB_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --out-dir) shift; OUTDIR="${1:?--out-dir needs a value}" ;;
    --force-vendor) AB_ARGS+=(--force-vendor) ;;
    --i-know-what-im-doing) OVERRIDE=1 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
REPO=$(git rev-parse --show-toplevel)
OUT="${OUTDIR:-$REPO/target/quiet-hw/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
export QHW_OUT="$OUT" QHW_SMOKE="$SMOKE"

# Selftests first — cheap, and smoke is the pass that runs them (spec §Testing).
bash "$HERE/lib-selftest.sh"
bash "$HERE/preflight-selftest.sh"

declare -A status
if bash "$HERE/preflight.sh" 2>&1 | tee "$OUT/preflight.out"; then
  status[preflight]=FIT
else
  status[preflight]=UNFIT
  if [ "$SMOKE" = 1 ]; then
    echo "(smoke mode: continuing on unfit hardware; all output NON-RECORDABLE)"
  elif [ "$OVERRIDE" = 1 ]; then
    # Spec §Risks escape hatch for a preflight false-positive the operator
    # has judged wrong: every output gets the UNFIT-OVERRIDE stamp so
    # provenance survives into whatever gets recorded.
    export QHW_OVERRIDE=1
    status[preflight]=UNFIT-OVERRIDE
    echo "(UNFIT-OVERRIDE: operator forced the run; every output is stamped)"
  else
    echo "ABORT: preflight UNFIT and not --smoke; no gate may run (spec: UNFIT = hard stop)." >&2
    exit 1
  fi
fi

run_gate() { # <name> <cmd...>
  local name="$1"; shift
  echo "=== gate: $name ==="
  local rc=0
  "$@" 2>&1 | tee "$OUT/gate-$name.out" || rc=$?
  case "$rc" in
    0) status[$name]=PASS ;;
    3) status[$name]=SKIPPED ;;
    *) status[$name]=FAILED ;;
  esac
}

run_gate prefill-scaling bash "$HERE/gate-prefill-scaling.sh" "$MODEL"
run_gate decode-cap      bash "$HERE/gate-decode-cap.sh" "$MODEL"
run_gate bw-curve        bash "$HERE/gate-bw-curve.sh"
run_gate pf-dist         bash "$HERE/gate-pf-dist.sh"
run_gate bench-protocol  bash "$HERE/gate-bench-protocol.sh" "$MODEL"
run_gate prefill-attr    bash "$HERE/gate-prefill-attr.sh" "$MODEL"
run_gate decode-attr     bash "$HERE/gate-decode-attr.sh" "$MODEL"
run_gate attn-split      bash "$HERE/gate-attn-split.sh" "$MODEL"
run_gate attn-perturb    bash "$HERE/gate-attn-perturb.sh" "$MODEL"
run_gate attn-perf       bash "$HERE/gate-attn-perf.sh" "$MODEL"
run_gate intel-ab        bash "$HERE/gate-intel-ab.sh" "${AB_ARGS[@]}"

{
  [ "$SMOKE" = 1 ] && echo "### SMOKE — NON-RECORDABLE (plumbing check on unfit hardware; never paste into a spec) ###"
  [ "${QHW_OVERRIDE:-0}" = 1 ] && echo "### UNFIT-OVERRIDE (preflight failed; operator forced the run — record the override alongside any data) ###"
  echo "# quiet-hw verification pass — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  machine_block
  echo
  echo "| stage | status |"
  echo "|---|---|"
  echo "| preflight | ${status[preflight]} |"
  for g in prefill-scaling decode-cap bw-curve pf-dist bench-protocol prefill-attr decode-attr attn-split attn-perturb attn-perf intel-ab; do
    echo "| $g | ${status[$g]} |"
  done
  echo
  echo "PASS = script completed and printed its table — the VERDICTS are human;"
  echo "paste each gate's output into its owning spec's Amendments per"
  echo "docs/runbooks/quiet-hw-verification.md."
} | tee "$OUT/summary.md"
echo "results: $OUT"

for g in prefill-scaling decode-cap bw-curve pf-dist bench-protocol prefill-attr decode-attr attn-split attn-perturb; do
  [ "${status[$g]}" = PASS ] || exit 1
done
[ "${status[intel-ab]}" = FAILED ] && exit 1
[ "${status[attn-perf]}" = FAILED ] && exit 1
exit 0
