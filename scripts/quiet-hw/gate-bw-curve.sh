#!/usr/bin/env bash
# M4b.10 gate 6 — curve 2: aggregate streaming bandwidth vs lane count, from
# the real Q8_0 GEMV through the real pool. Paired with gate-decode-cap's
# knee on the same box, this is the falsifiability test for the M4b.10
# decision rule: rule 2 (ship a runtime bandwidth probe) fires only if this
# curve's P predicts the measured decode knee. Verdict destination: the
# M4b.10 spec §Amendments
# (docs/superpowers/specs/2026-07-12-m4b10-decode-cap-formula-design.md).
# Usage: gate-bw-curve.sh   (env: QHW_SMOKE QHW_NUMA_NODE)
# No QHW_OUT: unlike gate-decode-cap.sh, this gate has no per-run log to tee
# — the example's stdout below IS the one recordable curve. The orchestrator
# still redirects this script's own stdout to QHW_OUT/gate-bw-curve.out.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

PHYS=$(phys_cores)
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  LANES=2
else
  LANES="$PHYS"
fi

smoke_header "gate-bw-curve (M4b.10 bandwidth saturation)"
machine_block
[ -n "${QHW_NUMA_NODE:-}" ] && echo "numa: pinned to node ${QHW_NUMA_NODE} (cpubind+membind); phys_cores=$PHYS"
echo

# numa_wrap is empty unless QHW_NUMA_NODE is set; unquoted on purpose so it
# expands to zero words in the common case.
$(numa_wrap) cargo run --release -q -p inferno-pool --example bw_curve -- "$LANES"

if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo
  echo "SMOKE: evaluation skipped"
fi
