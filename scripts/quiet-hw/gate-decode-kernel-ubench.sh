#!/usr/bin/env bash
# M4b.15 instrument — the phase-marginal decode-attention µbench on quiet
# hardware. Prints the machine block then the criterion output verbatim.
# VERDICTS ARE HUMAN: paste into the M4b.15 spec §Amendments and compute
# the marginals, admissibility, and (post-Lever-1) r there per the spec's
# pre-registered formulas. QHW_SMOKE=1 runs criterion's --test mode
# (plumbing check only, no numbers).
# Usage: gate-decode-kernel-ubench.sh   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

OUT="${QHW_OUT:-$(mktemp -d)}"
smoke_header "gate-decode-kernel-ubench (M4b.15 instrument: phase-marginal µbench)"
machine_block
echo

ARGS=()
if [ "${QHW_SMOKE:-0}" = 1 ]; then ARGS+=(--test); fi
cargo bench -p inferno-kernels --bench attn_decode_phases -- "${ARGS[@]}" \
  | tee "$OUT/attn-decode-phases.txt"
echo
echo "raw criterion data: target/criterion/ (collected by mise run metal)"
