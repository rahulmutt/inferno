#!/usr/bin/env bash
# M4b.7 gate 5 — M4b.6's deferred cross-vendor verdict: re-run the
# reduce-unpack A/B on an Intel box (the SKL µop model said wash, not the
# Zen 2 loss) before declaring the op-reduction lever dead cross-vendor.
# Restores the bench arm by cherry-picking 092b191 (lives only in PR #11's
# pre-squash history: refs/pull/11/head) into a scratch worktree, runs the
# per-process bitwise pre-check, then N interleaved reps, and applies the
# M4b.6 ship-gate arithmetic (fixed weights and conditions from that spec's
# amendments — never re-derived). Verdict destination: M4b.6 spec
# §Amendments (docs/superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md).
# Exit: 0 completed, 3 SKIPPED (non-Intel), else failure.
# Usage: gate-intel-ab.sh [--force-vendor] [--reps N]   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

ARM=092b191
REPS=3; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --force-vendor) FORCE=1 ;;
    --reps) shift; REPS="${1:?--reps needs a value}" ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
OUT="${QHW_OUT:-$(mktemp -d)}"

smoke_header "gate-intel-ab (M4b.6 reduce-unpack cross-vendor A/B)"
machine_block

vendor=$(cpu_vendor)
if [ "$vendor" != GenuineIntel ] && [ "$FORCE" != 1 ]; then
  echo "SKIPPED: vendor is '$vendor', gate needs GenuineIntel (--force-vendor to smoke the plumbing)"
  exit 3
fi

REPO=$(git rev-parse --show-toplevel)
git -C "$REPO" fetch origin refs/pull/11/head
git -C "$REPO" rev-parse --verify --quiet "$ARM^{commit}" >/dev/null \
  || { echo "FATAL: $ARM not found after fetching refs/pull/11/head — if GitHub dropped the PR ref, re-transcribe the arm from the M4b.6 plan's Task 1 (see runbook)" >&2; exit 1; }

WT=$(mktemp -d)/m4b7-ab
git -C "$REPO" worktree add --detach "$WT" HEAD >/dev/null
trap 'git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true' EXIT
git -C "$WT" cherry-pick --no-commit "$ARM"
export CARGO_TARGET_DIR="$REPO/target"   # reuse dep builds across worktrees

# The worktree checks out mise.toml at a path mise has never seen, and a
# mise-shimmed cargo refuses to start under an untrusted config. Only bites
# when the gate runs standalone (the runbook's per-gate fallback, and any
# socket-pinned session); under `mise run` the toolchain is already resolved.
if command -v mise >/dev/null; then
  mise trust --yes "$WT/mise.toml" >/dev/null 2>&1 || true
fi

# Prove the toolchain runs in the worktree BEFORE the pre-check, so that an
# environment failure can never surface as "the kernels disagree bitwise".
(cd "$WT" && cargo --version) >/dev/null 2>&1 \
  || { echo "FATAL: cargo cannot run in the scratch worktree $WT — environment problem, NOT a kernel mismatch. Re-run under 'mise exec -- $0'." >&2; exit 2; }

if [ "${QHW_SMOKE:-0}" = 1 ]; then FILTER='gemv/Q8_0/(inferno-avx2|reduce-unpack)/896x896$'; EXTRA=(--quick); else FILTER='gemv/Q8_0/(inferno-avx2|reduce-unpack)/'; EXTRA=(); fi

echo "bitwise pre-check (arm vs library kernel, --test mode)…"
(cd "$WT" && cargo bench -p inferno-kernels --bench gemv -- 'gemv/Q8_0' --test) \
  > "$OUT/ab-test-mode.out" 2>&1 \
  || { echo "FATAL: bitwise pre-check failed — the arm and the library kernel disagree, or the arm did not build. Do not measure; see $OUT/ab-test-mode.out" >&2; exit 1; }

for rep in $(seq "$REPS"); do
  (cd "$WT" && cargo bench -p inferno-kernels --bench gemv -- "${EXTRA[@]}" "$FILTER") \
    > "$OUT/ab-rep$rep.out" 2>&1
done

MID_SHAPES="896x896 4864x896 896x4864"
declare -A wmed wall
straddle=0
shapes=$(crit_mid_ns "$OUT/ab-rep1.out" 'reduce-unpack/' | awk -F/ '{ print $NF }' | awk '{ print $1 }')
echo
echo "| shape | w per rep (%) | median w (%) | (w = 1 − t_unpack/t_base; positive = arm wins) |"
echo "|---|---|---|---|"
for shape in $shapes; do
  ws=""; pos=0; neg=0
  for rep in $(seq "$REPS"); do
    tb=$(crit_mid_ns "$OUT/ab-rep$rep.out" "inferno-avx2/${shape}\$" | awk '{ print $2 }')
    tu=$(crit_mid_ns "$OUT/ab-rep$rep.out" "reduce-unpack/${shape}\$" | awk '{ print $2 }')
    [ -n "$tb" ] && [ -n "$tu" ] || { echo "FATAL: missing time for $shape rep $rep" >&2; exit 1; }
    w=$(awk -v b="$tb" -v u="$tu" 'BEGIN { printf "%.2f", 100 * (1 - u / b) }')
    ws="$ws $w"
    awk -v w="$w" 'BEGIN { exit !(w > 0) }' && pos=$((pos + 1)) || neg=$((neg + 1))
  done
  wmed[$shape]=$(median $ws); wall[$shape]="$ws"
  [ "$pos" -gt 0 ] && [ "$neg" -gt 0 ] && straddle=1
  echo "| $shape |$ws | ${wmed[$shape]} | |"
done
echo
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: evaluation skipped"
  exit 0
fi

# Ship-gate arithmetic — fixed by the M4b.6 amendments, never re-derived.
c1=0
for shape in $MID_SHAPES; do
  allpos=1
  for w in ${wall[$shape]}; do awk -v w="$w" 'BEGIN { exit !(w > 0) }' || allpos=0; done
  c1=$((c1 + allpos))
done
c2=PASS
for shape in $shapes; do
  awk -v m="${wmed[$shape]}" 'BEGIN { exit !(m < -3.0) }' && c2=FAIL
done
proj=$(awk -v a="${wmed[151936x896]:-0}" -v b="${wmed[896x4864]:-0}" \
           -v c="${wmed[4864x896]:-0}"   -v d="${wmed[896x896]:-0}" \
       'BEGIN { printf "%.2f", 0.270*a + 0.211*b + 0.407*c + 0.087*d }')
echo "condition 1 (w_r>0 every rep on >=2 of 3 mid shapes): $c1 of 3 -> $([ "$c1" -ge 2 ] && echo MET || echo FAILED)"
echo "condition 2 (no shape median w < -3%): $c2"
echo "projected_decode_win = ${proj}% (weights .270/.211/.407/.087 per M4b.6 amendment)"
[ "$straddle" = 1 ] && echo "WARNING: a shape's w_r straddles 0 — if it is a deciding shape, re-run with --reps 6 before recording."
echo "verdict (human, to M4b.6 Amendments): SHIP iff condition 1 MET and condition 2 PASS."
