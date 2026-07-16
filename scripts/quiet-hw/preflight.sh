#!/usr/bin/env bash
# M4b.7 environment-fitness preflight (spec §The Preflight): automates the
# probes M4b.1's amendment ran by hand, so fitness is asserted BEFORE any
# data exists. FIT → exit 0 + machine block; UNFIT → exit 1 listing every
# failed probe. Tunables (calibration points from the M4b.1 devpod, which
# must fail all three: cpu.max=800000 100000, PSI some avg10 11–15, +164
# throttled periods during one prefill):
#   QHW_MIN_CPUS (12)  QHW_PSI_MAX (1.0)  QHW_CALIB_SECS (10)
# Test seams: QHW_CGROUP_ROOT QHW_PROC_ROOT QHW_CGROUP_FILE QHW_NPROC.
set -euo pipefail
. "$(dirname "$0")/lib.sh"

CG="${QHW_CGROUP_ROOT:-/sys/fs/cgroup}"
PROC="${QHW_PROC_ROOT:-/proc}"
CGFILE="${QHW_CGROUP_FILE:-/proc/self/cgroup}"
MIN_CPUS="${QHW_MIN_CPUS:-12}"
PSI_MAX="${QHW_PSI_MAX:-1.0}"
CALIB_SECS="${QHW_CALIB_SECS:-10}"
NPROC="${QHW_NPROC:-$(nproc)}"

fails=()

# Probe 1 — core count.
if [ "$NPROC" -lt "$MIN_CPUS" ]; then
  fails+=("cores: $NPROC < required $MIN_CPUS")
fi

# Probe 2 — cgroup-v2 CPU quota anywhere up the hierarchy (the check that
# catches the devpod's 800000 100000).
rel=$(awk -F: '/^0::/ { print $3; exit }' "$CGFILE")
rel="${rel%/}"   # "/" → "" so the flat/root topology is walked exactly once
quota_summary="unquota'd"
path="$rel"
while :; do
  f="$CG$path/cpu.max"
  if [ -f "$f" ]; then
    read -r quota period < "$f"
    if [ "$quota" != "max" ]; then
      fails+=("cgroup quota: $f = '$quota ${period:-}' (must be 'max')")
      quota_summary="$quota/${period:-}"
    fi
  fi
  [ -z "$path" ] && break
  path="${path%/*}"
done

# Probe 3 — external CPU pressure (PSI).
psi=""
if [ -f "$PROC/pressure/cpu" ]; then
  psi=$(awk '/^some/ { sub(/.*avg10=/, ""); sub(/ .*/, ""); print; exit }' \
        "$PROC/pressure/cpu")
  if ! awk -v p="$psi" -v m="$PSI_MAX" 'BEGIN { exit !(p + 0 <= m + 0) }'; then
    fails+=("cpu pressure: some avg10 = $psi > $PSI_MAX")
  fi
else
  fails+=("cpu pressure: $PROC/pressure/cpu missing (cannot verify quiet)")
fi

# Probe 4 — throttling delta across an all-core calibration load (the
# direct version of M4b.1's +164-periods observation).
throttled_now() {
  local total=0 p="$rel" f n
  while :; do
    f="$CG$p/cpu.stat"
    if [ -f "$f" ]; then
      n=$(awk '/^nr_throttled/ { print $2; exit }' "$f")
      total=$((total + ${n:-0}))
    fi
    [ -z "$p" ] && break
    p="${p%/*}"
  done
  echo "$total"
}
before=$(throttled_now)
for _ in $(seq "$NPROC"); do
  (end=$((SECONDS + CALIB_SECS)); while [ "$SECONDS" -lt "$end" ]; do :; done) &
done
wait
after=$(throttled_now)
if [ "$after" -ne "$before" ]; then
  fails+=("throttling: nr_throttled +$((after - before)) during ${CALIB_SECS}s calibration load")
fi

# Probe 5 — invariant TSC (M4b.12: the dispatch-split instrument compares
# rdtsc across threads; only meaningful with constant+nonstop TSC).
tsc_flags=$(awk '/^flags/ { print; exit }' "$PROC/cpuinfo")
tsc_summary=ok
for f in constant_tsc nonstop_tsc; do
  case " $tsc_flags " in
    *" $f "*) ;;
    *) fails+=("tsc: cpuinfo flags lack $f"); tsc_summary=missing ;;
  esac
done

machine_block
echo "probes: cpus=$NPROC quota=$quota_summary psi_some_avg10=${psi:-?} throttled_delta=$((after - before)) calib=${CALIB_SECS}s tsc=$tsc_summary"

if [ "${#fails[@]}" -eq 0 ]; then
  echo "PREFLIGHT: FIT"
else
  echo "PREFLIGHT: UNFIT"
  printf ' - %s\n' "${fails[@]}"
  exit 1
fi
