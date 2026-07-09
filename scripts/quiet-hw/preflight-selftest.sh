#!/usr/bin/env bash
# Preflight FIT/UNFIT paths against fake cgroup/proc trees — deterministic
# on any box (the real-devpod UNFIT observation is a manual exit-criterion
# step, not this test). Run standalone or via verify.sh --smoke.
set -euo pipefail
PF="$(dirname "$0")/preflight.sh"
fail() { echo "SELFTEST FAIL: $*" >&2; exit 1; }

mktree() { # <cpu.max content> <psi avg10> — builds a fake root, echoes it
  local root; root=$(mktemp -d)
  mkdir -p "$root/cg/podX" "$root/proc/pressure"
  echo "0::/podX" > "$root/proc/self_cgroup"   # see QHW_CGROUP_FILE below
  printf '%s\n' "$1" > "$root/cg/podX/cpu.max"
  printf 'nr_periods 100\nnr_throttled 7\nthrottled_usec 0\n' > "$root/cg/podX/cpu.stat"
  printf 'some avg10=%s avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n' \
    "$2" > "$root/proc/pressure/cpu"
  grep -m1 . /proc/cpuinfo >/dev/null  # sanity: real /proc exists
  printf 'vendor_id\t: FakeVendor\nmodel name\t: Fake CPU\n' > "$root/proc/cpuinfo"
  echo "$root"
}

run_pf() { # <root> — runs preflight against the fake tree, fast calibration
  QHW_CGROUP_ROOT="$1/cg" QHW_PROC_ROOT="$1/proc" \
  QHW_CGROUP_FILE="$1/proc/self_cgroup" \
  QHW_NPROC=16 QHW_CALIB_SECS=1 bash "$PF"
}

# FIT: unquota'd, quiet, enough cores, static cpu.stat (delta 0).
root=$(mktree "max 100000" "0.10")
out=$(run_pf "$root") || fail "expected FIT, got exit $? on: $out"
echo "$out" | grep -q "PREFLIGHT: FIT" || fail "missing FIT line: $out"
echo "$out" | grep -q "FakeVendor"    || fail "missing machine block: $out"

# UNFIT: the devpod signature — quota + pressure + too few cores.
root=$(mktree "800000 100000" "12.50")
if out=$(run_pf "$root" 2>&1); then fail "expected UNFIT to exit nonzero"; fi
out=$(QHW_CGROUP_ROOT="$root/cg" QHW_PROC_ROOT="$root/proc" \
      QHW_CGROUP_FILE="$root/proc/self_cgroup" \
      QHW_NPROC=8 QHW_CALIB_SECS=1 bash "$PF" 2>&1) && fail "UNFIT exited 0"
echo "$out" | grep -q "PREFLIGHT: UNFIT"      || fail "missing UNFIT line: $out"
echo "$out" | grep -q "cgroup quota"          || fail "quota probe silent: $out"
echo "$out" | grep -q "cpu pressure"          || fail "PSI probe silent: $out"
echo "$out" | grep -q "cores: 8"              || fail "core probe silent: $out"

echo "preflight-selftest: OK"
