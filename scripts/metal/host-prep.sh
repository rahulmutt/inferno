#!/bin/sh
# Host prep for a freshly provisioned PhoenixNAP box. Invoked over ssh via
# `sudo sh -s` by the default login user (PhoenixNAP cloud images grant
# that user passwordless sudo) — portable sh — Debian or Ubuntu; never
# bash-isms.
#   host-prep.sh <expected-flags-csv> <vocabulary-csv> <expected-vendor>
# Exit codes: 0 ok, 4 = CPU vendor/flag drift vs cpu-features.json (the
# caller aborts BEFORE the slow devpod stage; fix the table in a commit —
# there is deliberately no skip flag).
# Test hooks: METAL_PROC_ROOT (fake /proc), METAL_SKIP_SETUP=1 (no docker/
# governor writes), mirroring quiet-hw's QHW_PROC_ROOT convention.
set -eu
EXPECTED="$1"; VOCAB="$2"; VENDOR="$3"
CPUINFO="${METAL_PROC_ROOT:-/proc}/cpuinfo"

# Drift check FIRST — cheapest possible abort on a mislabeled box.
actual_vendor=$(awk -F': *' '/^vendor_id/ { print $2; exit }' "$CPUINFO")
[ "$actual_vendor" = "$VENDOR" ] || {
  echo "VENDOR DRIFT: cpu-features.json says $VENDOR, box says $actual_vendor" >&2
  exit 4
}
flags=$(awk -F': *' '/^flags/ { print $2; exit }' "$CPUINFO")
drift=0
for f in $(echo "$EXPECTED" | tr ',' ' '); do
  case " $flags " in *" $f "*) ;; *)
    echo "MISSING FLAG: $f (table promises it, box lacks it)" >&2; drift=1 ;;
  esac
done
for f in $(echo "$VOCAB" | tr ',' ' '); do
  case ",$EXPECTED," in *",$f,"*) continue ;; esac
  case " $flags " in *" $f "*)
    echo "UNEXPECTED FLAG: $f (box has it, table omits it)" >&2; drift=1 ;;
  esac
done
[ "$drift" = 0 ] || exit 4
awk -F': *' '/^model name/ { print "cpu: " $2; exit }' "$CPUINFO"

if [ "${METAL_SKIP_SETUP:-0}" != 1 ]; then
  command -v docker >/dev/null 2>&1 || {
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq docker.io
  }
  # devpod's ssh sessions run as the invoking (non-root) user; docker.io's
  # postinst creates the docker group, so add that user to it here — new
  # ssh sessions (which is what devpod opens) pick up the membership.
  if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
    usermod -aG docker "$SUDO_USER"
  fi
  found=0 written=0
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -e "$g" ] || continue
    found=$((found + 1))
    if echo performance > "$g" 2>/dev/null; then
      written=$((written + 1))
    fi
  done
  if [ "$found" -eq 0 ]; then
    echo "governor: no cpufreq interface (leaving as-is)"
  elif [ "$written" -lt "$found" ]; then
    echo "governor: FAILED to set performance on $((found - written))/$found cpus (need root/sudo)" >&2
    exit 1
  else
    echo "governor: performance on $written cpus"
  fi
fi
echo "host-prep: OK"
