#!/usr/bin/env bash
# metal-probe: ask the silicon what it is. Provisions a box, reads
# /proc/cpuinfo + lscpu, prints the cpu-features.json entry that box actually
# justifies, and deletes it. No devpod, no image pull, no devenv — minutes and
# cents, not an hour and dollars.
#
# This exists because the PhoenixNAP products API is not a source of truth. It
# advertises d2.c1.medium as a Dual Xeon Gold 5315Y and delivers a 6336Y
# (f72d67c, observed twice), and it reports four CPU model numbers that name no
# real part in any vendor catalog (Gold 6536 / 6540 / 6436, Xeon 6770P). Those
# types cannot be mapped from documentation, and run.sh refuses to provision a
# type with no table entry — so without this, they are simply unreachable.
#
# Deliberately has no --keep: a probe is a diagnostic, and an operator waiting
# on a one-line answer is exactly who forgets a running box.
#
# Usage: probe.sh <server-type> [--yes] [--ssh-key <pub>] [--os <os>]
#                               [--location <loc>]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"
require_tools curl jq ssh ssh-keygen

TYPE="${1:?usage: probe.sh <server-type> [--yes] [--ssh-key <pub>] [--os <os>] [--location <loc>]}"
shift
YES=0
SSH_KEY="$HOME/.ssh/id_ed25519.pub"
OS="$(metal_default_os)" LOCATION="${METAL_LOCATION:-PHX}"
while [ $# -gt 0 ]; do
  case "$1" in
    --yes) YES=1 ;;
    --ssh-key) shift; SSH_KEY="${1:?--ssh-key needs a path}" ;;
    --os) shift; OS="${1:?--os needs a value}" ;;
    --location) shift; LOCATION="${1:?--location needs a value}" ;;
    *) metal_die "unknown arg: $1" ;;
  esac
  shift
done
if [ "${METAL_PARSE_ONLY:-0}" = 1 ]; then
  printf 'type=%s yes=%s os=%s location=%s\n' "$TYPE" "$YES" "$OS" "$LOCATION"
  exit 0
fi

require_env
[ -f "$SSH_KEY" ] || metal_die "ssh public key not found: $SSH_KEY (--ssh-key)"

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$(git rev-parse --show-toplevel)/target/metal/probe-$TYPE-$RUN_ID"
mkdir -p "$OUT"

# Same teardown discipline as run.sh: once provisioning returns an id, every
# exit path deletes it. pnap_api now retries a 409 on DELETE, so a box whose
# network is still coming up no longer strands the sweep (#19).
SERVER_ID=""
cleanup() {
  local rc="$1"
  if [ -n "$SERVER_ID" ]; then
    echo "metal-probe: deleting server $SERVER_ID"
    if ! (pnap_api DELETE "/bmc/v1/servers/$SERVER_ID" >/dev/null); then
      echo "metal-probe: DELETE FAILED — run 'mise run metal-gc' NOW" >&2
    fi
  fi
  exit "$rc"
}
trap 'cleanup "$?"' EXIT

PRICE=$(pnap_api GET "/billing/v1/products?productCategory=SERVER" \
  | jq -r --arg t "$TYPE" \
      '([.[] | select(.productCode == $t) | .plans[]? | select(.pricingModel == "HOURLY") | .price] | first) // "unknown"')
ADVERTISED=$(pnap_api GET "/billing/v1/products?productCategory=SERVER" \
  | jq -r --arg t "$TYPE" \
      '([.[] | select(.productCode == $t) | .metadata | "\(.cpu) (\(.cpuCount)x\(.coresPerCpu))"] | first) // "unknown"')
echo "metal-probe: $TYPE in $LOCATION at \$$PRICE/hr"
echo "metal-probe: API advertises: $ADVERTISED — this is what we are here to check"
if [ "$YES" != 1 ]; then
  printf 'provision it to read its cpuinfo (the meter starts now)? [y/N] '
  read -r answer
  [ "$answer" = y ] || { echo "metal-probe: aborted"; SERVER_ID=""; exit 0; }
fi

echo "metal-probe: provisioning $TYPE..."
SERVER_ID=$(metal_provision "$TYPE" "$OS" "$LOCATION" "$SSH_KEY" \
  "inferno-probe-${TYPE//./-}-$RUN_ID")
echo "metal-probe: server $SERVER_ID created; waiting for power-on + ssh"
SERVER_IP=$(metal_wait_ready "$SERVER_ID" "$OUT/ssh-probe.log")
echo "metal-probe: ready at $SERVER_IP; reading cpuinfo"

ssh -o StrictHostKeyChecking=accept-new "$(metal_default_ssh_user)@$SERVER_IP" \
  'cat /proc/cpuinfo' > "$OUT/cpuinfo.txt"
ssh -o StrictHostKeyChecking=accept-new "$(metal_default_ssh_user)@$SERVER_IP" \
  'lscpu -p=CORE,SOCKET' > "$OUT/lscpu.txt"

echo
echo "=== $TYPE — what the box actually is ======================================"
grep -m1 '^model name' "$OUT/cpuinfo.txt"
grep -m1 '^vendor_id' "$OUT/cpuinfo.txt"
echo "advertised by the API: $ADVERTISED"
echo
echo "cpu-features.json entry (paste into .types, then replace the TODO source):"
metal_probe_entry "$TYPE" "$OUT/cpuinfo.txt" "$OUT/lscpu.txt" | tee "$OUT/entry.json"
echo
echo "raw capture: $OUT"
echo "NOTE: 'source' must cite the vendor spec sheet for the model above —"
echo "      the flag list is what the box reports, not what the vendor promises."
