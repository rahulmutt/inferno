#!/usr/bin/env bash
# metal-gc: the cost-leak backstop. Lists every server tagged inferno-metal
# (EXIT traps don't survive a killed terminal), deletes on confirmation.
# Usage: gc.sh [--force]   (--force skips the confirmation, for scripts)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"
require_tools curl jq
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# column (util-linux) only aligns the listing; fall back to raw TSV when it's
# absent so gc — the cost-leak backstop — never fails to run on a minimal box.
fmt_table() {
  if command -v column >/dev/null 2>&1; then column -t -s "$(printf '\t')"; else cat; fi
}

servers=$(mktemp)
trap 'rm -f "$servers"' EXIT
pnap_api GET /bmc/v1/servers > "$servers"
list=$(gc_candidates "$servers")
if [ -z "$list" ]; then
  echo "metal-gc: no $METAL_TAG servers running"
  exit 0
fi
{
  printf 'ID\tTYPE\tHOSTNAME\tPROVISIONED\n'
  printf '%s\n' "$list"
} | fmt_table
if [ "$FORCE" != 1 ]; then
  printf 'delete ALL of the above (the meter is running)? [y/N] '
  read -r answer
  [ "$answer" = y ] || { echo "metal-gc: aborted"; exit 0; }
fi
# Keep hammering the ones that refuse to go. A server whose network is still
# provisioning answers 409 for minutes (pnap_api already retries that for a
# couple of them), and a single stubborn id must never abort the sweep and
# leave the rest — or itself — billing. Bounded, then a loud failure naming
# exactly what is still alive.
pending=$(printf '%s\n' "$list" | cut -f1 | tr '\n' ' ')
deadline=$((SECONDS + ${METAL_GC_TIMEOUT:-900}))
while [ -n "${pending// /}" ]; do
  still=""
  for id in $pending; do
    echo "metal-gc: deleting $id"
    if pnap_api DELETE "/bmc/v1/servers/$id" >/dev/null; then
      echo "metal-gc: deleted $id"
    else
      still="$still $id"
    fi
  done
  pending="$still"
  [ -n "${pending// /}" ] || break
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "metal-gc: STILL ALIVE after $((SECONDS))s:$pending" >&2
    echo "metal-gc: THE METER MAY BE RUNNING — rerun, or delete in the PhoenixNAP portal" >&2
    exit 1
  fi
  echo "metal-gc:$pending not deletable yet — retrying in ${METAL_GC_SLEEP:-30}s"
  sleep "${METAL_GC_SLEEP:-30}"
done
echo "metal-gc: all deleted"
