#!/usr/bin/env bash
# metal-gc: the cost-leak backstop. Lists every server tagged inferno-metal
# (EXIT traps don't survive a killed terminal), deletes on confirmation.
# Usage: gc.sh [--force]   (--force skips the confirmation, for scripts)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"
require_tools curl jq column
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

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
} | column -t -s "$(printf '\t')"
if [ "$FORCE" != 1 ]; then
  printf 'delete ALL of the above (the meter is running)? [y/N] '
  read -r answer
  [ "$answer" = y ] || { echo "metal-gc: aborted"; exit 0; }
fi
printf '%s\n' "$list" | cut -f1 | while read -r id; do
  echo "metal-gc: deleting $id"
  pnap_api DELETE "/bmc/v1/servers/$id" >/dev/null
done
