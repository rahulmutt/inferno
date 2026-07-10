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
printf '%s\n' "$list" | cut -f1 | while read -r id; do
  echo "metal-gc: deleting $id"
  pnap_api DELETE "/bmc/v1/servers/$id" >/dev/null
done
