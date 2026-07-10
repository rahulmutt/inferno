#!/usr/bin/env bash
# metal-catalog: PhoenixNAP server types joined with the curated ISA table
# and live availability. Read-only; needs credentials but never provisions.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"
require_tools curl jq
check_features_table || metal_die "cpu-features.json failed its integrity check"

# column (util-linux) only aligns the table; it's cosmetic. Fall back to raw
# TSV when it's absent so the catalog still works on a minimal operator box.
fmt_table() {
  if command -v column >/dev/null 2>&1; then column -t -s "$(printf '\t')"; else cat; fi
}

products=$(mktemp) avail=$(mktemp)
trap 'rm -f "$products" "$avail"' EXIT
pnap_api GET "/billing/v1/products?productCategory=SERVER" > "$products"
pnap_api GET "/billing/v1/product-availability?productCategory=SERVER&minQuantity=1" > "$avail"
{
  printf 'TYPE\tCPU\tVENDOR\tCORES\tFLAGS\tUSD/HR\tIN-STOCK\n'
  catalog_join "$products" "$avail" "$(features_table)"
} | fmt_table
