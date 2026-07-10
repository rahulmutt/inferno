#!/usr/bin/env bash
# Re-record scripts/metal/fixtures/ from the live BMC API (needs
# credentials). Sanitizes to only the fields the tooling reads — fixtures
# are committed, so nothing account-specific may land in them. Re-run when
# the PhoenixNAP catalog changes, then re-run lib-selftest.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"
require_env
require_tools curl jq
mkdir -p "$HERE/fixtures"

pnap_api GET "/billing/v1/products?productCategory=SERVER" | jq '
  [.[] | {productCode, productCategory,
          plans: [.plans[]? | {pricingModel, price}],
          metadata: {cpu: .metadata.cpu, cpuCount: .metadata.cpuCount,
                     coresPerCpu: .metadata.coresPerCpu,
                     cpuFrequency: .metadata.cpuFrequency}}]' \
  > "$HERE/fixtures/products.json"

pnap_api GET "/billing/v1/product-availability?productCategory=SERVER&minQuantity=1" | jq '
  [.[] | {productCode,
          locationAvailabilityDetails:
            [.locationAvailabilityDetails[]? | {location, minQuantityAvailable}]}]' \
  > "$HERE/fixtures/availability.json"

echo "recorded $(jq length "$HERE/fixtures/products.json") products; re-run lib-selftest.sh"
