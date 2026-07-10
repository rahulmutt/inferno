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

# The API returns one plan object per (pricingModel, location, ...) combo; once
# we strip everything but pricingModel+price they collapse into ~6 identical
# copies each. `unique` dedups (and sorts, for stable diffs) — the tooling only
# reads the first HOURLY price, so dropping the copies changes nothing but the
# fixture size (~6x smaller and reviewable).
pnap_api GET "/billing/v1/products?productCategory=SERVER" | jq '
  [.[] | {productCode, productCategory,
          plans: ([.plans[]? | {pricingModel, price}] | unique),
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
