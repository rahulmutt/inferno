#!/usr/bin/env bash
# Offline tests for the metal tooling — no network, no credentials.
# Follows scripts/quiet-hw/lib-selftest.sh conventions.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"

fail() { echo "SELFTEST FAIL: $*" >&2; exit 1; }
expect() { # <label> <got> <want>
  [ "$2" = "$3" ] || fail "$1: got '$2', want '$3'"
}

# --- pnap_api: retries 5xx then succeeds; body passthrough ---------------
attempts=$(mktemp); echo 0 > "$attempts"
_pnap_curl() {
  local n; n=$(cat "$attempts"); n=$((n + 1)); echo "$n" > "$attempts"
  if [ "$n" -lt 3 ]; then printf 'server melting\n503\n'
  else printf '{"ok":true}\n200\n'; fi
}
out=$(PNAP_TOKEN=test METAL_RETRY_SLEEP=0 pnap_api GET /bmc/v1/servers)
expect "retry then success" "$out" '{"ok":true}'
expect "attempt count" "$(cat "$attempts")" "3"
rm -f "$attempts"

# --- pnap_api: 401 is fatal, no retry ------------------------------------
if out=$(
  _pnap_curl() { printf 'denied\n401\n'; }
  PNAP_TOKEN=test pnap_api GET /bmc/v1/servers 2>/dev/null
); then fail "401 should be fatal (got '$out')"; fi

# --- pnap_api: gives up after 5 attempts of 503 ---------------------------
attempts=$(mktemp); echo 0 > "$attempts"
if out=$(
  _pnap_curl() {
    local n; n=$(cat "$attempts"); echo $((n + 1)) > "$attempts"
    printf 'nope\n503\n'
  }
  PNAP_TOKEN=test METAL_RETRY_SLEEP=0 pnap_api GET /x 2>/dev/null
); then fail "unending 503 should fail"; fi
expect "bounded retries" "$(cat "$attempts")" "5"
rm -f "$attempts"

# --- require_env ----------------------------------------------------------
if (unset PNAP_CLIENT_ID PNAP_CLIENT_SECRET 2>/dev/null; require_env 2>/dev/null); then
  fail "require_env should fail without credentials"
fi

echo "metal lib-selftest: OK"
