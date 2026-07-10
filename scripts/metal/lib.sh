#!/usr/bin/env bash
# Shared helpers for the PhoenixNAP bare-metal bench tooling (spec:
# docs/superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md).
# Sourced by catalog.sh, gc.sh, run.sh, record-fixtures.sh. Tested by
# lib-selftest.sh, which stubs _pnap_curl — every API byte flows through it.

METAL_TAG="inferno-metal"
PNAP_AUTH_URL="${PNAP_AUTH_URL:-https://auth.phoenixnap.com/auth/realms/BMC/protocol/openid-connect/token}"
PNAP_API_BASE="${PNAP_API_BASE:-https://api.phoenixnap.com}"

metal_die() { echo "metal: $*" >&2; exit 2; }

require_env() {
  [ -n "${PNAP_CLIENT_ID:-}" ] && [ -n "${PNAP_CLIENT_SECRET:-}" ] \
    || metal_die "PNAP_CLIENT_ID / PNAP_CLIENT_SECRET must be set (PhoenixNAP portal -> API Credentials)"
}

require_tools() {
  local t
  for t in "$@"; do
    command -v "$t" >/dev/null || metal_die "missing tool: $t"
  done
}

# _pnap_curl <METHOD> <url> [json-body] — one HTTP attempt. Prints the
# response body, then the HTTP status code on the final line. The selftest
# overrides this function; nothing else may call curl against the API.
_pnap_curl() {
  local method="$1" url="$2" body="${3:-}"
  local args=(-sS -X "$method" \
    -H "Authorization: Bearer $PNAP_TOKEN" \
    -H "Content-Type: application/json" \
    -w $'\n%{http_code}')
  [ -n "$body" ] && args+=(-d "$body")
  curl "${args[@]}" "$url"
}

# pnap_token — OAuth2 client-credentials grant; sets PNAP_TOKEN (idempotent).
pnap_token() {
  [ -n "${PNAP_TOKEN:-}" ] && return 0
  require_env
  local resp
  resp=$(curl -sS -X POST "$PNAP_AUTH_URL" \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    -d "grant_type=client_credentials&client_id=${PNAP_CLIENT_ID}&client_secret=${PNAP_CLIENT_SECRET}") \
    || metal_die "token request to $PNAP_AUTH_URL failed"
  PNAP_TOKEN=$(jq -er '.access_token' <<<"$resp" 2>/dev/null) \
    || metal_die "no access_token in auth response (check credentials)"
}

# pnap_api <METHOD> <path> [json-body] — authed call, bounded retry on
# 429/5xx (backoff attempt*5s; METAL_RETRY_SLEEP overrides for tests).
# 401/403 die immediately with a credentials hint. Prints the body.
pnap_api() {
  local method="$1" path="$2" body="${3:-}"
  pnap_token
  local attempt out code
  for attempt in 1 2 3 4 5; do
    out=$(_pnap_curl "$method" "$PNAP_API_BASE$path" "$body")
    code=${out##*$'\n'}
    out=${out%$'\n'*}
    case "$code" in
      2??) printf '%s\n' "$out"; return 0 ;;
      401|403) metal_die "API $code on $method $path — check PNAP_CLIENT_ID/PNAP_CLIENT_SECRET" ;;
      429|5??) sleep "${METAL_RETRY_SLEEP:-$((attempt * 5))}" ;;
      *) echo "metal: API $code on $method $path: $out" >&2; return 1 ;;
    esac
  done
  echo "metal: API still failing ($code) after 5 attempts on $method $path" >&2
  return 1
}

# Verified against the BMC OpenAPI definition on 2026-07-10 (Task 2 recon):
# the create-server `os` enum offers debian/bullseye, debian/bookworm and
# debian/trixie (newest), so Debian is used per spec preference; bookworm
# chosen as the current well-tested stable (trixie is newer but only ~1yr
# out; override via METAL_OS if trixie is preferred). Default SSH login
# user is documented per-OS in the phoenixNAP KB (not the OpenAPI spec
# itself): Debian servers use `debian`, not `root` (root is only
# documented for ESXi/Proxmox) — source:
# https://phoenixnap.com/kb/bmc-remote-console.
metal_default_os() { echo "${METAL_OS:-debian/bookworm}"; }
metal_default_ssh_user() { echo "${METAL_SSH_USER:-debian}"; }
