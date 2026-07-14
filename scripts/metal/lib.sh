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
    --connect-timeout 10 --max-time 120 \
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
    --connect-timeout 10 --max-time 120 \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    -d "grant_type=client_credentials&client_id=${PNAP_CLIENT_ID}&client_secret=${PNAP_CLIENT_SECRET}") \
    || metal_die "token request to $PNAP_AUTH_URL failed"
  PNAP_TOKEN=$(jq -er '.access_token' <<<"$resp" 2>/dev/null) \
    || metal_die "no access_token in auth response (check credentials)"
}

# pnap_api <METHOD> <path> [json-body] — authed call, bounded retry on
# 429/5xx (backoff attempt*5s; METAL_RETRY_SLEEP overrides for tests).
# 401/403 die immediately with a credentials hint. Prints the body.
# METAL_NO_RETRY=1: a 429/5xx is NOT retried (the caller's request is
# non-idempotent — e.g. POST /bmc/v1/servers — and a blind retry after a
# real create risks provisioning a second, orphaned, billed server).
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
      429|5??)
        if [ "${METAL_NO_RETRY:-0}" = 1 ]; then
          echo "metal: API $code on $method $path (non-idempotent, not retrying — check for a stray server: mise run metal-gc): $out" >&2
          return 1
        fi
        sleep "${METAL_RETRY_SLEEP:-$((attempt * 5))}" ;;
      409)
        # "The resource is in an incompatible state": PhoenixNAP refuses to
        # modify a server whose network is still provisioning — precisely the
        # state a FAILED provision leaves behind, so the 409 lands on the
        # teardown path when a stray server is most likely. It clears on its
        # own once the network settles. DELETE is idempotent, so retrying is
        # safe; this is the cost-leak backstop and giving up here orphans a
        # billed server. Every other method keeps the fail-fast behaviour —
        # a 409 on create/update is a real conflict, not a transient one.
        if [ "$method" != DELETE ]; then
          echo "metal: API $code on $method $path: $out" >&2; return 1
        fi
        echo "metal: API 409 on DELETE $path (network still provisioning?) — retrying" >&2
        sleep "${METAL_RETRY_SLEEP:-30}" ;;
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

# metal_devpod_source <git-remote-url> <commit-sha> — the workspace source
# string for `devpod up`. devpod clones this ON the box and pins the exact
# commit through its "@sha256:<hash>" delimiter (git.CommitDelimiter). The
# remote must already carry a scheme devpod's positional git.NormalizeRepository
# recognizes (ssh://, git@, http(s)://, file://). Anything else — notably a bare
# "git:" prefix, which is a --source flag convenience that NormalizeRepository
# does NOT strip — gets "https://" blindly prepended, yielding a mangled
# "https://git:https://…" clone URL that dies with `git clone` exit 128 only
# once the billed box is already up. Guard it here so a bad remote fails local
# preflight, before the meter starts.
metal_devpod_source() {
  local remote="$1" sha="$2"
  case "$remote" in
    ssh://*|git@*|http://*|https://*|file://*) : ;;
    *) metal_die "git remote '$remote' has no devpod-recognized scheme (ssh://, git@, http(s)://, file://) — devpod would prepend https:// and mangle the clone URL" ;;
  esac
  printf '%s@sha256:%s\n' "$remote" "$sha"
}

# features_table — path to the curated ISA table (override for tests).
features_table() {
  echo "${METAL_FEATURES_TABLE:-$(dirname "${BASH_SOURCE[0]}")/cpu-features.json}"
}

# check_features_table — integrity: every entry has cpu_model / a real
# vendor_id / physical_cores >= 1 / flags drawn from flag_vocabulary, and
# any avx512f entry enumerates the bw/cd/dq/vl sub-features explicitly
# (kernel dispatch will care exactly which subset exists).
check_features_table() {
  jq -e '
    .flag_vocabulary as $v
    | [ .types | to_entries[]
        | select(
            ((.value.cpu_model // "") == "")
            or ((.value.vendor // "") | IN("GenuineIntel", "AuthenticAMD") | not)
            or ((.value.physical_cores // 0) < 1)
            or (((.value.flags // []) - $v) | length > 0)
            or (((.value.flags // []) | index("avx512f")) != null
                and ((["avx512bw", "avx512cd", "avx512dq", "avx512vl"] - .value.flags) | length > 0))
          )
        | .key ]
    | if length == 0 then true else error("bad entries: \(join(","))") end
  ' "$(features_table)" >/dev/null
}

# gc_candidates <servers.json> — TSV (id, type, hostname, provisionedOn)
# for servers whose description EQUALS the tag. Equality, never contains:
# a substring match against someone's "not-inferno-metal-related" box would
# delete production hardware.
gc_candidates() {
  jq -r --arg tag "$METAL_TAG" \
    '.[] | select(.description == $tag)
         | [.id, .type, .hostname, (.provisionedOn // "-")] | @tsv' "$1"
}

# catalog_join <products.json> <availability.json> <features.json> — TSV:
# type, cpu_model, vendor, cores, flags(csv), $/hr, in-stock locations.
# Types missing from the features table print UNMAPPED — visible on
# purpose: that is the prompt to extend cpu-features.json.
catalog_join() {
  jq -r --slurpfile av "$2" --slurpfile ft "$3" '
    ($av[0] | map({key: .productCode,
                   value: ([.locationAvailabilityDetails[]?
                            | select(.minQuantityAvailable == true) | .location]
                           | join(","))})
             | from_entries) as $stock
    | $ft[0].types as $t
    | .[] | select(.productCategory == "SERVER")
    | [ .productCode,
        ($t[.productCode].cpu_model // "UNMAPPED"),
        ($t[.productCode].vendor // "-"),
        ($t[.productCode].physical_cores // "-"),
        (($t[.productCode].flags // []) | join(",")),
        (([.plans[]? | select(.pricingModel == "HOURLY") | .price] | first) // "-"),
        (($stock[.productCode] // "") | if . == "" then "-" else . end) ]
    | @tsv' "$1"
}

# metal_provision <type> <os> <location> <ssh-pub-path> <hostname>
#   Prints the new server id. Shared by run.sh and probe.sh so the one
#   non-idempotent call in the whole tool has exactly one implementation.
metal_provision() {
  local type="$1" os="$2" location="$3" ssh_key="$4" hostname="$5" body
  body=$(jq -n --arg h "$hostname" --arg t "$type" --arg os "$os" \
    --arg loc "$location" --arg tag "$METAL_TAG" --arg key "$(cat "$ssh_key")" \
    '{hostname: $h, description: $tag, os: $os, type: $t, location: $loc, sshKeys: [$key]}')
  # POST /bmc/v1/servers is not idempotent: a blind retry after a 429/5xx that
  # actually succeeded server-side would provision a second, orphaned, billed
  # server (only the retry's id gets tracked/torn down).
  METAL_NO_RETRY=1 pnap_api POST /bmc/v1/servers "$body" | jq -er '.id'
}

# metal_wait_ready <server-id> <ssh-probe-log> — prints the IP once the box is
#   powered on and answering ssh. Non-zero on error state or timeout; the
#   caller's EXIT trap is what deletes the server.
metal_wait_ready() {
  local id="$1" log="$2"
  local deadline=$(( $(date +%s) + ${METAL_PROVISION_TIMEOUT:-1800} ))
  local s status ip lastline
  while [ "$(date +%s)" -lt "$deadline" ]; do
    s=$(pnap_api GET "/bmc/v1/servers/$id")
    status=$(jq -r '.status' <<<"$s")
    ip=$(jq -r '.publicIpAddresses[0] // empty' <<<"$s")
    if [ "$status" = "error" ]; then
      echo "metal: server entered error state" >&2; return 1
    fi
    if [ "$status" = "powered-on" ] && [ -n "$ip" ]; then
      # PhoenixNAP recycles public IPs across servers, so this IP may carry a
      # stale known_hosts entry from an earlier box. accept-new only adds a
      # *new* key — against a mismatched stale one it fails "Host key
      # verification failed" and never recovers. Purge the old entry first,
      # then accept-new records this host's real key (devpod's own plain `ssh`
      # later depends on that entry existing and being correct).
      ssh-keygen -R "$ip" >/dev/null 2>&1 || true
      if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes \
             "$(metal_default_ssh_user)@$ip" true 2>>"$log"; then
        echo "$ip"; return 0
      fi
      lastline=$(tail -n1 "$log" 2>/dev/null)
      echo "metal: waiting ($status, ssh: ${lastline:-no output yet})..." >&2
    else
      echo "metal: waiting ($status)..." >&2
    fi
    sleep "${METAL_POLL_SLEEP:-20}"
  done
  echo "metal: not ready after ${METAL_PROVISION_TIMEOUT:-1800}s — deleting via trap" >&2
  return 1
}

# metal_probe_entry <type> <cpuinfo> <lscpu-p> — the cpu-features.json entry a
#   box actually justifies. Pure: reads two captured files, no network, so the
#   parsing is testable offline. `source` is deliberately left as a TODO — only
#   a human can cite a vendor spec sheet, and AGENTS.md requires one.
metal_probe_entry() {
  local type="$1" cpuinfo="$2" lscpu_p="$3"
  local model vendor cores present=()
  model=$(awk -F': *' '/^model name/ { print $2; exit }' "$cpuinfo")
  vendor=$(awk -F': *' '/^vendor_id/ { print $2; exit }' "$cpuinfo")
  # Unique (core,socket) pairs — physical cores, not SMT siblings.
  cores=$(grep -v '^#' "$lscpu_p" | cut -d, -f1,2 | sort -u | grep -c .)
  local flags
  flags=$(awk -F': *' '/^flags/ { print $2; exit }' "$cpuinfo")
  local f
  for f in $(jq -r '.flag_vocabulary[]' "$(features_table)"); do
    case " $flags " in *" $f "*) present+=("$f") ;; esac
  done
  jq -n --arg t "$type" --arg m "$model" --arg v "$vendor" \
        --argjson c "${cores:-0}" --args '
    { ($t): { cpu_model: $m, vendor: $v, physical_cores: $c,
              flags: $ARGS.positional,
              source: "TODO: vendor spec sheet for \($m)" } }' "${present[@]}"
}
