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

# --- pnap_api: METAL_NO_RETRY=1 fails a 503 on the first attempt, no retry -
attempts=$(mktemp); echo 0 > "$attempts"
if out=$(
  _pnap_curl() {
    local n; n=$(cat "$attempts"); echo $((n + 1)) > "$attempts"
    printf 'nope\n503\n'
  }
  PNAP_TOKEN=test METAL_NO_RETRY=1 METAL_RETRY_SLEEP=0 pnap_api POST /bmc/v1/servers '{}' 2>/dev/null
); then fail "METAL_NO_RETRY=1 with a 503 should fail (got '$out')"; fi
expect "no-retry attempt count" "$(cat "$attempts")" "1"
rm -f "$attempts"

# --- require_env ----------------------------------------------------------
if (unset PNAP_CLIENT_ID PNAP_CLIENT_SECRET 2>/dev/null; require_env 2>/dev/null); then
  fail "require_env should fail without credentials"
fi

# --- cpu-features.json integrity ------------------------------------------
check_features_table || fail "shipped cpu-features.json must pass its own check"
badtable=$(mktemp)
cat > "$badtable" <<'EOF'
{"schema": 1,
 "flag_vocabulary": ["avx2", "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"],
 "types": {"x.bad": {"cpu_model": "Fake 9000", "vendor": "AuthenticAMD",
                     "physical_cores": 8, "flags": ["avx2", "axv512f"], "source": "test"}}}
EOF
if METAL_FEATURES_TABLE="$badtable" check_features_table 2>/dev/null; then
  fail "typo'd flag (axv512f) must fail the integrity check"
fi
cat > "$badtable" <<'EOF'
{"schema": 1,
 "flag_vocabulary": ["avx2", "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"],
 "types": {"x.bad": {"cpu_model": "Fake 9000", "vendor": "GenuineIntel",
                     "physical_cores": 8, "flags": ["avx2", "avx512f"], "source": "test"}}}
EOF
if METAL_FEATURES_TABLE="$badtable" check_features_table 2>/dev/null; then
  fail "avx512f without sub-feature enumeration must fail the integrity check"
fi
rm -f "$badtable"

# --- host-prep flag verification (fixture /proc, setup skipped) -----------
hp() { # <cpuinfo-fixture> <flags-csv> <vendor> — runs host-prep in test mode
  METAL_PROC_ROOT="$HERE/fixtures/$1.proc" METAL_SKIP_SETUP=1 \
    sh "$HERE/host-prep.sh" "$2" "avx2,avx512f,avx512bw,avx512cd,avx512dq,avx512vl" "$3"
}
mkdir -p "$HERE/fixtures/cpuinfo-match.proc" "$HERE/fixtures/cpuinfo-drift.proc"
cp "$HERE/fixtures/cpuinfo-match.txt" "$HERE/fixtures/cpuinfo-match.proc/cpuinfo"
cp "$HERE/fixtures/cpuinfo-drift.txt" "$HERE/fixtures/cpuinfo-drift.proc/cpuinfo"
hp cpuinfo-match "avx2" "AuthenticAMD" >/dev/null || fail "matching flags should pass host-prep"
rc=0; hp cpuinfo-drift "avx2" "AuthenticAMD" >/dev/null 2>&1 || rc=$?
expect "unexpected-flag drift exit code" "$rc" "4"          # box has avx512f, table omits it
rc=0; hp cpuinfo-match "avx2,avx512f" "AuthenticAMD" >/dev/null 2>&1 || rc=$?
expect "missing-flag drift exit code" "$rc" "4"             # table promises avx512f, box lacks it
rc=0; hp cpuinfo-match "avx2" "GenuineIntel" >/dev/null 2>&1 || rc=$?
expect "vendor drift exit code" "$rc" "4"
rm -rf "$HERE/fixtures/cpuinfo-match.proc" "$HERE/fixtures/cpuinfo-drift.proc"

# --- catalog_join -----------------------------------------------------------
# Capture the full join, then take the first line by parameter expansion —
# `catalog_join | head -1` SIGPIPEs jq (141) under `set -o pipefail` once the
# fixture is large enough that jq is still streaming when head closes the pipe.
rows=$(catalog_join "$HERE/fixtures/products.json" "$HERE/fixtures/availability.json" "$(features_table)")
row=${rows%%$'\n'*}
[ -n "$row" ] || fail "catalog_join produced no rows"
expect "catalog_join column count" "$(printf '%s' "$row" | awk -F'\t' '{print NF}')" "7"
# Every mapped type's flags column must be non-empty; UNMAPPED rows say so.
catalog_join "$HERE/fixtures/products.json" "$HERE/fixtures/availability.json" "$(features_table)" \
  | awk -F'\t' '$2 != "UNMAPPED" && $5 == "" { exit 1 }' \
  || fail "mapped catalog row with empty flags column"

# --- gc_candidates: exact-tag match only ------------------------------------
gc_out=$(gc_candidates "$HERE/fixtures/servers.json")
expect "gc finds the tagged server" "$(echo "$gc_out" | wc -l)" "1"
expect "gc picks the right id" "$(echo "$gc_out" | cut -f1)" "aaa-111"
case "$gc_out" in
  *bbb-222*|*ccc-333*) fail "gc must never match untagged/substring-tagged servers" ;;
esac

# --- run.sh arg parsing (METAL_PARSE_ONLY short-circuits before preflight) --
parse() { METAL_PARSE_ONLY=1 bash "$HERE/run.sh" "$@"; }
expect "parse basic" \
  "$(parse d3.m5.xlarge --yes -- mise run lint)" \
  "type=d3.m5.xlarge yes=1 keep=0 reuse= workload=mise run lint"
expect "parse keep+reuse" \
  "$(parse d3.m5.xlarge --keep --reuse aaa-111 -- echo hi)" \
  "type=d3.m5.xlarge yes=0 keep=1 reuse=aaa-111 workload=echo hi"
if parse d3.m5.xlarge --yes 2>/dev/null; then fail "run.sh without a workload must fail"; fi
if parse d3.m5.xlarge --bogus -- echo hi 2>/dev/null; then fail "unknown flag must fail"; fi

echo "metal lib-selftest: OK"
