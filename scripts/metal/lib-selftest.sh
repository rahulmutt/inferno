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

# --- pnap_api: a 409 on DELETE is retried (network still provisioning) -----
# The teardown path is where 409 actually shows up: a failed provision leaves
# a server whose network is mid-flight, and giving up there orphans a billed
# box. Idempotent, so retrying is safe.
attempts=$(mktemp); echo 0 > "$attempts"
_pnap_curl() {
  local n; n=$(cat "$attempts"); echo $((n + 1)) > "$attempts"
  if [ "$n" -lt 2 ]; then printf 'in-progress\n409\n'; else printf '{}\n200\n'; fi
}
PNAP_TOKEN=test METAL_RETRY_SLEEP=0 pnap_api DELETE /bmc/v1/servers/x >/dev/null 2>&1 \
  || fail "409 on DELETE should retry to success"
expect "409 DELETE retried" "$(cat "$attempts")" "3"
rm -f "$attempts"

# --- pnap_api: an unending 409 on DELETE gives up after 5, doesn't hang ----
attempts=$(mktemp); echo 0 > "$attempts"
if out=$(
  _pnap_curl() {
    local n; n=$(cat "$attempts"); echo $((n + 1)) > "$attempts"
    printf 'in-progress\n409\n'
  }
  PNAP_TOKEN=test METAL_RETRY_SLEEP=0 pnap_api DELETE /bmc/v1/servers/x 2>/dev/null
); then fail "unending 409 on DELETE should fail"; fi
expect "409 DELETE bounded" "$(cat "$attempts")" "5"
rm -f "$attempts"

# --- pnap_api: a 409 on a NON-DELETE is a real conflict — fail fast --------
attempts=$(mktemp); echo 0 > "$attempts"
if out=$(
  _pnap_curl() {
    local n; n=$(cat "$attempts"); echo $((n + 1)) > "$attempts"
    printf 'conflict\n409\n'
  }
  PNAP_TOKEN=test METAL_RETRY_SLEEP=0 pnap_api POST /bmc/v1/servers '{}' 2>/dev/null
); then fail "409 on POST should fail"; fi
expect "409 POST not retried" "$(cat "$attempts")" "1"
rm -f "$attempts"

# --- metal_probe_entry: what the box says, not what the API claims ---------
# Fixture is 2 sockets x 2 cores with SMT siblings duplicated, and carries
# flags both inside the vocabulary (avx512_vnni) and outside it (ht,
# arch_capabilities) — the entry must count physical cores, not threads, and
# must never invent a flag the vocabulary doesn't know.
entry=$(metal_probe_entry x.probe "$HERE/fixtures/probe-cpuinfo.txt" "$HERE/fixtures/probe-lscpu.txt")
expect "probe reads the model name" \
  "$(jq -r '.["x.probe"].cpu_model' <<<"$entry")" \
  "Intel(R) Xeon(R) Gold 9999X CPU @ 2.40GHz"
expect "probe reads the vendor" "$(jq -r '.["x.probe"].vendor' <<<"$entry")" "GenuineIntel"
expect "probe counts physical cores, not SMT siblings" \
  "$(jq -r '.["x.probe"].physical_cores' <<<"$entry")" "4"
expect "probe keeps in-vocabulary flags" \
  "$(jq -r '.["x.probe"].flags | join(",")' <<<"$entry")" \
  "sse4_2,avx,avx2,fma,f16c,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512_vnni"
case "$(jq -r '.["x.probe"].flags | join(",")' <<<"$entry")" in
  *ht*|*arch_capabilities*) fail "probe leaked a flag outside the vocabulary" ;;
esac
case "$(jq -r '.["x.probe"].source' <<<"$entry")" in
  TODO*) ;; *) fail "probe must not fabricate a source — only a human can cite a spec sheet" ;;
esac
# The emitted entry must satisfy the table's own integrity check.
probetable=$(mktemp)
jq -n --slurpfile e <(printf '%s' "$entry") \
   --slurpfile t "$(features_table)" \
   '{schema: 1, flag_vocabulary: $t[0].flag_vocabulary, types: $e[0]}' > "$probetable"
METAL_FEATURES_TABLE="$probetable" check_features_table \
  || fail "a probed entry must pass check_features_table"
rm -f "$probetable"

# --- metal_provision: the one call that can double-bill --------------------
attempts=$(mktemp); echo 0 > "$attempts"
postbody=$(mktemp)
_pnap_curl() {
  local n; n=$(cat "$attempts"); echo $((n + 1)) > "$attempts"
  printf '%s' "$3" > "$postbody"
  printf 'nope\n503\n'
}
key=$(mktemp); echo "ssh-ed25519 AAAA test" > "$key"
if (PNAP_TOKEN=test METAL_RETRY_SLEEP=0 metal_provision t1.small ubuntu PHX "$key" hostx 2>/dev/null); then
  fail "metal_provision should fail on a 503"
fi
expect "provision never blind-retries a POST (would double-bill)" "$(cat "$attempts")" "1"
expect "provision tags the server for gc" "$(jq -r '.description' "$postbody")" "$METAL_TAG"
expect "provision sends the type" "$(jq -r '.type' "$postbody")" "t1.small"
expect "provision sends the ssh key" "$(jq -r '.sshKeys[0]' "$postbody")" "ssh-ed25519 AAAA test"
rm -f "$attempts" "$postbody" "$key"

# --- metal_wait_ready: an error state aborts rather than polling to timeout -
_pnap_curl() { printf '{"status":"error","publicIpAddresses":[]}\n200\n'; }
if (PNAP_TOKEN=test metal_wait_ready srv-1 /dev/null 2>/dev/null); then
  fail "metal_wait_ready must fail fast on an error state"
fi

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

# --- probe.sh arg parsing (METAL_PARSE_ONLY short-circuits before creds) ----
probe_parse() { METAL_PARSE_ONLY=1 bash "$HERE/probe.sh" "$@"; }
expect "probe parse basic" \
  "$(probe_parse d3.c1.large --yes)" \
  "type=d3.c1.large yes=1 os=$(metal_default_os) location=PHX"
expect "probe parse location" \
  "$(probe_parse s5.x6.c9.medium --yes --location NLD)" \
  "type=s5.x6.c9.medium yes=1 os=$(metal_default_os) location=NLD"
if probe_parse d3.c1.large --keep 2>/dev/null; then
  fail "probe has no --keep on purpose; it must reject it"
fi

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
# Regression for the 2026-07-11 payload run: a lost backslash continuation
# left a literal newline inside the quoted workload, so the box ran it as two
# statements (a bare `mise run`, then an orphan command — exit 127) after the
# meter had already paid for provisioning + devpod up.
if parse d3.m5.xlarge --yes -- $'mise run\nverify-quiet-hw' 2>/dev/null; then
  fail "a workload containing a literal newline must fail preflight"
fi

# --- metal_devpod_source: a devpod-clonable workspace source ----------------
# Regression for the mangled clone URL that failed the live E2E smoke with
# `git clone` exit 128: a "git:"-prefixed remote (a --source flag convenience
# that devpod's positional git.NormalizeRepository does NOT strip) had
# "https://" blindly prepended, producing "https://git:https://…@sha256:…".
# The source must pin the commit and start with a devpod-recognized scheme.
expect "devpod source pins the commit" \
  "$(metal_devpod_source https://github.com/o/r.git deadbeef)" \
  "https://github.com/o/r.git@sha256:deadbeef"
if (metal_devpod_source "git:https://github.com/o/r.git" deadbeef) 2>/dev/null; then
  fail "a 'git:'-prefixed remote must be rejected — devpod would mangle it into https://git:https://…"
fi

echo "metal lib-selftest: OK"
