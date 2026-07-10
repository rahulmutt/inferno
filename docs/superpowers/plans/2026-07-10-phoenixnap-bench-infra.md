# PhoenixNAP Bare-Metal Bench Infra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One-shot tooling that provisions a PhoenixNAP bare-metal server, verifies its CPU features against a curated ISA table, runs any workload inside the existing devpod/devenv environment, collects results locally, and deprovisions — plus a catalog and a stray-server GC.

**Architecture:** Bash scripts under `scripts/metal/` (mirroring `scripts/quiet-hw/` conventions: shared `lib.sh`, offline `lib-selftest.sh`, small single-purpose scripts) wrapping the PhoenixNAP BMC REST API via curl/jq, exposed as mise tasks `metal`, `metal-catalog`, `metal-gc`. The environment on the box is devpod's SSH provider reusing `.devcontainer/devcontainer.json`. Spec: [docs/superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md](../specs/2026-07-10-phoenixnap-bench-infra-design.md).

**Tech Stack:** bash (`set -euo pipefail`), curl, jq, ssh, tar, devpod, mise tasks. No Rust changes, no new languages.

## Global Constraints

- Every bash script starts `#!/usr/bin/env bash` + `set -euo pipefail`, except `host-prep.sh` which is portable `#!/bin/sh` + `set -eu` (it runs on a distro we don't control).
- The server tag is the exact string `inferno-metal` in the server `description` field — `gc.sh` filters on equality with it; never a substring match.
- Credentials only via env vars `PNAP_CLIENT_ID` / `PNAP_CLIENT_SECRET`. Never written to any file in-repo. Fixtures must be sanitized (no ids, tokens, account data).
- Scripts never write to `docs/` (quiet-hw discipline). Results go to `target/metal/<type>-<timestamp>/` (already gitignored via `target/`).
- No CI integration: nothing under `.github/` changes; selftests run locally and are wired into `run.sh` like `verify.sh` runs its selftests.
- Config env knobs, all with defaults: `METAL_OS`, `METAL_LOCATION` (default `PHX`), `METAL_SSH_USER`, `METAL_PROVISION_TIMEOUT` (default `1800`), `METAL_RETRY_SLEEP` (test hook), `METAL_PROC_ROOT` (test hook, mirrors `QHW_PROC_ROOT`), `METAL_SKIP_SETUP` (test hook), `METAL_PARSE_ONLY` (test hook), `PNAP_AUTH_URL` / `PNAP_API_BASE` (test hooks).
- API knowledge encoded below (auth URL, endpoint paths, JSON shapes) is best-current-knowledge; **Task 2 verifies it against the live API and fixes anything that differs.** Tasks 3–9 build on the *fixtures*, so a Task 2 correction propagates by re-running the selftest.
- Commit style: `feat(metal): ...` / `test(metal): ...` / `docs(metal): ...`.
- Adding tools to `mise.toml [tools]` (Task 8 adds devpod) invalidates the GitHub Actions tool cache — one ~10 min CI run, then self-heals (see the CAUTION comment in `mise.toml`). Mention this in the PR description.
- Before pushing: `mise run lint` (memory: CI runs clippy `-D warnings`; no Rust is touched here but the check is cheap).

---

### Task 1: `lib.sh` core — auth, API wrapper with retries, selftest harness

**Files:**
- Create: `scripts/metal/lib.sh`
- Create: `scripts/metal/lib-selftest.sh`

**Interfaces:**
- Produces: `METAL_TAG` (string const), `metal_die <msg...>` (stderr + exit 2), `require_env`, `require_tools <t...>`, `pnap_token` (sets `PNAP_TOKEN`), `pnap_api <METHOD> <path> [json-body]` (prints response body; retries 429/5xx ×5; dies on 401/403), `_pnap_curl <METHOD> <url> [body]` (one HTTP attempt, prints body then status code on the last line — **the selftest seam**: tests override this function).
- Consumes: nothing.

- [ ] **Step 1: Write the failing selftest**

Create `scripts/metal/lib-selftest.sh`:

```bash
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `bash scripts/metal/lib-selftest.sh`
Expected: FAIL — `lib.sh: No such file or directory`.

- [ ] **Step 3: Write `lib.sh`**

Create `scripts/metal/lib.sh`:

```bash
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

# PROVISIONAL defaults — Task 2's live recon verifies both against the BMC
# OpenAPI definition (Debian preferred per spec) and updates this comment.
metal_default_os() { echo "${METAL_OS:-ubuntu/jammy}"; }
metal_default_ssh_user() { echo "${METAL_SSH_USER:-root}"; }
```

- [ ] **Step 4: Run the selftest to verify it passes**

Run: `bash scripts/metal/lib-selftest.sh`
Expected: `metal lib-selftest: OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/metal/lib.sh scripts/metal/lib-selftest.sh
git commit -m "feat(metal): BMC API wrapper with bounded retries + offline selftest"
```

---

### Task 2: Live API recon — record real fixtures, resolve the spec's open items

**Requires `PNAP_CLIENT_ID`/`PNAP_CLIENT_SECRET` in the environment.** If credentials are not available when this task comes up, create hand-written fixtures with the shapes shown in Step 3's sanitizer (so Tasks 3–9 can proceed) and leave this task's checkboxes unchecked — it MUST be completed before Task 10.

**Files:**
- Create: `scripts/metal/record-fixtures.sh`
- Create: `scripts/metal/fixtures/products.json`
- Create: `scripts/metal/fixtures/availability.json`
- Modify: `scripts/metal/lib.sh` (set the verified `METAL_OS` / `METAL_SSH_USER` defaults)
- Modify: `docs/superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md` (§Amendments)

**Interfaces:**
- Consumes: `pnap_api`, `require_env`, `require_tools` from Task 1.
- Produces: `fixtures/products.json` — array of `{productCode, productCategory, plans: [{pricingModel, price}], metadata: {cpu, cpuCount, coresPerCpu, cpuFrequency}}`; `fixtures/availability.json` — array of `{productCode, locationAvailabilityDetails: [{location, minQuantityAvailable}]}`; `metal_default_os()` and `metal_default_ssh_user()` in `lib.sh`.

- [ ] **Step 1: Write `record-fixtures.sh`**

```bash
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
```

- [ ] **Step 2: Run it against the live API**

Run: `bash scripts/metal/record-fixtures.sh`
Expected: `recorded N products; re-run lib-selftest.sh` with N ≥ 10.

If the auth URL, endpoint paths, or JSON field names differ from what Task 1 encoded (this is the verification the spec's "open items" call for): fix `lib.sh`/`record-fixtures.sh` to match reality, re-run `bash scripts/metal/lib-selftest.sh`, and note the corrections in the spec amendment (Step 5).

- [ ] **Step 3: Inspect the fixtures and verify sanitization**

Run: `jq '.[0]' scripts/metal/fixtures/products.json && gitleaks dir scripts/metal/fixtures/`
Expected: one product object with exactly the fields listed in Interfaces above; gitleaks reports no leaks. If any account-specific field survived, tighten the jq sanitizer and re-record.

- [ ] **Step 4: Resolve OS + SSH-user defaults from the BMC OpenAPI definition**

PhoenixNAP publishes the BMC OpenAPI spec; fetch it and read the create-server `os` enum and the documented default login user:

Run: `curl -sSL https://developers.phoenixnap.com/assets/bmc-api.yaml -o /tmp/claude-1000/-workspace/f7501d0f-5cb3-4487-b4a0-6ea45ad00c94/scratchpad/bmc-api.yaml || true`

If that URL 404s, find the current definition link on https://developers.phoenixnap.com/apis (the BMC API page links its OpenAPI/Swagger file) and fetch that instead. Then grep the file for the `os` schema enum (search for `ubuntu/`) and for the SSH login user documentation (search for `root` / `whoami` in the create-server description).

Task 1 shipped `metal_default_os` (provisionally `ubuntu/jammy`) and `metal_default_ssh_user` (provisionally `root`) in `lib.sh`. Update both to what the definition actually says — Debian (e.g. `debian/bookworm`) preferred if the enum offers it, else the newest Ubuntu LTS in the enum — and replace the `PROVISIONAL defaults` comment with `Verified against the BMC OpenAPI definition on 2026-MM-DD (Task 2 recon): <what the catalog offered>.` using the real date and finding.

- [ ] **Step 5: Record the recon in the spec's Amendments and commit**

Append to the spec's `## Amendments` section: date, confirmation (or correction) of the auth URL/endpoints, the chosen `METAL_OS` default and whether Debian was available, the documented SSH user, and the fixture recording date.

```bash
bash scripts/metal/lib-selftest.sh
git add scripts/metal/record-fixtures.sh scripts/metal/fixtures/ scripts/metal/lib.sh docs/superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md
git commit -m "feat(metal): live API recon — recorded fixtures, verified auth/OS/ssh-user (spec open items)"
```

---

### Task 3: `cpu-features.json` + integrity check + `host-prep.sh` flag verification

**Files:**
- Create: `scripts/metal/cpu-features.json`
- Create: `scripts/metal/host-prep.sh`
- Create: `scripts/metal/fixtures/cpuinfo-match.txt`, `scripts/metal/fixtures/cpuinfo-drift.txt`
- Modify: `scripts/metal/lib.sh` (add `features_table`, `check_features_table`)
- Modify: `scripts/metal/lib-selftest.sh`

**Interfaces:**
- Consumes: fixtures from Task 2 (`products.json` lists the types to map).
- Produces: `features_table()` → path to `cpu-features.json`; `check_features_table()` → exit 0/nonzero; `cpu-features.json` schema `{schema: 1, flag_vocabulary: [...], types: {"<productCode>": {cpu_model, vendor, physical_cores, flags: [...], source}}}`; `host-prep.sh <expected-flags-csv> <vocabulary-csv> <expected-vendor>` run as root on the box — exit 0 OK, **exit 4 = drift**, honors `METAL_PROC_ROOT` and `METAL_SKIP_SETUP=1` for tests.

- [ ] **Step 1: Write the failing selftest cases**

Append to `scripts/metal/lib-selftest.sh` (before the final `echo`):

```bash
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
```

Create the two cpuinfo fixtures. `fixtures/cpuinfo-match.txt` (AVX2-only AMD box):

```
processor	: 0
vendor_id	: AuthenticAMD
model name	: AMD EPYC 7402P 24-Core Processor
flags		: fpu vme de pse msr sse sse2 sse4_1 sse4_2 avx avx2 fma f16c
```

`fixtures/cpuinfo-drift.txt` (same but the box unexpectedly has AVX-512):

```
processor	: 0
vendor_id	: AuthenticAMD
model name	: AMD EPYC 7402P 24-Core Processor
flags		: fpu vme de pse msr sse sse2 sse4_1 sse4_2 avx avx2 fma f16c avx512f avx512bw avx512cd avx512dq avx512vl
```

(Real cpuinfo `flags` lines are much longer; the vocabulary-scoped comparison must ignore everything outside the vocabulary, which these fixtures exercise via the extra `fpu vme de ...` flags.)

- [ ] **Step 2: Run the selftest to verify the new cases fail**

Run: `bash scripts/metal/lib-selftest.sh`
Expected: FAIL — `check_features_table: command not found`.

- [ ] **Step 3: Add table helpers to `lib.sh`**

Append to `scripts/metal/lib.sh`:

```bash
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
```

- [ ] **Step 4: Write `host-prep.sh`**

```sh
#!/bin/sh
# Host prep for a freshly provisioned PhoenixNAP box. Runs as root ON the
# target host over ssh (portable sh — Debian or Ubuntu; never bash-isms).
#   host-prep.sh <expected-flags-csv> <vocabulary-csv> <expected-vendor>
# Exit codes: 0 ok, 4 = CPU vendor/flag drift vs cpu-features.json (the
# caller aborts BEFORE the slow devpod stage; fix the table in a commit —
# there is deliberately no skip flag).
# Test hooks: METAL_PROC_ROOT (fake /proc), METAL_SKIP_SETUP=1 (no docker/
# governor writes), mirroring quiet-hw's QHW_PROC_ROOT convention.
set -eu
EXPECTED="$1"; VOCAB="$2"; VENDOR="$3"
CPUINFO="${METAL_PROC_ROOT:-/proc}/cpuinfo"

# Drift check FIRST — cheapest possible abort on a mislabeled box.
actual_vendor=$(awk -F': *' '/^vendor_id/ { print $2; exit }' "$CPUINFO")
[ "$actual_vendor" = "$VENDOR" ] || {
  echo "VENDOR DRIFT: cpu-features.json says $VENDOR, box says $actual_vendor" >&2
  exit 4
}
flags=$(awk -F': *' '/^flags/ { print $2; exit }' "$CPUINFO")
drift=0
for f in $(echo "$EXPECTED" | tr ',' ' '); do
  case " $flags " in *" $f "*) ;; *)
    echo "MISSING FLAG: $f (table promises it, box lacks it)" >&2; drift=1 ;;
  esac
done
for f in $(echo "$VOCAB" | tr ',' ' '); do
  case ",$EXPECTED," in *",$f,"*) continue ;; esac
  case " $flags " in *" $f "*)
    echo "UNEXPECTED FLAG: $f (box has it, table omits it)" >&2; drift=1 ;;
  esac
done
[ "$drift" = 0 ] || exit 4
awk -F': *' '/^model name/ { print "cpu: " $2; exit }' "$CPUINFO"

if [ "${METAL_SKIP_SETUP:-0}" != 1 ]; then
  command -v docker >/dev/null 2>&1 || {
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq docker.io
  }
  n=0
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -e "$g" ] || continue
    echo performance > "$g" && n=$((n + 1))
  done
  if [ "$n" -gt 0 ]; then echo "governor: performance on $n cpus"
  else echo "governor: no cpufreq interface (leaving as-is)"; fi
fi
echo "host-prep: OK"
```

- [ ] **Step 5: Populate `cpu-features.json` from the recorded catalog**

For every `productCode` in `scripts/metal/fixtures/products.json` with an x86 CPU (skip ARM types — spec §out-of-scope): map `metadata.cpu` to flags using vendor spec sheets (ark.intel.com / amd.com — record the URL in each entry's `source` field). `physical_cores` = `cpuCount * coresPerCpu` from the fixture. Microarchitecture cheat sheet (verify each against its sheet; the on-box check in Task 10+ is the ground-truth backstop):

| CPU family | flags |
|---|---|
| AMD Zen 1–3 (EPYC 7xx1/7xx2/7xx3, Ryzen ≤5xxx) | `sse4_2,avx,avx2,fma,f16c` |
| AMD Zen 4/5 (EPYC 9xx4/9xx5) | above + `avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512_vnni,avx512_bf16` |
| Intel Coffee Lake (E-21xx/E-22xx) | `sse4_2,avx,avx2,fma,f16c` |
| Intel Rocket Lake (E-23xx) | above + `avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512_vnni` |
| Intel Skylake-SP (x1xx Silver/Gold/Platinum) | `sse4_2,avx,avx2,fma,f16c,avx512f,avx512bw,avx512cd,avx512dq,avx512vl` |
| Intel Cascade Lake (x2xx) | Skylake-SP + `avx512_vnni` |
| Intel Ice Lake-SP (x3xx) | Cascade Lake set (plus more not in vocabulary) |
| Intel Sapphire Rapids (x4xx) | Ice Lake + `avx512_bf16,avx512_fp16,amx_tile,amx_int8,amx_bf16` |

File shape (entries below are EXAMPLES — replace with the actual recorded catalog):

```json
{
  "schema": 1,
  "flag_vocabulary": ["sse4_2", "avx", "avx2", "fma", "f16c",
                      "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
                      "avx512_vnni", "avx512_bf16", "avx512_fp16",
                      "amx_tile", "amx_int8", "amx_bf16"],
  "types": {
    "d3.m5.xlarge": {
      "cpu_model": "AMD EPYC 7443P",
      "vendor": "AuthenticAMD",
      "physical_cores": 24,
      "flags": ["sse4_2", "avx", "avx2", "fma", "f16c"],
      "source": "https://www.amd.com/en/products/cpu/amd-epyc-7443p"
    },
    "d3.c2.medium": {
      "cpu_model": "Intel Xeon Gold 6326",
      "vendor": "GenuineIntel",
      "physical_cores": 16,
      "flags": ["sse4_2", "avx", "avx2", "fma", "f16c",
                "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
                "avx512_vnni"],
      "source": "https://ark.intel.com/content/www/us/en/ark/products/215274/intel-xeon-gold-6326-processor.html"
    }
  }
}
```

- [ ] **Step 6: Run the selftest to verify it passes**

Run: `bash scripts/metal/lib-selftest.sh`
Expected: `metal lib-selftest: OK`

- [ ] **Step 7: Commit**

```bash
git add scripts/metal/cpu-features.json scripts/metal/host-prep.sh scripts/metal/lib.sh scripts/metal/lib-selftest.sh scripts/metal/fixtures/cpuinfo-*.txt
git commit -m "feat(metal): curated ISA table + integrity check + on-box drift verification"
```

---

### Task 4: `catalog.sh` — server types × ISA table × availability

**Files:**
- Create: `scripts/metal/catalog.sh`
- Modify: `scripts/metal/lib.sh` (add `catalog_join`)
- Modify: `scripts/metal/lib-selftest.sh`

**Interfaces:**
- Consumes: `pnap_api`, `features_table` (Tasks 1, 3); fixture shapes (Task 2).
- Produces: `catalog_join <products.json> <availability.json> <features.json>` → TSV rows `type, cpu_model, vendor, cores, flags-csv, hourly-usd, in-stock-locations`; unmapped types print `UNMAPPED` in the cpu column (they must be visible, not hidden — that's the prompt to extend the table).

- [ ] **Step 1: Write the failing selftest case**

Append to `scripts/metal/lib-selftest.sh` (before the final `echo`):

```bash
# --- catalog_join -----------------------------------------------------------
row=$(catalog_join "$HERE/fixtures/products.json" "$HERE/fixtures/availability.json" "$(features_table)" | head -1)
[ -n "$row" ] || fail "catalog_join produced no rows"
expect "catalog_join column count" "$(printf '%s' "$row" | awk -F'\t' '{print NF}')" "7"
# Every mapped type's flags column must be non-empty; UNMAPPED rows say so.
catalog_join "$HERE/fixtures/products.json" "$HERE/fixtures/availability.json" "$(features_table)" \
  | awk -F'\t' '$2 != "UNMAPPED" && $5 == "" { exit 1 }' \
  || fail "mapped catalog row with empty flags column"
```

- [ ] **Step 2: Run the selftest to verify it fails**

Run: `bash scripts/metal/lib-selftest.sh`
Expected: FAIL — `catalog_join: command not found`.

- [ ] **Step 3: Add `catalog_join` to `lib.sh`**

```bash
# catalog_join <products.json> <availability.json> <features.json> — TSV:
# type, cpu_model, vendor, cores, flags(csv), $/hr, in-stock locations.
# Types missing from the features table print UNMAPPED — visible on
# purpose: that is the prompt to extend cpu-features.json.
catalog_join() {
  jq -r --slurpfile av "$2" --slurpfile ft "$3" '
    ($av[0] | map({key: .productCode,
                   value: ([.locationAvailabilityDetails[]?
                            | select(.minQuantityAvailable >= 1) | .location]
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
```

- [ ] **Step 4: Run the selftest to verify it passes**

Run: `bash scripts/metal/lib-selftest.sh`
Expected: `metal lib-selftest: OK`

- [ ] **Step 5: Write `catalog.sh`**

```bash
#!/usr/bin/env bash
# metal-catalog: PhoenixNAP server types joined with the curated ISA table
# and live availability. Read-only; needs credentials but never provisions.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"
require_tools curl jq column
check_features_table || metal_die "cpu-features.json failed its integrity check"

products=$(mktemp) avail=$(mktemp)
trap 'rm -f "$products" "$avail"' EXIT
pnap_api GET "/billing/v1/products?productCategory=SERVER" > "$products"
pnap_api GET "/billing/v1/product-availability?productCategory=SERVER&minQuantity=1" > "$avail"
{
  printf 'TYPE\tCPU\tVENDOR\tCORES\tFLAGS\tUSD/HR\tIN-STOCK\n'
  catalog_join "$products" "$avail" "$(features_table)"
} | column -t -s "$(printf '\t')"
```

- [ ] **Step 6: Live check (only if credentials are available)**

Run: `bash scripts/metal/catalog.sh`
Expected: a table with one row per server type; ISA flags filled in for mapped types. If any row says `UNMAPPED`, go back to Task 3 Step 5 and map it (or confirm it's ARM/out-of-scope and leave it visible).

- [ ] **Step 7: Commit**

```bash
git add scripts/metal/catalog.sh scripts/metal/lib.sh scripts/metal/lib-selftest.sh
git commit -m "feat(metal): catalog — server types joined with ISA table and availability"
```

---

### Task 5: `gc.sh` — stray-server backstop

**Files:**
- Create: `scripts/metal/gc.sh`
- Create: `scripts/metal/fixtures/servers.json`
- Modify: `scripts/metal/lib.sh` (add `gc_candidates`)
- Modify: `scripts/metal/lib-selftest.sh`

**Interfaces:**
- Consumes: `pnap_api`, `METAL_TAG` (Task 1).
- Produces: `gc_candidates <servers.json>` → TSV `id, type, hostname, provisionedOn` for servers whose `description` EQUALS `inferno-metal`; `gc.sh [--force]` lists candidates, confirms (skipped by `--force`), deletes each via `DELETE /bmc/v1/servers/{id}`.

- [ ] **Step 1: Write the fixture and the failing selftest case**

Create `scripts/metal/fixtures/servers.json` (hand-written — must cover: tagged, untagged, and the near-miss substring case):

```json
[
  {"id": "aaa-111", "type": "d3.m5.xlarge", "hostname": "inferno-metal-d3-m5-xlarge-20260710T120000Z",
   "description": "inferno-metal", "status": "powered-on", "provisionedOn": "2026-07-10T12:03:00Z",
   "publicIpAddresses": ["203.0.113.10"]},
  {"id": "bbb-222", "type": "s1.c1.medium", "hostname": "prod-db-do-not-touch",
   "description": "production database", "status": "powered-on", "provisionedOn": "2025-01-01T00:00:00Z",
   "publicIpAddresses": ["203.0.113.20"]},
  {"id": "ccc-333", "type": "s1.c1.medium", "hostname": "other-team-box",
   "description": "not-inferno-metal-related", "status": "powered-on", "provisionedOn": "2026-07-01T00:00:00Z",
   "publicIpAddresses": ["203.0.113.30"]}
]
```

Append to `scripts/metal/lib-selftest.sh` (before the final `echo`):

```bash
# --- gc_candidates: exact-tag match only ------------------------------------
gc_out=$(gc_candidates "$HERE/fixtures/servers.json")
expect "gc finds the tagged server" "$(echo "$gc_out" | wc -l)" "1"
expect "gc picks the right id" "$(echo "$gc_out" | cut -f1)" "aaa-111"
case "$gc_out" in
  *bbb-222*|*ccc-333*) fail "gc must never match untagged/substring-tagged servers" ;;
esac
```

- [ ] **Step 2: Run the selftest to verify it fails**

Run: `bash scripts/metal/lib-selftest.sh`
Expected: FAIL — `gc_candidates: command not found`.

- [ ] **Step 3: Add `gc_candidates` to `lib.sh`**

```bash
# gc_candidates <servers.json> — TSV (id, type, hostname, provisionedOn)
# for servers whose description EQUALS the tag. Equality, never contains:
# a substring match against someone's "not-inferno-metal-related" box would
# delete production hardware.
gc_candidates() {
  jq -r --arg tag "$METAL_TAG" \
    '.[] | select(.description == $tag)
         | [.id, .type, .hostname, (.provisionedOn // "-")] | @tsv' "$1"
}
```

- [ ] **Step 4: Run the selftest to verify it passes**

Run: `bash scripts/metal/lib-selftest.sh`
Expected: `metal lib-selftest: OK`

- [ ] **Step 5: Write `gc.sh`**

```bash
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
```

- [ ] **Step 6: Live check (only if credentials are available)**

Run: `bash scripts/metal/gc.sh`
Expected: `metal-gc: no inferno-metal servers running` (nothing provisioned yet).

- [ ] **Step 7: Commit**

```bash
git add scripts/metal/gc.sh scripts/metal/fixtures/servers.json scripts/metal/lib.sh scripts/metal/lib-selftest.sh
git commit -m "feat(metal): gc backstop for stray tagged servers (exact-tag match)"
```

---

### Task 6: `run.sh` part 1 — args, preflight, provision/poll/teardown lifecycle

**Files:**
- Create: `scripts/metal/run.sh`
- Modify: `scripts/metal/lib-selftest.sh`

**Interfaces:**
- Consumes: `pnap_api`, `metal_die`, `require_env`, `require_tools`, `features_table`, `check_features_table`, `metal_default_os`, `metal_default_ssh_user`, `METAL_TAG` (Tasks 1–3).
- Produces: `run.sh <server-type> [--yes] [--keep] [--reuse <id>] [--ssh-key <pub>] [--os <os>] [--location <loc>] [--out-dir <d>] -- <workload command...>`. Exit code = the workload's exit code. `METAL_PARSE_ONLY=1` prints the parsed config and exits 0 (selftest hook). Internal functions Task 7 fills in: `provision`, `wait_ready <id>` (prints IP), `host_prep <ip>`, and the `cleanup` trap. This task leaves the devpod stages as explicit `metal_die "not implemented: ..."` markers that Task 7 replaces.

- [ ] **Step 1: Write the failing selftest cases (arg parsing)**

Append to `scripts/metal/lib-selftest.sh` (before the final `echo`):

```bash
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
```

- [ ] **Step 2: Run the selftest to verify it fails**

Run: `bash scripts/metal/lib-selftest.sh`
Expected: FAIL — `run.sh: No such file or directory`.

- [ ] **Step 3: Write `run.sh` (lifecycle skeleton)**

```bash
#!/usr/bin/env bash
# metal: one-shot bare-metal workload runner (spec:
# docs/superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md).
# Pipeline: preflight -> provision -> host-prep -> devpod up -> run ->
# collect -> deprovision. Ephemeral by default: an EXIT trap deletes the
# server on ANY exit path once provisioning returns an id (--keep holds it,
# --reuse never deletes a box this run didn't create).
# Usage: run.sh <server-type> [--yes] [--keep] [--reuse <server-id>]
#               [--ssh-key <pub>] [--os <os>] [--location <loc>]
#               [--out-dir <dir>] -- <workload command...>
# The workload runs in the devpod workspace inside `devenv shell`; fetch
# models ON the box (e.g. scripts/fetch-qwen-gguf.sh), never upload them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"

TYPE="${1:?usage: run.sh <server-type> [flags] -- <workload...>}"
shift
YES=0 KEEP=0 REUSE="" OUTDIR=""
SSH_KEY="$HOME/.ssh/id_ed25519.pub"
OS="$(metal_default_os)" LOCATION="${METAL_LOCATION:-PHX}"
WORKLOAD=()
while [ $# -gt 0 ]; do
  case "$1" in
    --yes) YES=1 ;;
    --keep) KEEP=1 ;;
    --reuse) shift; REUSE="${1:?--reuse needs a server id}" ;;
    --ssh-key) shift; SSH_KEY="${1:?--ssh-key needs a path}" ;;
    --os) shift; OS="${1:?--os needs a value}" ;;
    --location) shift; LOCATION="${1:?--location needs a value}" ;;
    --out-dir) shift; OUTDIR="${1:?--out-dir needs a value}" ;;
    --) shift; WORKLOAD=("$@"); break ;;
    *) metal_die "unknown arg: $1" ;;
  esac
  shift
done
[ ${#WORKLOAD[@]} -ge 1 ] || metal_die "no workload given (everything after -- runs in the workspace)"

if [ "${METAL_PARSE_ONLY:-0}" = 1 ]; then
  printf 'type=%s yes=%s keep=%s reuse=%s workload=%s\n' \
    "$TYPE" "$YES" "$KEEP" "$REUSE" "${WORKLOAD[*]}"
  exit 0
fi

# --- preflight (local, free) -------------------------------------------
require_tools curl jq ssh tar devpod git
require_env
[ -f "$SSH_KEY" ] || metal_die "ssh public key not found: $SSH_KEY (--ssh-key)"
check_features_table || metal_die "cpu-features.json failed its integrity check"
ENTRY=$(jq -e --arg t "$TYPE" '.types[$t]' "$(features_table)") \
  || metal_die "server type '$TYPE' not in cpu-features.json — add it (with a vendor-sheet source) first"

PRICE=$(pnap_api GET "/billing/v1/products?productCategory=SERVER" \
  | jq -r --arg t "$TYPE" \
      '([.[] | select(.productCode == $t) | .plans[]? | select(.pricingModel == "HOURLY") | .price] | first) // "unknown"')
echo "metal: $TYPE in $LOCATION at \$$PRICE/hr — $(jq -r '.cpu_model' <<<"$ENTRY") ($(jq -r '.flags | join(",")' <<<"$ENTRY"))"
if [ "$YES" != 1 ]; then
  printf 'provision it (the meter starts now)? [y/N] '
  read -r answer
  [ "$answer" = y ] || { echo "metal: aborted before spending money"; exit 0; }
fi

# --- results dir + incremental metadata (exists from here on: even an ---
# --- early abort leaves a record) ----------------------------------------
REPO=$(git rev-parse --show-toplevel)
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
OUT="${OUTDIR:-$REPO/target/metal/${TYPE}-${RUN_ID}}"
mkdir -p "$OUT"
jq -n --arg t "$TYPE" --arg os "$OS" --arg loc "$LOCATION" \
  --arg sha "$(git -C "$REPO" rev-parse HEAD)" \
  --argjson dirty "$([ -n "$(git -C "$REPO" status --porcelain)" ] && echo true || echo false)" \
  --arg started "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{server_type: $t, os: $os, location: $loc, git_sha: $sha, git_dirty: $dirty, started: $started}' \
  > "$OUT/metadata.json"
meta_set() { # <key> <json-value>
  local tmp; tmp=$(mktemp)
  jq --arg k "$1" --argjson v "$2" '.[$k] = $v' "$OUT/metadata.json" > "$tmp" \
    && mv "$tmp" "$OUT/metadata.json"
}

# --- teardown trap --------------------------------------------------------
# cleanup takes the exit code as $1: Task 7 chains another cleanup function
# before it in the trap, which would clobber $? by the time this runs.
SERVER_ID="" SERVER_IP=""
cleanup() {
  local rc="$1"
  if [ -n "$SERVER_ID" ] && [ -z "$REUSE" ]; then
    if [ "$KEEP" = 1 ]; then
      echo "metal: --keep — server $SERVER_ID ($SERVER_IP) stays up. THE METER IS RUNNING."
      echo "metal: rerun with '--reuse $SERVER_ID', or delete via 'mise run metal-gc'."
    else
      echo "metal: deleting server $SERVER_ID"
      pnap_api DELETE "/bmc/v1/servers/$SERVER_ID" >/dev/null \
        || echo "metal: DELETE FAILED — run 'mise run metal-gc' NOW" >&2
    fi
  fi
  echo "metal: results in $OUT (exit $rc)"
  exit "$rc"
}
trap 'cleanup "$?"' EXIT

# --- provision ------------------------------------------------------------
provision() { # prints the new server id
  local hostname="inferno-metal-${TYPE//./-}-$RUN_ID"
  local body
  body=$(jq -n --arg h "$hostname" --arg t "$TYPE" --arg os "$OS" \
    --arg loc "$LOCATION" --arg tag "$METAL_TAG" --arg key "$(cat "$SSH_KEY")" \
    '{hostname: $h, description: $tag, os: $os, type: $t, location: $loc, sshKeys: [$key]}')
  pnap_api POST /bmc/v1/servers "$body" | jq -er '.id'
}

wait_ready() { # <server-id> — prints IP once powered-on + ssh answers
  local deadline=$(( $(date +%s) + ${METAL_PROVISION_TIMEOUT:-1800} ))
  local s status ip
  while [ "$(date +%s)" -lt "$deadline" ]; do
    s=$(pnap_api GET "/bmc/v1/servers/$1")
    status=$(jq -r '.status' <<<"$s")
    ip=$(jq -r '.publicIpAddresses[0] // empty' <<<"$s")
    if [ "$status" = "error" ]; then
      echo "metal: server entered error state" >&2; return 1
    fi
    if [ "$status" = "powered-on" ] && [ -n "$ip" ] \
       && ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes \
              "$(metal_default_ssh_user)@$ip" true 2>/dev/null; then
      echo "$ip"; return 0
    fi
    echo "metal: waiting ($status)..." >&2
    sleep 20
  done
  echo "metal: not ready after ${METAL_PROVISION_TIMEOUT:-1800}s — deleting via trap" >&2
  return 1
}

host_prep() { # <ip> — drift check + docker + governor, BEFORE slow devpod
  local vocab expected vendor
  vocab=$(jq -r '.flag_vocabulary | join(",")' "$(features_table)")
  expected=$(jq -r '.flags | join(",")' <<<"$ENTRY")
  vendor=$(jq -r '.vendor' <<<"$ENTRY")
  # shellcheck disable=SC2029
  if ! ssh -o StrictHostKeyChecking=accept-new "$(metal_default_ssh_user)@$1" \
       'sh -s' "$expected" "$vocab" "$vendor" < "$HERE/host-prep.sh" \
       2>&1 | tee "$OUT/host-prep.log"; then
    metal_die "host-prep failed — if it printed DRIFT lines, fix cpu-features.json in a commit (no override exists on purpose)"
  fi
}

if [ -n "$REUSE" ]; then
  SERVER_ID="$REUSE"
  SERVER_IP=$(pnap_api GET "/bmc/v1/servers/$REUSE" | jq -er '.publicIpAddresses[0]') \
    || metal_die "--reuse $REUSE: could not fetch server IP"
  echo "metal: reusing $SERVER_ID ($SERVER_IP); skipping provision + host-prep"
else
  echo "metal: provisioning $TYPE..."
  SERVER_ID=$(provision)
  meta_set server_id "\"$SERVER_ID\""
  echo "metal: server $SERVER_ID created; waiting for power-on + ssh"
  SERVER_IP=$(wait_ready "$SERVER_ID")
  meta_set server_ip "\"$SERVER_IP\""
  echo "metal: ready at $SERVER_IP; running host-prep"
  host_prep "$SERVER_IP"
fi
meta_set cpu_expected "$(jq -c '.' <<<"$ENTRY")"

metal_die "not implemented: devpod workspace + workload + collect (Task 7)"
```

- [ ] **Step 4: Run the selftest to verify it passes**

Run: `bash scripts/metal/lib-selftest.sh`
Expected: `metal lib-selftest: OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/metal/run.sh scripts/metal/lib-selftest.sh
git commit -m "feat(metal): run.sh lifecycle — preflight, provision/poll, drift-checked host-prep, teardown trap"
```

---

### Task 7: `run.sh` part 2 — devpod workspace, workload execution, collect

**Files:**
- Modify: `scripts/metal/run.sh` (replace the `not implemented` marker)
- Modify: `mise.toml` (add pinned devpod to `[tools]`)

**Interfaces:**
- Consumes: `SERVER_IP`, `OUT`, `meta_set`, `WORKLOAD`, `RUN_ID`, `cleanup` trap from Task 6.
- Produces: the complete pipeline. devpod objects are named `metal-$RUN_ID` (provider) and `inferno-metal-$RUN_ID` (workspace); both are removed in the trap. Results collected via tar streamed over `devpod ssh`. The workload's exit code is the script's exit code.

- [ ] **Step 1: Pin devpod in `mise.toml`**

Add to the `[tools]` section (check `mise ls-remote aqua:loft-sh/devpod | tail -3` for the current version and pin that exact version — `0.6.15` below is the syntax example):

```toml
"aqua:loft-sh/devpod" = "0.6.15"
```

Run: `mise install && devpod version`
Expected: the pinned version prints. (Reminder: this edit invalidates the CI tool cache — one slow CI run; say so in the PR.)

- [ ] **Step 2: Verify the devpod SSH-provider CLI surface**

devpod's CLI has drifted across versions; confirm the exact flags before wiring them in:

Run: `devpod provider add --help && devpod up --help | head -40 && devpod ssh --help | head -20`
Expected: `provider add ssh` accepts `--name` and `-o KEY=VALUE` options (the SSH provider's option is `HOST`); `up` accepts `--provider`, `--id`, `--ide`; `ssh` accepts `--command`. If any flag differs, adapt Step 3's code to the installed version's syntax — the shapes below are for devpod 0.6.x.

- [ ] **Step 3: Replace the Task-6 marker in `run.sh`**

Replace the line `metal_die "not implemented: devpod workspace + workload + collect (Task 7)"` with:

```bash
# --- devpod workspace ------------------------------------------------------
PROVIDER="metal-$RUN_ID"
WORKSPACE="inferno-metal-$RUN_ID"
devpod_cleanup() {
  devpod delete "$WORKSPACE" --force >/dev/null 2>&1 || true
  devpod provider delete "$PROVIDER" >/dev/null 2>&1 || true
}
# Extend the teardown path: devpod objects go first, then the server.
# The script's exit code is captured before devpod_cleanup can clobber $?.
trap 'rc=$?; devpod_cleanup; cleanup "$rc"' EXIT

echo "metal: creating devpod workspace on $SERVER_IP (image pull + devenv — minutes, not seconds)"
devpod provider add ssh --name "$PROVIDER" --use=false \
  -o "HOST=$(metal_default_ssh_user)@$SERVER_IP"
devpod up "$REPO" --provider "$PROVIDER" --id "$WORKSPACE" --ide none \
  2>&1 | tee "$OUT/devpod-up.log"
meta_set devpod_workspace "\"$WORKSPACE\""

# --- workload ---------------------------------------------------------------
# Joined into one string, run in the workspace root inside devenv shell —
# same environment every runbook assumes. %q-quoted so operator quoting
# survives the two ssh hops.
WORKLOAD_STR="${WORKLOAD[*]}"
echo "metal: running: $WORKLOAD_STR"
meta_set workload "$(jq -n --arg w "$WORKLOAD_STR" '$w')"
WORKLOAD_RC=0
devpod ssh "$WORKSPACE" --command \
  "cd /workspace && devenv shell -- bash -c $(printf '%q' "$WORKLOAD_STR")" \
  2>&1 | tee "$OUT/workload.log" || WORKLOAD_RC=$?
meta_set workload_exit "$WORKLOAD_RC"
meta_set finished "\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""

# --- collect (ALWAYS — a failed gate's partial output is the diagnostic) ----
echo "metal: collecting results"
devpod ssh "$WORKSPACE" --command \
  'cd /workspace && tar -cf - --ignore-failed-read target/quiet-hw target/criterion 2>/dev/null || true' \
  | tar -xf - -C "$OUT" 2>/dev/null || true
ls -R "$OUT" | head -30

exit "$WORKLOAD_RC"
```

- [ ] **Step 4: Re-run the selftest (arg parsing must still pass)**

Run: `bash scripts/metal/lib-selftest.sh`
Expected: `metal lib-selftest: OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/metal/run.sh mise.toml
git commit -m "feat(metal): devpod workspace + workload execution + result collection"
```

---

### Task 8: mise tasks + selftest wiring

**Files:**
- Modify: `mise.toml` (three tasks)
- Modify: `scripts/metal/run.sh` (run selftest at start, like `verify.sh` does)

**Interfaces:**
- Consumes: all scripts from Tasks 1–7.
- Produces: `mise run metal -- <type> [flags] -- '<workload>'`, `mise run metal-catalog`, `mise run metal-gc [-- --force]`.

- [ ] **Step 1: Add the tasks to `mise.toml`** (after the existing `[tasks.verify-quiet-hw]` block; mise appends everything after `--` to the command, same mechanism `mise run bench -- <model>` uses):

```toml
[tasks.metal]
description = "Provision a PhoenixNAP box, run a workload, collect results, auto-deprovision (spec: docs/superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md; runbook: docs/runbooks/metal.md): mise run metal -- <type> [--yes|--keep|--reuse ID] -- '<workload>'"
run = "bash scripts/metal/run.sh"

[tasks.metal-catalog]
description = "List PhoenixNAP server types with CPU model, ISA flags (AVX2/AVX-512/...), $/hr, availability"
run = "bash scripts/metal/catalog.sh"

[tasks.metal-gc]
description = "List + delete stray inferno-metal servers (run after any interrupted metal session)"
run = "bash scripts/metal/gc.sh"
```

- [ ] **Step 2: Wire the selftest into `run.sh`** — in `run.sh`, immediately after the `METAL_PARSE_ONLY` block (so tests don't recurse) and before the preflight section, add:

```bash
# Selftest first — cheap, and it's the pass that guards every later stage
# (verify.sh discipline).
bash "$HERE/lib-selftest.sh" >/dev/null
```

- [ ] **Step 3: Verify the task surface**

Run: `mise tasks | grep metal && METAL_PARSE_ONLY=1 mise run metal -- d3.m5.xlarge --yes -- echo hi`
Expected: three `metal*` tasks listed; the parse-only run prints `type=d3.m5.xlarge yes=1 keep=0 reuse= workload=echo hi`.

- [ ] **Step 4: Commit**

```bash
git add mise.toml scripts/metal/run.sh
git commit -m "feat(metal): mise tasks metal / metal-catalog / metal-gc"
```

---

### Task 9: Runbook + doc pointers

**Files:**
- Create: `docs/runbooks/metal.md`
- Modify: `docs/runbooks/quiet-hw-verification.md` (add "on PhoenixNAP" recipe)
- Modify: `AGENTS.md` (one bullet under Non-obvious constraints)

**Interfaces:**
- Consumes: the full command surface from Task 8.
- Produces: operator documentation; no code.

- [ ] **Step 1: Write `docs/runbooks/metal.md`**

```markdown
# Runbook: PhoenixNAP bare-metal benchmarks (`mise run metal`)

Provision a bare-metal box, run any mise task or shell workload on it
inside the standard dev environment, collect results, deprovision. Spec:
[design](../superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md).

## Prerequisites (operator machine)

- `PNAP_CLIENT_ID` / `PNAP_CLIENT_SECRET` exported (PhoenixNAP portal →
  API Credentials). Never write them into any file in this repo.
- An SSH keypair (`~/.ssh/id_ed25519[.pub]` by default; `--ssh-key` to
  override).
- devpod, jq, curl — all mise-pinned (`mise install`).

## Pick hardware

    mise run metal-catalog

Columns: TYPE, CPU, VENDOR, CORES, FLAGS (the ISA feature list from
`scripts/metal/cpu-features.json`), USD/HR, IN-STOCK locations. `UNMAPPED`
rows need a `cpu-features.json` entry (with a vendor spec-sheet `source`)
before they can be provisioned.

## Run a workload

    mise run metal -- <type> --yes -- 'mise run test'
    mise run metal -- <type> -- 'MODEL=$(bash scripts/fetch-qwen-gguf.sh) && mise run bench -- "$MODEL"'

The workload string runs in the workspace root inside `devenv shell`.
Models are fetched ON the box — never uploaded from your machine. Results
land in `target/metal/<type>-<timestamp>/` (workload.log, host-prep.log,
metadata.json, plus anything under the box's `target/quiet-hw/` and
`target/criterion/`). The server is deleted on every exit path — including
Ctrl-C — unless `--keep`.

Iterating: `--keep` holds the box (prints the id; THE METER RUNS), then
`--reuse <id>` skips provisioning on the next run. `--reuse` never deletes
the box; delete it with `mise run metal-gc` when done.

## After ANY interrupted session

    mise run metal-gc

EXIT traps don't survive a killed terminal or laptop sleep. gc lists every
server tagged `inferno-metal` and deletes on confirm. Nothing else is ever
matched.

## CPU-feature drift

If host-prep aborts with `MISSING FLAG` / `UNEXPECTED FLAG` / `VENDOR
DRIFT`, the curated table disagrees with the silicon. Fix
`scripts/metal/cpu-features.json` in a commit (update the `source` link)
and rerun. There is deliberately no skip flag — a wrong entry mislabels
every result recorded for that type.

## Catalog changes

When PhoenixNAP adds/changes types: re-record fixtures
(`bash scripts/metal/record-fixtures.sh`), re-run
`bash scripts/metal/lib-selftest.sh`, extend `cpu-features.json`.
```

- [ ] **Step 2: Add the "on PhoenixNAP" recipe to `docs/runbooks/quiet-hw-verification.md`** — append after "The one-command path" section:

```markdown
## On PhoenixNAP bare metal

No local quiet hardware? Rent it (see [metal runbook](metal.md); costs
real money). Two sequential invocations — gates 1–4 want a quiet ≥12-core
AMD box, gate 5 a quiet Intel SKL+ box; pick types with
`mise run metal-catalog`:

    mise run metal -- <amd-type> --yes -- \
      'MODEL=$(bash scripts/fetch-qwen-gguf.sh) && mise run verify-quiet-hw -- "$MODEL"'
    mise run metal -- <intel-type> --yes -- \
      'MODEL=$(bash scripts/fetch-qwen-gguf.sh) && mise run verify-quiet-hw -- "$MODEL"'

Gate outputs land in `target/metal/<type>-<timestamp>/quiet-hw/`; paste
verdicts into the owning specs per the table above. The preflight still
rules: if the rented box is noisy, UNFIT is the correct answer there too.
```

- [ ] **Step 3: Add the AGENTS.md bullet** — append to the Non-obvious constraints list:

```markdown
- **`mise run metal` spends real money** (PhoenixNAP bare metal, hourly):
  operator-driven only, never CI. After any interrupted session run
  `mise run metal-gc` — EXIT traps don't survive killed terminals. The
  ISA table (`scripts/metal/cpu-features.json`) is verified against
  `/proc/cpuinfo` on every provision; on drift, fix the table in a
  commit, never override (see docs/runbooks/metal.md).
```

- [ ] **Step 4: Commit**

```bash
git add docs/runbooks/metal.md docs/runbooks/quiet-hw-verification.md AGENTS.md
git commit -m "docs(metal): runbook + quiet-hw-on-PhoenixNAP recipe + AGENTS.md constraint"
```

---

### Task 10: Paid E2E smoke (operator gate — requires credentials + spends money)

**This task provisions real hardware (~1–2 hrs of the cheapest box). It is the spec's one-time E2E verification, recorded like a bench data point. Do not run it from CI or unattended.**

**Files:**
- Modify: `docs/superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md` (§Amendments)

**Interfaces:**
- Consumes: everything.
- Produces: a recorded amendment: date, server type, cost, pipeline outcome, trap-kill outcome, gc outcome, plus confirmation (or correction) of the SSH user and OS default from Task 2.

- [ ] **Step 1: Pick the cheapest in-stock type**

Run: `mise run metal-catalog`
Note the cheapest USD/HR row that is in stock and mapped (not UNMAPPED).

- [ ] **Step 2: Full-pipeline smoke**

Run: `mise run metal -- <cheapest-type> --yes -- 'mise run lint'`
Expected: provision → host-prep prints `cpu:` + `host-prep: OK` (if it exits 4 with DRIFT lines, that's the table check working — fix `cpu-features.json` per the runbook and rerun) → devpod up → lint passes → results in `target/metal/<type>-<ts>/` containing `metadata.json`, `host-prep.log`, `workload.log` → `metal: deleting server`.

Then: `mise run metal-gc`
Expected: `metal-gc: no inferno-metal servers running` (teardown really happened).

- [ ] **Step 3: Trap-kill smoke**

Start `mise run metal -- <cheapest-type> --yes -- 'sleep 600'` and press Ctrl-C during the devpod-up stage.
Expected: `metal: deleting server ...` prints on the way out.
Then: `mise run metal-gc`
Expected: no servers. If one survived, that's a trap bug — fix it before recording, and delete the server via gc.

- [ ] **Step 4: Record the amendment and commit**

Append to the spec's `## Amendments`: date, type used, measured wall-clock for provision/devpod/total, approximate cost, all three outcomes (pipeline, trap-kill, gc), SSH user + OS image actually used. Never edit this data point later.

```bash
git add docs/superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md
git commit -m "docs(metal): record paid E2E smoke data point (spec amendment)"
```

---

## Plan self-review notes

- **Spec coverage:** layout/tasks (T1–T8), catalog+ISA table (T3–T4), gc (T5), pipeline stages 1–7 (T6–T7), error handling (trap T6, retries T1, drift T3, stock-out surfaces via pnap_api error passthrough), testing (selftests T1/T3/T4/T5/T6, integrity T3, paid smoke T10), runbook (T9), spec open items (T2, T10).
- **Deliberate deviation from spec wording:** collection uses tar streamed over `devpod ssh` instead of rsync — results live inside the container, where rsync isn't guaranteed; tar is. Noted here so the spec's "rsync" reads as "copy back".
- **Known-risky encodings, all fenced by verification steps:** BMC endpoint/JSON shapes (T2 Step 2), devpod CLI flags (T7 Step 2), SSH user/OS values (T2 Step 4, confirmed T10), microarch flag table (T3 cheat sheet, ground-truthed by host-prep on first provision).
