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

# Selftest first — cheap, and it's the pass that guards every later stage
# (verify.sh discipline).
bash "$HERE/lib-selftest.sh" >/dev/null

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
      # Explicit subshell: pnap_api can metal_die (exit) on a fatal status
      # (e.g. 401/403). An `exit` inside a `||`-guarded call terminates the
      # whole shell without ever reaching the `||` branch, silently
      # swallowing this hint. Running it in "$(...)"-free `(...)` contains
      # the exit to the subshell so the `if !` always gets to react.
      if ! (pnap_api DELETE "/bmc/v1/servers/$SERVER_ID" >/dev/null); then
        echo "metal: DELETE FAILED — run 'mise run metal-gc' NOW" >&2
      fi
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
  # POST /bmc/v1/servers is not idempotent: a blind retry after a 429/5xx
  # that actually succeeded server-side would provision a second, orphaned,
  # billed server (only the retry's id gets tracked/torn down).
  METAL_NO_RETRY=1 pnap_api POST /bmc/v1/servers "$body" | jq -er '.id'
}

wait_ready() { # <server-id> — prints IP once powered-on + ssh answers
  local deadline=$(( $(date +%s) + ${METAL_PROVISION_TIMEOUT:-1800} ))
  local s status ip lastline
  while [ "$(date +%s)" -lt "$deadline" ]; do
    s=$(pnap_api GET "/bmc/v1/servers/$1")
    status=$(jq -r '.status' <<<"$s")
    ip=$(jq -r '.publicIpAddresses[0] // empty' <<<"$s")
    if [ "$status" = "error" ]; then
      echo "metal: server entered error state" >&2; return 1
    fi
    if [ "$status" = "powered-on" ] && [ -n "$ip" ]; then
      # Default known_hosts + accept-new (not /dev/null): devpod's own
      # plain `ssh` later depends on the host key accepted here.
      if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes \
             "$(metal_default_ssh_user)@$ip" true 2>>"$OUT/ssh-probe.log"; then
        echo "$ip"; return 0
      fi
      lastline=$(tail -n1 "$OUT/ssh-probe.log" 2>/dev/null)
      echo "metal: waiting ($status, ssh: ${lastline:-no output yet})..." >&2
    else
      echo "metal: waiting ($status)..." >&2
    fi
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
  # sudo: PhoenixNAP cloud-image default login users have passwordless
  # sudo (verified live in Task 10) — host-prep.sh needs root for apt-get
  # and the /sys governor + docker-group writes.
  # shellcheck disable=SC2029
  if ! ssh -o StrictHostKeyChecking=accept-new "$(metal_default_ssh_user)@$1" \
       'sudo sh -s' "$expected" "$vocab" "$vendor" < "$HERE/host-prep.sh" \
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
# same environment every runbook assumes. %q-protects the one remote
# re-parse; the workload string is then shell source evaluated exactly
# once on the box (bash -c rules apply to quoting inside it).
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
ls -R "$OUT" | head -30 || true

exit "$WORKLOAD_RC"
