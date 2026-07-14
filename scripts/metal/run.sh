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
# A literal newline inside the workload is almost always a lost backslash
# continuation (single quotes keep the newline verbatim): the 2026-07-11
# payload run shipped '… && mise run\n  verify-quiet-hw …', which the box ran
# as two statements — a bare `mise run`, then an orphan command, exit 127 —
# AFTER the meter had paid for provisioning + devpod up. A genuinely
# multi-line workload can always be joined with && or ; — die before the
# meter starts.
case "${WORKLOAD[*]}" in *$'\n'*) metal_die \
  "workload contains a literal newline (lost backslash continuation?) — it would run as separate statements on the box; join it into one line with && or ;" ;;
esac

if [ "${METAL_PARSE_ONLY:-0}" = 1 ]; then
  printf 'type=%s yes=%s keep=%s reuse=%s workload=%s\n' \
    "$TYPE" "$YES" "$KEEP" "$REUSE" "${WORKLOAD[*]}"
  exit 0
fi

# Selftest first — cheap, and it's the pass that guards every later stage
# (verify.sh discipline).
bash "$HERE/lib-selftest.sh" >/dev/null

# --- preflight (local, free) -------------------------------------------
require_tools curl jq ssh ssh-keygen tar devpod git
require_env
[ -f "$SSH_KEY" ] || metal_die "ssh public key not found: $SSH_KEY (--ssh-key)"

# devpod forwards the operator's ssh-agent when SSH_AUTH_SOCK is set; a stale
# socket (common when the operator is itself a devpod workspace and inherited a
# now-dead agent-forwarding socket) makes `devpod up` fatal with
# "forward agent: dial unix ...: no such file". The box clones a public repo and
# pulls a public image, so it needs no forwarded agent — drop a dead socket
# rather than let devpod try to forward it. A live agent is left untouched.
if [ -n "${SSH_AUTH_SOCK:-}" ] && [ ! -S "$SSH_AUTH_SOCK" ]; then
  echo "metal: SSH_AUTH_SOCK ($SSH_AUTH_SOCK) is a dead socket — unsetting so devpod won't try to forward it" >&2
  unset SSH_AUTH_SOCK
fi

check_features_table || metal_die "cpu-features.json failed its integrity check"
ENTRY=$(jq -e --arg t "$TYPE" '.types[$t]' "$(features_table)") \
  || metal_die "server type '$TYPE' not in cpu-features.json — add it (with a vendor-sheet source) first"

# The box CLONES the repo from its git remote on provision — it never uploads
# the local working tree (target/ alone is tens of GB, and devpod's folder
# upload ignores .gitignore). So the exact committed HEAD must be reachable on
# a remote, or the box would build stale code. Check before the meter starts.
REPO=$(git rev-parse --show-toplevel) || metal_die "not inside a git repo"
REMOTE=$(git -C "$REPO" remote get-url origin 2>/dev/null) \
  || metal_die "no git remote 'origin' — metal clones the repo onto the box; add a remote and push first"
HEAD_SHA=$(git -C "$REPO" rev-parse HEAD)
[ -n "$(git -C "$REPO" branch -r --contains "$HEAD_SHA" 2>/dev/null)" ] \
  || metal_die "HEAD ($HEAD_SHA) isn't on any remote branch — the box clones $REMOTE and would run stale code; push first"

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
# provision + wait_ready live in lib.sh: probe.sh needs exactly the same
# non-idempotent POST and the same power-on/ssh poll, and the one call in this
# tool that can double-bill deserves a single implementation.

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
  SERVER_ID=$(metal_provision "$TYPE" "$OS" "$LOCATION" "$SSH_KEY" \
    "inferno-metal-${TYPE//./-}-$RUN_ID")
  meta_set server_id "\"$SERVER_ID\""
  echo "metal: server $SERVER_ID created; waiting for power-on + ssh"
  SERVER_IP=$(metal_wait_ready "$SERVER_ID" "$OUT/ssh-probe.log")
  meta_set server_ip "\"$SERVER_IP\""
  echo "metal: ready at $SERVER_IP; running host-prep"
  host_prep "$SERVER_IP"
fi
meta_set cpu_expected "$(jq -c '.' <<<"$ENTRY")"

# --- devpod workspace ------------------------------------------------------
# devpod provider/workspace names accept only [a-z0-9-]; RUN_ID is an ISO-8601
# basic timestamp (20260710T211824Z) whose literal T/Z are uppercase, so
# lowercase it for the derived names (the ISO form is kept for $OUT/metadata).
PROVIDER="metal-${RUN_ID,,}"
WORKSPACE="inferno-metal-${RUN_ID,,}"
devpod_cleanup() {
  devpod delete "$WORKSPACE" --force >/dev/null 2>&1 || true
  devpod provider delete "$PROVIDER" >/dev/null 2>&1 || true
}
# Extend the teardown path: devpod objects go first, then the server.
# The script's exit code is captured before devpod_cleanup can clobber $?.
trap 'rc=$?; devpod_cleanup; cleanup "$rc"' EXIT

echo "metal: creating devpod workspace on $SERVER_IP (image pull + devenv — minutes, not seconds)"
# `devpod up --provider X` refuses to run until X has been initialized
# ("used") at least once; --use=false skips that init and makes `up` fail with
# "provider is not initialized". Let add both register and initialize the
# ephemeral, per-run provider (devpod_cleanup deletes it on exit).
# INJECT_DOCKER_CREDENTIALS defaults true: mid-`up` the box's agent asks THIS
# machine for docker credentials. When the operator is itself a devpod
# workspace, ~/.docker/config.json says credsStore=devpod, whose helper posts
# to the OUTER devpod's credentials server on a hardcoded localhost port —
# alive only while the outer client holds a tunnel, dead in a headless
# session. devpod treats the failed lookup as fatal for image resolution
# ("retrieve image ...: EOF") instead of falling back to anonymous. The
# devcontainer image is public, so don't inject at all (same class as the
# dead-SSH_AUTH_SOCK guard in preflight).
devpod provider add ssh --name "$PROVIDER" \
  -o "HOST=$(metal_default_ssh_user)@$SERVER_IP" \
  -o INJECT_DOCKER_CREDENTIALS=false
# Clone the repo from git ON the box (pinned to the committed HEAD), rather
# than uploading the local working tree — devpod's folder upload ships the
# whole checkout including the tens-of-GB target/. The box builds fresh inside
# devenv anyway, so only tracked source needs to travel, and via git it does.
devpod up "$(metal_devpod_source "$REMOTE" "$HEAD_SHA")" \
  --provider "$PROVIDER" --id "$WORKSPACE" --ide none \
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
# A trailing "Error tunneling to container: wait: remote command exited
# without exit status or exit signal" in the log is a benign devpod teardown
# race (v0.6.15 pkg/tunnel/container.go: the auxiliary container-tunnel
# goroutine gets cancelled after the command session already returned the
# real exit code; the error is logged and swallowed there). It does NOT
# affect $?: a lost exit status on the *command* session surfaces as a
# non-zero devpod exit — a false failure, never a false success.
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
