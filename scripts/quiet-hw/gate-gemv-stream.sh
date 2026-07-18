#!/usr/bin/env bash
# M4b.17 arms 1+2+3 — decode-shaped GEMV stream-rate arms (roofline,
# page/TLB, counter lane) on quiet hardware, via the gemv_stream example
# (shipping kernel through the shipping dispatch). Arm 4 (idle-gap) comes
# from gate-decode-attr.sh's profiles in the same session, not from here.
# VERDICTS ARE HUMAN: paste the tables into the M4b.17 spec §Amendments and
# apply gate rules 1–3 there. Counters are corroboration only (spec).
# Usage: gate-gemv-stream.sh   (env: QHW_OUT QHW_SMOKE QHW_NUMA_NODE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

OUT="${QHW_OUT:-$(mktemp -d)}"
mkdir -p "$OUT"
PHYS=$(phys_cores)
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  LANES=2 LAYERS=2 REPS=2
else
  LANES="$PHYS" LAYERS=24 REPS=5
fi

smoke_header "gate-gemv-stream (M4b.17: roofline + page/TLB + counters)"
machine_block
numa_require
[ -n "${QHW_NUMA_NODE:-}" ] && echo "numa: pinned to node ${QHW_NUMA_NODE} (cpubind+membind); phys_cores=$PHYS"
echo

cargo build --release -q -p inferno-pool --example gemv_stream

echo "--- arms at best-t (lanes=$LANES) ---"
$(numa_wrap) cargo run --release -q -p inferno-pool --example gemv_stream -- "$LANES" "$LAYERS" "$REPS" \
  | tee "$OUT/gemv-stream-t$LANES.txt"
echo
echo "--- arms at t=1 (per-thread quality context, not a gate quantity) ---"
$(numa_wrap) cargo run --release -q -p inferno-pool --example gemv_stream -- 1 "$LAYERS" "$REPS" \
  | tee "$OUT/gemv-stream-t1.txt"
echo

if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: counter lane skipped"
  exit 0
fi

# --- counter lane (corroboration only; spec §Task 1 arm 3) ---
if ! command -v perf >/dev/null; then
  echo "DEVIATION: perf unavailable — counter lane skipped (record in amendment)"
  exit 0
fi
EVENTS="cycles,instructions,dTLB-load-misses,LLC-load-misses"
for ARM in mmap4k thp; do
  LOG="$OUT/spin-$ARM.log"
  : > "$LOG"
  $(numa_wrap) cargo run --release -q -p inferno-pool --example gemv_stream -- \
    "$LANES" "$LAYERS" 1 --spin "$ARM" 25 > "$LOG" 2>&1 &
  BG=$!
  for _ in $(seq 240); do grep -q "STREAMING" "$LOG" && break; sleep 0.5; done
  grep -q "STREAMING" "$LOG" || { echo "spin never reached STREAMING ($ARM)"; cat "$LOG"; exit 1; }
  PID=$(sed -n 's/.*STREAMING pid=\([0-9]*\).*/\1/p' "$LOG" | head -1)
  echo "--- perf ($ARM, 5 s attach at lanes=$LANES) ---"
  perf stat -e "$EVENTS" -p "$PID" -- sleep 5 2>&1 | tee "$OUT/perf-$ARM.txt"
  wait "$BG" || true
  grep "AnonHugePages" "$LOG" || true
  echo
done
echo "gate arithmetic destination: M4b.17 spec §Amendments (rules 1-3)"
