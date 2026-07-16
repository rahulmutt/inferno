#!/usr/bin/env bash
# M4b.11 attribution gate — the decode profiles the pre-registered gates
# consume: `inferno run --profile` at t=1 (comparable to the M4b.2
# 2026-07-07 baseline) and at t=<physical cores> (the operating point,
# exposing the serial-attention Amdahl share). Prints both profile outputs
# verbatim plus a parsed attention-share convenience summary. VERDICTS ARE
# HUMAN: paste the output into the M4b.11 spec §Amendments and compute
# S, in-situ GB/s, P1, P2 there per the spec's pre-registered formulas
# (docs/runbooks/quiet-hw-verification.md).
# Usage: gate-decode-attr.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-decode-attr.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PROMPT_BYTES=256; MAXTOK=8; else PROMPT_BYTES=2048; MAXTOK=64; fi

smoke_header "gate-decode-attr (M4b.11 attribution: decode profile t=1 + best-t)"
machine_block
echo

# The M4b.2 profile protocol: random base64 prompt (~1.3K tokens at 2048
# bytes), 64 generated tokens. --profile compiles a distinct (profiled)
# cache entry on first use; that build time is not part of any measurement.
PROMPT="$(head -c "$PROMPT_BYTES" /dev/urandom | base64 | tr -d '\n')"
TBEST="$(phys_cores)"

for T in 1 "$TBEST"; do
  echo "--- profile at --threads $T ---"
  cargo run --release -q -p inferno -- run "$MODEL" \
    --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads "$T" --profile \
    > "$OUT/profile-t$T.txt" 2>&1
  # The generated text is noise; show only the profile tables.
  sed -n '/^profile \[/,$p' "$OUT/profile-t$T.txt"
  echo
done

# Convenience summary: the decode-table attention share per thread count.
# The gate arithmetic (S, S', P1, P2) is controller work in the spec.
echo "attention decode share (parsed):"
for T in 1 "$TBEST"; do
  share=$(sed -n '/^profile \[decode\]/,$p' "$OUT/profile-t$T.txt" \
    | awk '$1 == "attention" { print $3; exit }')
  echo "  t=$T: ${share:-NOT-FOUND}"
done
