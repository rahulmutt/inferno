#!/usr/bin/env bash
# M4b.12 attribution gate — the dispatch-split profile the pre-registered
# gates consume, then the INFERNO_ATTN_SHARDS scaling sweep (menu guard's
# C(n) curve). Both run on a pool-profile build; prints the op tables and
# `pool [decode attention]` sections verbatim. VERDICTS ARE HUMAN: paste
# into the M4b.12 spec §Amendments and compute decode-wall shares, the
# menu guard C(max) vs C(1)/2, and P_W/P_A/P_D there, per the spec's
# pre-registered formulas (docs/runbooks/quiet-hw-verification.md).
# C(n) = kernel-max cycles / calls, from each sweep point's pool section.
# Usage: gate-attn-split.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-attn-split.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PROMPT_BYTES=256; MAXTOK=8; else PROMPT_BYTES=2048; MAXTOK=64; fi

smoke_header "gate-attn-split (M4b.12 attribution: dispatch-split profile + shard sweep)"
machine_block
echo

PROMPT="$(head -c "$PROMPT_BYTES" /dev/urandom | base64 | tr -d '\n')"
TBEST="$(phys_cores)"

echo "--- dispatch-split profile at --threads $TBEST ---"
cargo run --release -q -p inferno --features pool-profile -- run "$MODEL" \
  --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads "$TBEST" --profile \
  > "$OUT/attn-split-t$TBEST.txt" 2>&1
sed -n '/^profile \[/,$p' "$OUT/attn-split-t$TBEST.txt"
echo

echo "--- shard sweep (INFERNO_ATTN_SHARDS; pool sections only) ---"
for S in 1 2 4 7 "$TBEST"; do
  echo "--- INFERNO_ATTN_SHARDS=$S ---"
  INFERNO_ATTN_SHARDS="$S" cargo run --release -q -p inferno --features pool-profile -- run "$MODEL" \
    --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads "$TBEST" --profile \
    > "$OUT/attn-sweep-s$S.txt" 2>&1
  sed -n '/^pool \[decode attention\]/,$p' "$OUT/attn-sweep-s$S.txt"
  echo
done
