#!/usr/bin/env bash
# M4b.14 mid-milestone attention sub-bracket gate — the fresh split-bracket
# t=1 prefill profile the pre-registered ladder rule consumes (spec §Mid-
# Milestone Gate). Prints the t=1 prefill op table PLUS the attention kernel's
# scores/softmax/output sub-brackets (via the attn-subprofile feature, OFF in
# every shipping/bench build). The pp ratios come from gate-bench-protocol.sh
# in the same session. VERDICTS ARE HUMAN: paste into the M4b.14 spec
# §Amendments and compute there attn_share = attn_total / prefill_total and
# the ceiling check pp_ratio / (1 - f*c) >= 1.0 per the spec's pre-registered
# rule (docs/runbooks/quiet-hw-verification.md).
# Usage: gate-prefill-attn-split.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-prefill-attn-split.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PROMPT_BYTES=256; MAXTOK=8; else PROMPT_BYTES=2048; MAXTOK=32; fi

smoke_header "gate-prefill-attn-split (M4b.14: attn scores/softmax/output sub-brackets)"
machine_block
echo

PROMPT="$(head -c "$PROMPT_BYTES" /dev/urandom | base64 | tr -d '\n')"

echo "--- t=1 prefill profile + attn sub-brackets ---"
cargo run --release -q --features attn-subprofile -p inferno -- run "$MODEL" \
  --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads 1 --profile \
  > "$OUT/prefill-attn-split-t1.txt" 2>&1
sed -n '/^profile \[/,$p' "$OUT/prefill-attn-split-t1.txt"
echo
echo "--- attn sub-brackets (grep) ---"
grep -E '^attn:(scores|softmax|output)' "$OUT/prefill-attn-split-t1.txt" || true
