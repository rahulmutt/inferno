# Runbook: quiet-hardware verification pass (M4b.7)

The five M4b performance verdicts deferred to quiet hardware, packaged as
one command. Spec:
[M4b.7 design](../superpowers/specs/2026-07-09-m4b7-quiet-hw-verification-design.md).

## Hardware requirements

- **Gates 1–4:** genuinely quiet, unquota'd machine, ≥12 dedicated cores
  (the specs' protocol assumption is a bare-metal Ryzen 9 3900-class box).
  A CPU-quota'd container CANNOT produce these verdicts — the preflight
  will refuse, and that refusal is correct (M4b.1 §Amendments is the
  cautionary tale: llama.cpp's own prefill scaled negatively there).
- **Gate 5:** a quiet Intel (SKL or newer) box; it is vendor-gated and
  reports SKIPPED elsewhere.

## The one-command path

    devenv shell                       # cc/LLVM/llama-bench/jq
    MODEL=$(bash scripts/fetch-qwen-gguf.sh)
    mise run verify-quiet-hw -- "$MODEL"

Everything lands in `target/quiet-hw/<timestamp>/` (`preflight.out`,
`gate-*.out`, `summary.md`). Before a real session, re-run the plumbing
check first (bitrot guard):

    mise run verify-quiet-hw -- "$MODEL" --smoke

Smoke output is stamped `SMOKE — NON-RECORDABLE` and must never be pasted
into a spec.

## Recording verdicts (human step — scripts never touch docs/)

| Gate output | Paste into | Decision recorded |
|---|---|---|
| `gate-prefill-scaling.out` | [M4b.1 spec](../superpowers/specs/2026-07-06-m4b1-threading-design.md) §Amendments | ≥6x @ t=12 met/not; on a miss, take the spec's attribution fork |
| `gate-decode-cap.out` | [M4b.5 spec](../superpowers/specs/2026-07-08-m4b5-phase-aware-decode-threading-design.md) §Amendments | cap default keep/change; knee; leg-2 verdict |
| `gate-pf-dist.out` | [M4b.4 spec](../superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md) §Amendments | PF_DIST keep/revert + distance; Task-3 (interleave) go/no-go |
| `gate-bench-protocol.out` | [M4a spec](../superpowers/specs/2026-07-06-m4a-bench-sampling-design.md) §Amendments | the v1 win criterion |
| `gate-intel-ab.out` | [M4b.6 spec](../superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md) §Amendments | op-reduction lever dead cross-vendor, or reopened |

Never edit a recorded data point. If a deciding shape straddles 0 in gate
5, re-run it with `--reps 6` before recording
(`bash scripts/quiet-hw/gate-intel-ab.sh --reps 6`).

## Per-gate manual fallbacks

Each gate runs standalone (same env vars the orchestrator sets —
`QHW_OUT` for the output dir):

    bash scripts/quiet-hw/preflight.sh
    bash scripts/quiet-hw/gate-prefill-scaling.sh "$MODEL"
    bash scripts/quiet-hw/gate-decode-cap.sh "$MODEL"
    bash scripts/quiet-hw/gate-pf-dist.sh
    bash scripts/quiet-hw/gate-bench-protocol.sh "$MODEL"
    bash scripts/quiet-hw/gate-intel-ab.sh            # Intel only

## Known fragilities

- **Gate 5's arm commit `092b191`** lives only in PR #11's pre-squash
  history (`refs/pull/11/head`). If GitHub ever drops that ref, the gate
  fails loudly; re-transcribe the arm from the M4b.6 plan's Task 1
  (`docs/superpowers/plans/2026-07-09-m4b6-reduce-unpack-restructure.md`),
  whose plan text contains the full module source.
- **Preflight thresholds** (12 CPUs, PSI 1.0, throttle delta 0) are
  tunables in the script header with M4b.1's observations as calibration
  points. On a false UNFIT you have judged wrong, `verify.sh
  --i-know-what-im-doing` forces the run — every output is then stamped
  `UNFIT-OVERRIDE` so provenance survives; record the override and your
  reasoning alongside any data you paste into an amendment.
- **`INFERNO_PF_DIST`** is a compile-time input (`option_env!`); gate 3
  builds one bench binary per value up front and interleaves the saved
  binaries, so no mid-measurement rebuilds occur.
