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

## On PhoenixNAP bare metal

No local quiet hardware? Rent it (see [metal runbook](metal.md); costs
real money). Two sequential invocations — gates 1–4 want a quiet ≥12-core
AMD box, gate 5 a quiet Intel SKL+ box; pick types with
`mise run metal-catalog`:

    mise run metal -- <amd-type> --yes -- \
      'MODEL=$(bash scripts/fetch-qwen-gguf.sh) && mise run verify-quiet-hw -- "$MODEL"'
    mise run metal -- <intel-type> --yes -- \
      'MODEL=$(bash scripts/fetch-qwen-gguf.sh) && mise run verify-quiet-hw -- "$MODEL"'

Gate outputs land in `target/metal/<type>-<timestamp>/target/quiet-hw/`
(the collect tar preserves the box's `target/` prefix); paste
verdicts into the owning specs per the table above. The preflight still
rules: if the rented box is noisy, UNFIT is the correct answer there too.

Note: as of 2026-07 PhoenixNAP's catalog is Intel Xeon + Ampere ARM only —
no AMD EPYC. Until that changes, the AMD leg of gates 1–4 needs a different
vendor (or your own hardware); the Intel leg and gate 5 work as shown.

## Recording verdicts (human step — scripts never touch docs/)

| Gate output | Paste into | Decision recorded |
|---|---|---|
| `gate-prefill-scaling.out` | [M4b.1 spec](../superpowers/specs/2026-07-06-m4b1-threading-design.md) §Amendments | ≥6x @ t=12 met/not; on a miss, take the spec's attribution fork |
| `gate-decode-cap.out` | [M4b.5 spec](../superpowers/specs/2026-07-08-m4b5-phase-aware-decode-threading-design.md) §Amendments | cap default keep/change; knee; leg-2 verdict |
| `gate-bw-curve.out` | [M4b.10 spec](../superpowers/specs/2026-07-12-m4b10-decode-cap-formula-design.md) §Amendments | bandwidth curve; P (95%-of-peak lane count); does P predict the decode knee? |
| `gate-pf-dist.out` | [M4b.4 spec](../superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md) §Amendments | PF_DIST keep/revert + distance; Task-3 (interleave) go/no-go |
| `gate-bench-protocol.out` | [M4a spec](../superpowers/specs/2026-07-06-m4a-bench-sampling-design.md) §Amendments | the v1 win criterion |
| `gate-prefill-attr.out` | [M4b.13 spec](../superpowers/specs/2026-07-17-m4b13-prefill-attribution-design.md) §Amendments | matmul_share (sum of prefill table's matmul:* rows / prefill total) and ceiling-check arithmetic pp_ratio / (1 - matmul_share * 0.5) >= 1.0 |
| `gate-intel-ab.out` | [M4b.6 spec](../superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md) §Amendments | op-reduction lever dead cross-vendor, or reopened |
| `gate-decode-attr.out` | [M4b.11 spec](../superpowers/specs/2026-07-16-m4b11-decode-attention-f16kv-design.md) §Amendments | S (attention decode share, t=1 + best-t); in-situ GB/s; Gate 1 (P1) and Gate 2 (P2) arithmetic and verdicts |
| `gate-attn-split.out` | [M4b.12 spec](../superpowers/specs/2026-07-16-m4b12-decode-attention-headroom-attribution-design.md) §Amendments | blame table (publish/wake/kernel/drain + parked bits + H-alloc), sum identity (admissible 90–110%), C(n) sweep, menu guard, P_W/P_A/P_D arithmetic and verdicts |
| `gate-attn-perturb.out` | M4b.12 spec §Amendments | admissibility #2: within-session ship-vs-recording tg ratio (rework instrument if >1%) |
| `gate-attn-perf.out` | M4b.12 spec §Amendments | topdown + scheduler counters (worker-side view; escalation evidence if the menu guard fires) |

Never edit a recorded data point. If a deciding shape straddles 0 in gate
5, re-run it with `--reps 6` before recording
(`bash scripts/quiet-hw/gate-intel-ab.sh --reps 6`).

## Per-gate manual fallbacks

Each gate runs standalone (same env vars the orchestrator sets —
`QHW_OUT` for the output dir; `QHW_SMOKE=1` for a stamped plumbing
check — there is no per-gate `--smoke` flag):

**Socket-pinned sessions:** If using `QHW_NUMA_NODE=N` for a socket-pinned
run, set it only on individual gate invocations of `gate-decode-cap.sh` and
`gate-bw-curve.sh` — do not export it across a `verify.sh` pass. Those two
gates pin their runs with CPU and memory binding; the other gates (notably
`gate-bench-protocol.sh`) read `phys_cores` but do not pin, so a global
export gives them a node-sized thread count on an unpinned (whole-machine)
CPU set, silently corrupting the M4a data point.

    bash scripts/quiet-hw/preflight.sh
    bash scripts/quiet-hw/gate-prefill-scaling.sh "$MODEL"
    bash scripts/quiet-hw/gate-decode-cap.sh "$MODEL"
    bash scripts/quiet-hw/gate-bw-curve.sh
    bash scripts/quiet-hw/gate-pf-dist.sh
    bash scripts/quiet-hw/gate-bench-protocol.sh "$MODEL"
    bash scripts/quiet-hw/gate-intel-ab.sh            # Intel only

## Known fragilities

- **Gate 5's arm commit `092b191`** lives only in PR #11's pre-squash
  history (`refs/pull/11/head`). If GitHub ever drops that ref, the gate
  fails loudly; re-transcribe the arm from the M4b.6 plan's Task 1
  (`docs/superpowers/plans/2026-07-09-m4b6-reduce-unpack-restructure.md`),
  whose plan text contains the full module source.
- **Preflight thresholds**: min CPUs and PSI are tunables (`QHW_MIN_CPUS`,
  `QHW_PSI_MAX`; defaults 12 and 1.0, M4b.1's observations as calibration
  points); the throttle-delta-must-be-0 check is deliberately hardcoded.
  On a false UNFIT you have judged wrong, `verify.sh
  --i-know-what-im-doing` forces the run — every output is then stamped
  `UNFIT-OVERRIDE` so provenance survives; record the override and your
  reasoning alongside any data you paste into an amendment.
- **`INFERNO_PF_DIST`** is a compile-time input (`option_env!`); gate 3
  builds one bench binary per value up front and interleaves the saved
  binaries, so no mid-measurement rebuilds occur.
