# M4b.7 — Quiet-Hardware Verification Pass (Turnkey Runbook) Design

**Date:** 2026-07-09
**Status:** Approved design, pre-implementation
**Milestone:** M4b.7 (see [inferno v1 design](2026-07-04-inferno-v1-design.md)
§Milestones; follows [M4b.6](2026-07-09-m4b6-decode-gemv-op-reduction-design.md))

Every M4b milestone deferred its performance verdict to "quiet hardware,"
and [M4b.1](2026-07-06-m4b1-threading-design.md) §Amendments proved why:
the dev environment is a devpod container whose root cgroup is CPU-quota'd
to 8 CPUs, with recorded mid-run throttling and nonzero external CPU
pressure — an environment in which **even llama.cpp's prefill scales
negatively**, so no engine's scaling can be demonstrated there. Five
verdicts are now queued on hardware that (per the user, 2026-07-09) is
**not currently available**:

1. **M4b.1** — prefill scaling ≥6× at t=12 (exit-criterion gate).
2. **M4b.4** — `PF_DIST` keep/revert + sweep {2,4,8} + the interleave
   (Task 3) go/no-go.
3. **M4b.5** — `decode_cap` default: sweep confirming
   `clamp(active/3, 2, active)` meets-or-beats best-fixed.
4. **M4a / v1 win criterion** — official `inferno bench` vs llama.cpp
   protocol run at full threads ("beat llama.cpp prefill **and** decode").
5. **M4b.6** — Intel re-run of the reduce-tree A/B (cherry-pick `092b191`
   from PR #11's pre-squash history) before declaring the op-reduction
   lever dead cross-vendor.

**This milestone therefore ships tooling and a runbook, not verdicts.**
Its product is a verification pass that is *one command* on future
hardware, with an environment-fitness preflight that makes another
confounded data point (the M4b.1 failure mode) impossible to record by
accident. All five verdicts stay open; each gate script prints an
amendment-ready table for its owning spec when it eventually runs.

## Scope Decisions (M4b.7)

| Decision | Choice |
|---|---|
| Product | Automated verification pass: fitness preflight + five gate scripts + `mise run verify-quiet-hw` orchestrator + runbook. **No performance verdicts recorded in this milestone** |
| Preflight | `scripts/quiet-hw/preflight.sh`: nproc ≥ 12; cgroup-v2 `cpu.max` unquota'd along the full hierarchy; `/proc/pressure/cpu` some-avg10 below threshold; throttling probe (calibration load, `cpu.stat` `nr_throttled` delta must be 0); records CPU vendor/model/core-count as the amendment machine block. UNFIT → nonzero exit, **hard stop** for the orchestrator |
| Gate scripts | One per deferred verdict (see §Gates), each runnable standalone or via the orchestrator; each prints an amendment-ready markdown table plus an explicit gate evaluation |
| Verdict discipline | Scripts **never write to spec files**. Pasting output into a spec's Amendments is a human act; keep/revert-style decisions remain human, made from the printed data (never-edit-recorded-data rule unchanged) |
| Library change | **One, minimal:** `PF_DIST` in `q8_0.rs` and `q4_k.rs` becomes compile-time overridable via `option_env!("INFERNO_PF_DIST")` (const-parsed; default 4; `0` = prefetch disabled via a const-folded branch). Zero runtime cost; prefetch is a pure hint so output bits are unaffected at every value |
| f32k.rs | `PF_DIST_F32` **untouched** — M4b.4's deferred sweep covers q8_0/q4_k only |
| Tolerances / ABI | Untouched by construction (`tolerance.rs` diff empty; no numeric change at any `INFERNO_PF_DIST` value; no `HOST_ABI_VERSION` bump) |
| Orchestrator | `mise run verify-quiet-hw -- <model.gguf>`: preflight → applicable gates (vendor-routed) → per-gate PASS/FAIL/SKIPPED summary → all output collected under a timestamped results dir. One gate failing does not abort the others |
| Smoke mode | Every gate script accepts `--smoke`: tiny reps/sizes, runs on unfit hardware, output stamped **`SMOKE — NON-RECORDABLE`** in the header so it can never be pasted into an Amendments section as data |
| Runbook | `docs/runbooks/quiet-hw-verification.md`: hardware requirements, the one-command path, per-gate manual fallback commands, and the spec + Amendments section each verdict belongs to |
| Exit criterion | Provable on this box (see §Exit Criterion): preflight flags **this devpod UNFIT** for the three known reasons; every gate completes `--smoke` end-to-end here; `INFERNO_PF_DIST` ∈ {0, 2, 8} builds pass the full correctness suite; runbook + mise task committed |
| CI | **No perf gates in CI, ever** (AGENTS.md). CI's only exposure: the existing correctness suite runs at the default build; nothing new is CI-gated |

**Explicitly out of scope:**

- **Recording any of the five verdicts** — impossible without the
  hardware; the pass exists so recording them later is turnkey.
- **`inferno bench` product changes** (e.g. embedding the fitness block
  in its report) — considered and rejected for this milestone: the
  preflight script covers the need without growing the product's
  Linux-specific /sys surface. Revisit if the pass is ever run on
  non-devenv machines.
- **PF_DIST sweep values beyond {0, 2, 4, 8}** and any q8a/f32 prefetch
  tuning.
- **The M4b.4 Task 3 interleave work itself** — the gate only produces
  the go/no-go data.
- **Any kernel, threading, or sampling change** — this is a tooling
  milestone; net product diff is the `PF_DIST` const wiring only.

## The Preflight — `scripts/quiet-hw/preflight.sh`

Automates exactly the probes M4b.1's amendment ran by hand, so fitness
is asserted *before* any data exists rather than discovered after:

1. **Core count:** `nproc` ≥ 12 (the protocol machine assumption).
2. **cgroup quota:** walk the cgroup-v2 hierarchy from the process's
   cgroup to root; every `cpu.max` must be `max` (unquota'd). This is
   the check that catches the devpod's `800000 100000`.
3. **External pressure:** `/proc/pressure/cpu` `some avg10` below a
   fixed threshold (default 1.0; the devpod recorded 11–15 mid-run).
4. **Throttling probe:** run a short all-core calibration load
   (`cargo bench` warmup or a busy-spin helper), then assert the
   `cpu.stat` `nr_throttled` delta across it is exactly 0 — the direct
   version of M4b.1's +164-periods observation.
5. **Machine block:** print CPU vendor/model, core/thread count, and
   the probe readings as the header every gate's amendment table reuses.
   Vendor (`AuthenticAMD`/`GenuineIntel`) routes gate 5.

FIT → exit 0 and print the machine block. UNFIT → exit nonzero listing
every failed probe. The orchestrator refuses to run any gate (except
`--smoke`) without a FIT preflight in the same invocation.

## The Gates

Each script lives in `scripts/quiet-hw/`, takes the model path where
needed, honors `--smoke`, and ends by printing (a) an amendment-ready
markdown table and (b) the gate's evaluation in the owning spec's own
terms. Measurement discipline is inherited from M4b: interleaved reps
where the verdict is a comparison, medians of per-rep values, ratios
computed per-rep then aggregated — never ratios of aggregates.

1. **`gate-prefill-scaling.sh` (M4b.1).** `inferno bench --threads t
   --json` (llama-bench comparison omitted — this gate is inferno-only
   scaling) at t ∈ {1, 2, 4, 8, 12}; pp/tg tok/s per thread count;
   scale factors vs t=1. Evaluation: prefill scaling **≥6× at t=12**. Data lands in
   M4b.1 §Amendments; a miss on fit hardware finally takes the spec's
   attribution fork (serial attention vs memory bandwidth) that
   confounded data could not.
2. **`gate-decode-cap.sh` (M4b.5).** Decode-thread sweep via the
   existing `INFERNO_DECODE_THREADS` hatch, 1..=capacity; finds the
   knee; compares the shipped `clamp(active/3, 2, active)` default
   against the best fixed count. Evaluation: default meets-or-beats
   best-fixed, high-thread regression gone, t=1 decode unchanged (M4b.5
   exit-criterion leg 2, verbatim).
3. **`gate-pf-dist.sh` (M4b.4).** Rebuild-per-value over
   `INFERNO_PF_DIST` ∈ {0, 2, 4, 8} (0 = no prefetch → the keep/revert
   arm and the sweep in one mechanism), interleaved reps per value on
   the M4b.4 shapes. Evaluation: keep/revert for q8_0 and q4_k, best
   distance, and the ≥5%-class signal that would authorize Task 3
   (interleave) — all recorded to M4b.4 §Amendments for the human call.
4. **`gate-bench-protocol.sh` (M4a / v1 win criterion).** Wraps the
   official protocol: `inferno bench --pp 512 --tg 128 --reps 5
   --threads <full> --json` with the devenv-pinned `llama-bench`
   comparison, per M4a §llama.cpp-side measurement. Evaluation: the v1
   criterion itself — beat llama.cpp **prefill and decode** tok/s at
   its best thread count. This is the only place v1-done can be judged.
5. **`gate-intel-ab.sh` (M4b.6, Intel-only).** Preflight-gated on
   `GenuineIntel`; SKIPPED (not FAILED) on AMD. Fetches PR #11's
   pre-squash ref (`git fetch origin refs/pull/11/head`), cherry-picks
   `092b191` into a scratch worktree, runs the interleaved
   reduce-unpack A/B (`cargo bench -p inferno-kernels --bench gemv`)
   with its per-process bitwise pre-check, and applies the M4b.6 ship
   gate arithmetic (fixed weights: .270/.211/.407/.087; conditions 1
   and 2). Evaluation: the cross-vendor verdict M4b.6's amendment left
   open (SKL µop model says wash, not loss).

## The `PF_DIST` Compile-Time Knob

The only product change. In `q8_0.rs` and `q4_k.rs`:

```rust
const PF_DIST: usize = match option_env!("INFERNO_PF_DIST") {
    Some(s) => parse_usize(s), // small const fn; compile_error-free path: invalid → panic at const-eval
    None => 4,
};
```

with the prefetch call sites wrapped in `if PF_DIST != 0 { ... }` — a
branch on a `const` the compiler folds away, so the default build's
codegen is unchanged and a `0` build contains no prefetch instruction.
Both files get the same wiring (one shared helper if extraction is
cleaner than duplication — implementer's judgment within the crate's
style). The doc comment on each const is updated to name the env var
and its sweep purpose. Because `_mm_prefetch` is a pure hint, output
bits are identical at every value; the existing bitwise scalar-oracle
and differential suites are the regression net, run at {0, 2, 8} as
part of this milestone's exit criterion.

## Orchestrator and Output Layout

`mise run verify-quiet-hw -- <model.gguf>` →
`scripts/quiet-hw/verify.sh`:

1. Preflight; UNFIT → stop (unless `--smoke`, which stamps everything
   NON-RECORDABLE and proceeds on any box).
2. Gates 1–4 in order (vendor-independent), gate 5 if Intel; each
   gate's stdout is tee'd to
   `target/quiet-hw/<timestamp>/gate-<name>.out` (git-ignored via
   `target/`; overridable with `--out-dir`) alongside a `summary.md`
   with per-gate PASS / FAIL / SKIPPED and the machine block — never
   inside `docs/`.
3. A gate that errors is recorded FAILED and the pass continues — on
   rare quiet hardware, partial data beats an aborted run.

The runbook (`docs/runbooks/quiet-hw-verification.md`) documents:
hardware requirements (quiet, unquota'd, ≥12 dedicated cores; Intel box
for gate 5), devenv-shell requirement, the one-command path, each
gate's manual fallback invocation, and — per gate — which spec's
Amendments section the pasted verdict belongs to.

## Testing & Exit Criterion

All provable on the current (unfit) devpod:

1. **Preflight negative test (the acceptance test):** run on this
   devpod, it must report UNFIT citing the quota (`cpu.max` =
   `800000 100000`), and print the machine block. Its FIT path is
   exercised by a unit-style fake: point it at a temp fake-cgroup tree
   (the script takes an overridable sysfs root for exactly this).
2. **Smoke pass:** `verify.sh --smoke <model>` completes end-to-end
   here — every gate runs its plumbing at tiny reps, every output file
   appears, summary renders, everything stamped NON-RECORDABLE. Gate 5
   smoke: on this AMD box it must report SKIPPED (vendor routing
   verified); its cherry-pick plumbing is smoke-tested with the vendor
   check bypassed via an explicit `--force-vendor` flag.
3. **PF_DIST build matrix:** `INFERNO_PF_DIST` ∈ {0, 2, 8} each build
   and pass `mise run test` (correctness is hardware-independent);
   default build passes the full suite + lint as always.
4. **Docs:** runbook committed; mise task registered; each of the five
   owning specs gets **no edit** (their Amendments fill in when the
   pass truly runs — this spec is the single pointer to the tooling).

Milestone closes when all four hold. The five verdicts remain open by
design; the ledger records this milestone as tooling-complete, not as
any gate passing.

## Risks

- **Preflight over- or under-strictness.** Thresholds (PSI 1.0,
  throttle delta 0) are first guesses; recorded in the script header as
  tunables with the M4b.1 observations as the calibration points. An
  UNFIT false-positive costs a manual override decision (documented
  flag `--i-know-what-im-doing` that stamps output UNFIT-OVERRIDE, so
  provenance survives); a false-negative is what the stamp trail is
  for.
- **PR #11 pre-squash ref availability.** `092b191` lives only in the
  PR ref on GitHub; if GitHub ever garbage-collects it, gate 5's
  cherry-pick fails loudly. Mitigation documented in the runbook: the
  arm's full source also exists in-tree at the M4b.6 plan's Task 1
  section, transcribable by hand.
- **Bash testability.** The verdict arithmetic (scale factors, medians,
  gate conditions) is the riskiest logic to leave untested in shell.
  Mitigation: computation helpers live in one `scripts/quiet-hw/lib.sh`
  with a `lib-selftest.sh` exercising them on fixed inputs (golden
  expected outputs), run as part of the smoke pass.
- **Bitrot before the hardware appears.** The smoke pass is cheap and
  devpod-safe; the runbook instructs re-running it before any real
  session, and the nightly tier is deliberately NOT extended (no perf
  jobs in CI, and a nightly smoke would download models for tooling
  that may sit unused for months).

## Amendments

*(Recorded protocol data points and scoped amendments land here; never
edit a recorded data point. The five inherited verdicts do NOT land
here — each belongs to its owning spec's Amendments, as routed by the
runbook.)*
