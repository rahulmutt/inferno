# Runbook: PhoenixNAP bare-metal benchmarks (`mise run metal`)

Provision a bare-metal box, run any mise task or shell workload on it
inside the standard dev environment, collect results, deprovision. Spec:
[design](../superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md).

## Prerequisites (operator machine)

- `PNAP_CLIENT_ID` / `PNAP_CLIENT_SECRET` exported (PhoenixNAP portal →
  API Credentials). Never write them into any file in this repo.
- An SSH keypair (`~/.ssh/id_ed25519[.pub]` by default; `--ssh-key` to
  override).
- devpod (mise-pinned, `mise install`); jq and curl from the system
  (not mise-pinned).
- This tooling has never been exercised against the live PhoenixNAP API
  (fixtures are hand-written, not recorded — see the spec's
  [Amendments](../superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md#amendments)).
  Before first real use, complete the live recon + paid E2E smoke
  recorded there as OPEN, then re-record fixtures.

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
