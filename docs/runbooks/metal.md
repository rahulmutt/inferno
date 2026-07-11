# Runbook: PhoenixNAP bare-metal benchmarks (`mise run metal`)

Provision a bare-metal box, run any mise task or shell workload on it
inside the standard dev environment, collect results, deprovision. Spec:
[design](../superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md).

## Prerequisites (operator machine)

- `PNAP_CLIENT_ID` / `PNAP_CLIENT_SECRET` exported (PhoenixNAP portal →
  API Credentials). Never write them into any file in this repo.
  When creating the credential, grant the **`bmc`** scope (full BMC API
  access): `mise run metal` / `metal-gc` create and delete servers, which
  the read-only **`bmc.read`** scope does not allow. `bmc.read` alone is
  enough only for the read paths (`metal-catalog`,
  `metal-record-fixtures` — the billing products/availability endpoints
  are governed by the same `bmc`/`bmc.read` scope pair), so a
  least-privilege recon credential is possible, but provisioning needs
  `bmc`. Scope names are from PhoenixNAP's published OpenAPI specs
  (bmcapi/billingapi in the `phoenixnap/go-sdk-bmc` repo).
- An SSH keypair (`~/.ssh/id_ed25519[.pub]` by default; `--ssh-key` to
  override). Generate one if missing:

      ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519

  Accept or set a passphrase as you prefer; the public key is injected
  into the server at provision time, nothing needs uploading to the
  PhoenixNAP portal.
- devpod (mise-pinned, `mise install`); jq and curl from the system
  (not mise-pinned). `column` (util-linux) is optional — `metal-catalog`
  uses it only to align its table and falls back to raw tab-separated
  output when it's absent.
- Live status (see the spec's
  [Amendments](../superpowers/specs/2026-07-10-phoenixnap-bench-infra-design.md#amendments)):
  read path verified against the real API with re-recorded fixtures
  (2026-07-10), and the paid happy-path E2E smoke is green end-to-end
  (2026-07-11). Still OPEN: the kill-mid-run smoke (verify the EXIT trap
  deprovisions on Ctrl-C).

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

**The box clones the repo from its git `origin` at your committed `HEAD`**
(pinned by commit SHA), not your local working tree — the working tree
includes a tens-of-GB `target/` that devpod would otherwise upload wholesale.
So **commit and push before you run**: preflight aborts (before the meter
starts) if `HEAD` isn't reachable on a remote, and uncommitted changes never
reach the box. `metadata.json` records the `git_sha` that was benchmarked and
a `git_dirty` flag if your tree had uncommitted changes at launch.

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
