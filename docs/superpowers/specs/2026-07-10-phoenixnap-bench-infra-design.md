# PhoenixNAP Bare-Metal Bench Infra Design

**Date:** 2026-07-10
**Status:** Approved design, pre-implementation

Every M4b performance verdict is queued on quiet bare-metal hardware
([M4b.7](2026-07-09-m4b7-quiet-hw-verification-design.md) packaged them
as `mise run verify-quiet-hw` but shipped no verdicts), and kernel
dispatch correctness across ISA levels (AVX2-only vs AVX-512 machines)
has never been exercised on real silicon we don't own. This milestone
builds the missing layer: tooling that provisions PhoenixNAP bare-metal
servers on demand, knows each server type's exact CPU features, runs any
mise task on them inside the existing reproducible dev environment, and
tears them down so an idle box never bills overnight.

**Driving decision (user, 2026-07-10): build the reusable hardware
matrix first, with the M4b.7 verification pass as its first consumer.**
The M4b.7 runbook gains a short "on PhoenixNAP" recipe; it does not get
special orchestration.

## Scope Decisions

| Decision | Choice |
|---|---|
| Product | `scripts/metal/` script suite + `cpu-features.json` ISA table + mise tasks `metal`, `metal-catalog`, `metal-gc` + runbook section |
| Orchestration model | One-shot workload runner: a single command provisions → preps → runs → collects → deprovisions (Approach A; session-style up/run/down and Terraform rejected as heavier for ~4 REST calls) |
| Lifecycle default | Ephemeral: auto-deprovision on exit via trap, `--keep` to hold, `metal-gc` as the stray-server backstop |
| Execution environment | devpod SSH provider reusing `.devcontainer/devcontainer.json` (image `ghcr.io/rahulmutt/dev`, `INSTALL_DEVENV=true`, `post-create.sh`) — identical environment to local dev, syncs uncommitted changes |
| ISA mapping | Curated checked-in table (`cpu-features.json`) for zero-cost planning; every provisioned box verifies `/proc/cpuinfo` against its entry at prep time and hard-fails on drift |
| Tooling shape | Bash + curl/jq wrapping the BMC REST API directly, matching `scripts/quiet-hw/` conventions; no pnapctl, no Terraform, no new pinned tools beyond what mise already has |
| Host OS | Config default `METAL_OS`: Debian if PhoenixNAP's catalog offers it, else Ubuntu LTS (verify against the API during implementation). Host choice is deliberately low-stakes: the toolchain lives in the container |
| Credentials | `PNAP_CLIENT_ID` / `PNAP_CLIENT_SECRET` env vars (BMC API uses OAuth2 client credentials); never written inside the repo; gitleaks guards accidents |
| CI | None. Operator-driven tooling only — no credentials in GitHub Actions, no scheduled provisioning. Same trust model as the quiet-hw runbook |
| Out of scope (v1) | Multi-box parallel orchestration (M4b.7's two boxes = two sequential invocations); spot/reservation pricing; ARM instance types; Windows/ESXi images |

## Layout

```
scripts/metal/
  lib.sh              # OAuth2 token, pnap_api() curl wrapper, tagging, polling
  catalog.sh          # metal-catalog: server types × cpu-features.json × availability
  run.sh              # metal: the one-shot pipeline
  host-prep.sh        # runs ON the box as root (portable sh: apt-get + /sys writes)
  gc.sh               # metal-gc: list/delete servers tagged inferno-metal
  cpu-features.json   # curated: server-type → CPU model, vendor, cores, ISA flags
  fixtures/           # recorded API JSON for the offline selftest
  lib-selftest.sh     # offline tests (no network, no credentials)
```

Results land in `target/metal/<server-type>-<timestamp>/` (gitignored),
mirroring `target/quiet-hw/`.

### mise tasks

| Task | Invocation |
|---|---|
| `metal` | `mise run metal -- <server-type> <task> [args...]` — e.g. `mise run metal -- d3.c2.medium verify-quiet-hw -- "$MODEL"` |
| `metal-catalog` | List instance types with CPU model, ISA flags, cores, $/hr, per-location availability |
| `metal-gc` | List stray `inferno-metal`-tagged servers (age, type, $/hr); delete on confirm, `--force` for scripted use |

## The `metal` pipeline

1. **Preflight (local):** `PNAP_CLIENT_ID`/`SECRET` set; `devpod`, `jq`,
   `rsync` on PATH; server type present in `cpu-features.json`; print the
   type's $/hr and require `--yes` or interactive confirmation before
   spending money.
2. **Provision:** `POST /bmc/v1/servers` — OS per `METAL_OS`, SSH public
   key (`--ssh-key`, default `~/.ssh/id_ed25519.pub`), hostname
   `inferno-metal-<type>-<timestamp>`, description tag `inferno-metal`
   (the exact string `metal-gc` filters on). Poll until `powered-on` and
   SSH answers; bounded at ~30 min, after which the server is deleted and
   the run fails loudly.
3. **Host prep:** copy `host-prep.sh`, run as root: install Docker if
   absent; set every core's scaling governor to `performance` (direct
   `/sys/devices/system/cpu/*/cpufreq/scaling_governor` writes — no
   distro-specific tools); read `/proc/cpuinfo` flags and **hard-fail on
   any disagreement** with the `cpu-features.json` entry, printing
   expected-vs-actual. No skip flag: a wrong table entry invalidates
   every result labeled with that type — fix the table in a commit.
4. **Workspace up:** configure devpod's SSH provider for the box;
   `devpod up . --ide none`. devpod injects its agent, pulls the
   devcontainer image, syncs the local repo including uncommitted
   changes, and runs `post-create.sh` (devenv install). Slowest stage —
   minutes, not seconds.
5. **Run:** `devpod ssh` → `devenv shell -- mise run <task> [args...]`
   in the workspace, streaming output and teeing to the results dir.
   Model files are fetched **on the box** (e.g. `fetch-qwen-gguf.sh`) —
   datacenter bandwidth, never uploaded from the operator's machine.
   Whether compound workloads use a quoted command string or a small
   wrapper mise task is settled in the implementation plan.
6. **Collect:** rsync back `target/quiet-hw/`, `target/criterion/`, and
   write `metadata.json` (server type, CPU model, verified cpuinfo
   flags, timestamps, git SHA + dirty status). Collect runs even when
   the workload exits non-zero — a failed gate's partial output is the
   diagnostic. The results dir and streaming log are created at stage 1
   and `metadata.json` is written incrementally as facts become known,
   so even early aborts leave a record.
7. **Teardown:** `DELETE /bmc/v1/servers/{id}`, unless `--keep` (print
   IP, devpod workspace name, and a meter-is-running reminder).
   `--reuse <id-or-ip>` skips stages 2–3 against a held box. The
   workload's exit code is preserved and reported after cleanup.

## Error handling & cost safety

The expensive failure is a stray server; teardown is defense-in-depth:

- **EXIT trap:** the server ID is registered in a trap the moment
  provisioning returns it. Host-prep abort, devpod failure, workload
  error, Ctrl-C — all still delete the server (unless `--keep`).
- **`metal-gc`:** traps don't survive a killed terminal or laptop
  sleep. gc lists everything tagged `inferno-metal` and deletes on
  confirm. The runbook instructs running it after any interrupted
  session. The filter must never match untagged servers.
- **Auth/API errors:** 401/403 fail immediately pointing at the env
  vars; 5xx/429 retry with bounded backoff.
- **Stock-outs:** surface the API error verbatim and point at
  `metal-catalog`'s availability column; never retry-loop a stock-out.

## ISA feature table

`cpu-features.json` maps each offered x86 server type to CPU model,
vendor, physical cores, and an explicit flag list (`avx2`, `avx512f`,
`avx512bw`, `avx512vl`, `avx512_vnni`, `amx_tile`, ...), sourced from
vendor spec sheets during implementation. Two consumers: `metal-catalog`
(planning: "which types have AVX-512?") and host-prep (verification).
AVX-512 entries must enumerate sub-features explicitly — inferno's
kernel dispatch will care about exactly which subset exists.

The M4b.7 hardware ask maps onto the table as: gates 1–4 want a quiet
AMD box with ≥12 physical cores (Ryzen 9 3900-class per the specs;
EPYC is the likely PhoenixNAP equivalent — note core-count and clock
differences in the amendment when recording); gate 5 wants any quiet
Intel SKL-or-newer box.

## Testing

- **`lib-selftest.sh`** (offline, follows `scripts/quiet-hw/`
  selftest convention): all API calls go through one `pnap_api()`
  function the selftest overrides with `fixtures/` JSON recorded once
  from the real API. Covers the catalog join, the cpuinfo-vs-table
  comparison (match / mismatch / missing-entry), gc tag filtering
  (including the must-not-match case), and arg parsing (`--keep`,
  `--reuse`, compound task strings).
- **Table integrity check** (offline): every entry complete; flags from
  a known vocabulary (catches `axv512f` typos); AVX-512 entries list
  sub-features.
- **One paid E2E smoke during implementation:** cheapest type, trivial
  task (`mise run lint`) through the full pipeline; verify results +
  metadata land and the server is gone. Second box: kill mid-run,
  verify the trap cleans up. Recorded in §Amendments like a bench data
  point; not repeatable CI.

## Open items to verify during implementation (not design forks)

- Exact OAuth2 token endpoint/shape for the BMC API; whether the
  supplied credential is a client-ID/secret pair or needs generating in
  the PhoenixNAP portal.
- Whether the OS catalog offers Debian (sets the `METAL_OS` default).
- Which billing/products endpoints carry $/hr and per-location
  availability for `metal-catalog`.

## Amendments

(none yet)
