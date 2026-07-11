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

### 2026-07-10 — Task 2 recon (no-credentials fallback; live steps still OPEN)

`PNAP_CLIENT_ID`/`PNAP_CLIENT_SECRET` were **not available** in the
environment that ran Task 2. Per the task's own no-credentials fallback,
`scripts/metal/record-fixtures.sh` was written but never executed against
the live API, and `scripts/metal/fixtures/{products,availability}.json`
are **HAND-WRITTEN** from public PhoenixNAP catalog information (not
recorded from a real API response). This is a material gap:

- **Still OPEN, MUST be completed before Task 10:** run
  `bash scripts/metal/record-fixtures.sh` against the live API with real
  credentials, diff the result against the hand-written fixtures, and fix
  any field-shape or endpoint-path mismatch this recon could not observe
  live (e.g. actual HTTP status/error bodies, pagination, rate limits,
  whether the account's OAuth2 grant truly returns `access_token` in the
  shape `lib.sh`'s `pnap_token` expects). Re-run `lib-selftest.sh` after.

**What this recon *did* verify, statically, against PhoenixNAP's own
published OpenAPI source** (`specs/bmcapi.spec.yaml` and
`specs/billingapi.spec.yaml` in the `phoenixnap/go-sdk-bmc` GitHub repo —
the developer portal's `developers.phoenixnap.com/assets/bmc-api.yaml`
link 404s/redirects to the portal's JS app shell and is not fetchable
directly; the GitHub repo is the source the portal's app renders from and
is PhoenixNAP's own canonical spec):

- **Auth URL confirmed correct, statically:** `billingapi.spec.yaml`'s
  `securitySchemes.OAuth2.flows.clientCredentials.tokenUrl` is exactly
  `https://auth.phoenixnap.com/auth/realms/BMC/protocol/openid-connect/token`
  — matches `PNAP_AUTH_URL` in `lib.sh` verbatim. Not live-tested (no
  credentials to complete a real client-credentials grant and confirm the
  response actually contains `access_token`).
- **Endpoints/params confirmed correct, statically:** `/billing/v1/products`
  (query params `productCode`, `productCategory`, `skuCode`, `location`)
  and `/billing/v1/product-availability` (query params `productCategory`,
  `productCode`, `showOnlyMinQuantityAvailable`, `location`, `solution`,
  **`minQuantity`**) match `record-fixtures.sh`'s calls exactly.
- **Fixture field shapes confirmed correct** against the `ServerProduct` /
  `ProductAvailability` schemas: `productCode`, `productCategory`, `plans[].
  {pricingModel, price}`, `metadata.{cpu, cpuCount, coresPerCpu,
  cpuFrequency}`, and `locationAvailabilityDetails[].{location,
  minQuantityAvailable}` are all present with those exact names.
- **New finding, not previously known:** `minQuantityAvailable` in the real
  schema is a **boolean** ("is product available in specific location for
  the requested quantity"), not a count — the count lives in a separate
  `availableQuantity` field that the sanitizer does *not* capture (by
  design, matching the brief). Task 4 (`catalog.sh`) should treat
  `minQuantityAvailable` as a yes/no flag, not a quantity.
- **`METAL_OS` default — Debian was available in the enum, resolved:** the
  create-server `os` enum (`bmcapi.spec.yaml`, `Server.os` schema) offers
  `debian/bullseye`, `debian/bookworm`, `debian/trixie` (newest) alongside
  many Ubuntu/CentOS/Windows/etc. options. Per the spec's Debian
  preference, `metal_default_os` is now `debian/bookworm` (chosen over the
  newer `debian/trixie` as the better-tested current stable — cloud-init
  support in the same spec is also only documented through
  `debian/bookworm`, not `trixie`; override via `METAL_OS` if `trixie` is
  wanted). `ubuntu/jammy` remains available as a fallback value.
- **`METAL_SSH_USER` default — corrected, not just confirmed:** the BMC
  OpenAPI spec itself does not document per-OS default SSH users; that
  lives in the phoenixNAP KB
  (https://phoenixnap.com/kb/bmc-remote-console). It documents **`debian`**
  as the default login user for Debian servers (Ubuntu → `ubuntu`, CentOS
  → `centos`, Rocky Linux → `rockylinux`, ESXi/Proxmox → `root`). Task 1's
  provisional `root` default was **wrong** for Debian/Ubuntu Linux
  images (root has no password/key configured on those cloud images) —
  `metal_default_ssh_user` is now `debian`, matching the corrected
  `metal_default_os`.
- **AMD EPYC — not offered, contrary to the task brief's assumption:**
  PhoenixNAP's public Bare Metal Cloud instance catalog
  (phoenixnap.com/bare-metal-cloud/instances) lists only Intel Xeon (many
  generations: Haswell E3 through Xeon 6 Granite Rapids) and Ampere Altra
  ARM server types; no AMD EPYC bare-metal product was found anywhere in
  the public catalog as of this recon. The hand-written fixtures therefore
  span Intel Xeon generations plus one Ampere Altra (`a1.c5.large`, ARM,
  aarch64) as the "unmapped ISA" case for Task 3, and omit AMD EPYC rather
  than fabricate a nonexistent product code.
- **Fixture recording "date":** hand-written 2026-07-10, from public
  phoenixnap.com/bare-metal-cloud/instances pricing/spec pages and public
  CPU-vendor spec sheets for the cores/frequency of newer SKUs not listed
  with full specs on that page (Intel Xeon 6731E, 6767P; Ampere Altra
  Q80-30). Prices for a few newer/AI-ML SKUs without a published hourly
  rate are plausible placeholders, flagged as such above.

### 2026-07-10 — final-review fix wave

- **host-prep is invoked via `sudo`, not as root directly:** the default
  login user on PhoenixNAP's Debian/Ubuntu cloud images has no password
  set and is not `root`; `run.sh` runs `host-prep.sh` over ssh as that
  default user with `sudo sh -s`, relying on the cloud image granting it
  passwordless sudo. **Task 10 must explicitly verify passwordless sudo
  for the default login user on the chosen image** as part of its live
  verification pass (this was not — and could not be — checked by the
  no-credentials Task 2 recon above).

### 2026-07-10 — live read-path recon complete (write-path smoke still OPEN)

Ran against the live PhoenixNAP API with real `PNAP_CLIENT_ID` /
`PNAP_CLIENT_SECRET` (credential granted the `bmc` scope). **The read path
is now verified against real API responses, superseding the hand-written
fixtures from the Task 2 recon:**

- **OAuth2 grant confirmed live:** the client-credentials POST to
  `PNAP_AUTH_URL` returns a token whose `access_token` field is the shape
  `lib.sh`'s `pnap_token` expects — previously only static. Auth works.
- **Both billing endpoints confirmed live:** `mise run metal-record-fixtures`
  drove `/billing/v1/products?productCategory=SERVER` and
  `/billing/v1/product-availability?productCategory=SERVER&minQuantity=1`
  successfully; `mise run metal-catalog` joins the two plus the ISA table
  into a correct 7-column catalog (79 products). Real field shapes match
  what `catalog_join` / the sanitizer read — no endpoint or field-shape
  fix was needed.
- **`minQuantityAvailable` confirmed boolean live**, as the Task 2 static
  recon predicted; the catalog's IN-STOCK column is correct.
- **AMD absence confirmed live:** every priced SERVER product is Intel Xeon
  (`GenuineIntel`); the only ARM types (`a1.*`/`a2.*`, Ampere Altra) come
  back UNMAPPED. There is **no AMD EPYC** anywhere in the live catalog —
  the M4b.7 hardware ask (gates 1–4 wanted a quiet AMD ≥12-core box) has no
  PhoenixNAP equivalent and must be re-planned onto Intel (e.g.
  `d3.c3.large`, Xeon Gold 6542Y, 48c, full AVX-512+AMX, in stock PHX/NLD,
  $1.47/hr).
- **Fixtures re-recorded and committed.** Two fixes landed while doing so:
  (1) `record-fixtures.sh` now `unique`s the plans array — the API returns
  ~30 plan objects per product (one per pricingModel×location) that collapse
  to ~5 distinct once sanitized to `{pricingModel, price}`, so the raw
  recording was ~10k lines of duplicates; deduped it is ~2.5k and
  reviewable, and the tooling only reads the first HOURLY price so behavior
  is unchanged. (2) `lib-selftest.sh`'s `catalog_join | head -1` was
  replaced with a full-capture + parameter-expansion first-line read: under
  `set -o pipefail` the real (large) fixture made `jq` SIGPIPE (exit 141)
  when `head` closed the pipe early — a latent test bug the tiny
  hand-written fixture had masked. `mise run metal-selftest` is green
  against the real fixtures.
- New mise tasks expose these paths: `metal-selftest` (offline) and
  `metal-record-fixtures` (live, read-only, `bmc.read` sufficient).

**Still OPEN — the paid write-path E2E smoke** (needs the `bmc` write scope;
read-only `bmc.read` cannot exercise it): the server-lifecycle path
(`POST /bmc/v1/servers` create shape, power-on/ssh polling, ssh as the
`debian` user, **passwordless sudo** for that user per the fix-wave note
above, host-prep drift check on real silicon, devpod+devenv workload,
result collection, EXIT-trap deprovision) is entirely unexercised. Complete
both smokes from the §Testing plan — happy path via
`mise run metal -- <cheapest-type> --yes -- 'mise run lint'`, then a
kill-mid-run to verify the trap deprovisions — and record the result here.

### 2026-07-11 — paid happy-path E2E smoke GREEN (kill-mid-run smoke still OPEN)

`mise run metal -- s2.c2.medium --yes -- 'mise run lint'` completed the
full pipeline end-to-end: **s2.c2.medium** (Xeon E-2388G, 8c, $0.37/hr) in
NLD, server `6a521c0db55bb37c998de7e2`, started 10:33:47Z, finished
10:49:52Z (**~16 min wall**, dominated by devcontainer image pull + devenv
build), workload `mise run lint` **exit 0** on the box, results +
`metadata.json` landed in `target/metal/s2.c2.medium-20260711T103347Z/`,
server deleted by the teardown path. Every write-path item from the
2026-07-10 OPEN list is now verified live: create shape, power-on/ssh
polling, ssh as `debian`, **passwordless sudo confirmed** (host-prep ran
apt/governor/docker-group as root), drift check green on real silicon
(E-2388G matched the table), devpod+devenv workload, collect (empty tar —
correct for lint, which produces no `target/quiet-hw`/`target/criterion`),
EXIT-trap deprovision.

Getting to green peeled four live-only failures off the pipeline, each
fixed at root cause in a commit: (1) `devpod up --provider` on an
uninitialized provider (`--use=false` removed from `provider add`);
(2) `git:`-prefixed workspace source mangled by devpod's
`NormalizeRepository` into `https://git:https://…` (scheme guard in
`metal_devpod_source`, fails preflight before the meter);
(3) dead inherited `SSH_AUTH_SOCK` making `devpod up` fatal (preflight
drops dead sockets); (4) devpod's default `INJECT_DOCKER_CREDENTIALS=true`
forwarding the operator's docker credsStore to the box — when the operator
is itself a devpod workspace, that credsStore points at the OUTER devpod's
credentials server (dead in headless sessions) and the failed lookup
aborts the image pull with `retrieve image …: EOF` even for a public image
(no anonymous fallback in devpod). Fixed with
`-o INJECT_DOCKER_CREDENTIALS=false` on the ephemeral provider
(commit 24b2520). A trailing `Error tunneling to container: wait: remote
command exited without exit status or exit signal` in workload logs is a
benign devpod v0.6.15 teardown race (logged-and-swallowed in a goroutine
after the command session already returned the real exit code — see the
comment at the workload `devpod ssh` call in `run.sh`); it cannot fabricate
a false success.

**Still OPEN — the kill-mid-run smoke:** Ctrl-C a run after provisioning
returns a server id and verify the EXIT trap deprovisions (then
`mise run metal-gc` to confirm nothing is left). Not yet exercised live.
