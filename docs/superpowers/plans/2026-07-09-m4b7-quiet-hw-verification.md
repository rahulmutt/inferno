# M4b.7 Quiet-Hardware Verification Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the turnkey verification pass (fitness preflight + five gate scripts + orchestrator + runbook) that will record the five deferred M4b verdicts the moment quiet hardware exists — recording **no verdicts now**.

**Architecture:** Plain bash under `scripts/quiet-hw/` following the `scripts/nightly-*.sh` precedent, sharing verdict arithmetic through a self-tested `lib.sh`; one minimal Rust change makes `PF_DIST` a compile-time `option_env!` knob so the M4b.4 sweep is rebuild-per-value. An orchestrator behind `mise run verify-quiet-hw` chains preflight → gates and collects everything under `target/quiet-hw/<timestamp>/`.

**Tech Stack:** bash + awk + jq (all in devenv), cargo/criterion, the existing `inferno bench` / `inferno run` CLIs.

**Spec:** `docs/superpowers/specs/2026-07-09-m4b7-quiet-hw-verification-design.md` (approved). Read the §Scope Decisions table if a requirement below seems surprising.

## Global Constraints

- **No verdicts recorded in this milestone.** Scripts never write into `docs/`; every gate prints an amendment-ready table for a *human* to paste. Smoke output is stamped `SMOKE — NON-RECORDABLE` in its header.
- **Preflight UNFIT = hard stop** (nonzero exit); the orchestrator runs gates only after a FIT preflight in the same invocation, except under `--smoke`.
- **Library change is exactly one:** `PF_DIST` in `q8_0.rs` and `q4_k.rs` via `option_env!("INFERNO_PF_DIST")`, default `4`, `0` = no prefetch emitted. `f32k.rs`'s `PF_DIST_F32` untouched. `crates/inferno-graph/src/tolerance.rs` untouched (diff must be empty every commit). No `HOST_ABI_VERSION` bump.
- **No CI additions.** Nothing new runs in CI; shell selftests run inside the smoke pass, not in `mise run test`.
- **Measurement discipline (standing M4b):** interleaved reps; per-rep ratios, medians of per-rep ratios — never ratios of medians/aggregates.
- **Preflight tunables with calibration points** (record in script header comments): min 12 CPUs; PSI `some avg10` ≤ 1.0 (devpod recorded 11–15); throttle delta over calibration load must be 0 (devpod recorded +164 periods in one prefill); devpod quota signature `cpu.max = 800000 100000`.
- **Ship-gate arithmetic for gate 5 (fixed, from the M4b.6 amendments — never re-derive):** `projected = 0.270·w(151936x896) + 0.211·w(896x4864) + 0.407·w(4864x896) + 0.087·w(896x896)`; condition 1 = `w_r > 0` in every rep on ≥2 of the 3 mid shapes (896x896, 4864x896, 896x4864); condition 2 = no shape median `w < −3%`; deciding-shape straddle-0 → extend 3 → 6 reps.
- **Before every commit:** `mise run test` (265+ tests green) and `devenv shell -- mise run lint` (clippy needs LLVM; plain `mise run lint` fails outside devenv).
- Bash scripts: `#!/usr/bin/env bash`, `set -euo pipefail`, `command -v` guard for tools that need devenv shell — the `nightly-speedup.sh` conventions.

## Execution Notes (controller)

- Branch: `m4b7-quiet-hw-verification` off `main`.
- Tasks 1–3 are independent of each other; 4–6 need Task 2 (`lib.sh`); Task 5 needs Task 1; Task 7 needs all.
- The smoke model comes from `bash scripts/fetch-qwen-gguf.sh` (echoes a cached GGUF path; needs devenv shell + network on first run).
- `cargo` invocations that build (`bench --no-run`, `run --release`) must run inside `devenv shell` (cc/LLVM). Selftests that only exercise awk/bash need no devenv.

## File Structure

| File | Responsibility |
|---|---|
| `crates/inferno-kernels/src/pf.rs` (new) | `parse_pf_dist` const fn + unit tests |
| `crates/inferno-kernels/src/q8_0.rs`, `src/q4_k.rs`, `src/lib.rs` (modify) | `PF_DIST` knob wiring |
| `scripts/quiet-hw/lib.sh` (new) | shared verdict arithmetic + parsers + smoke stamp |
| `scripts/quiet-hw/lib-selftest.sh` (new) | golden tests for every lib.sh function |
| `scripts/quiet-hw/preflight.sh` (new) | fitness probes, FIT/UNFIT verdict, machine block |
| `scripts/quiet-hw/preflight-selftest.sh` (new) | fake cgroup/proc trees: FIT and UNFIT paths |
| `scripts/quiet-hw/gate-prefill-scaling.sh` (new) | M4b.1 gate |
| `scripts/quiet-hw/gate-decode-cap.sh` (new) | M4b.5 gate |
| `scripts/quiet-hw/gate-pf-dist.sh` (new) | M4b.4 gate |
| `scripts/quiet-hw/gate-bench-protocol.sh` (new) | M4a / v1-win gate |
| `scripts/quiet-hw/gate-intel-ab.sh` (new) | M4b.6 Intel A/B gate |
| `scripts/quiet-hw/verify.sh` (new) | orchestrator |
| `mise.toml` (modify) | `verify-quiet-hw` task |
| `docs/runbooks/quiet-hw-verification.md` (new) | the runbook |

Environment-variable protocol between orchestrator and gates (every gate honors all of these):

- `QHW_OUT` — output dir for the gate's files (default: a `mktemp -d`).
- `QHW_SMOKE=1` — smoke mode: tiny reps/sizes, `SMOKE — NON-RECORDABLE` header, gate evaluation lines replaced by `SMOKE: evaluation skipped`.
- `QHW_CGROUP_ROOT`, `QHW_PROC_ROOT`, `QHW_NPROC`, `QHW_MIN_CPUS`, `QHW_PSI_MAX`, `QHW_CALIB_SECS` — preflight-only test/tuning seams.

---

### Task 1: `PF_DIST` compile-time knob (`INFERNO_PF_DIST`)

**Files:**
- Create: `crates/inferno-kernels/src/pf.rs`
- Modify: `crates/inferno-kernels/src/lib.rs` (add `mod pf;` beside the other private mods, line ~20)
- Modify: `crates/inferno-kernels/src/q8_0.rs:14-19` (const), `:155-163` (guard)
- Modify: `crates/inferno-kernels/src/q4_k.rs:16-21` (const), `:164-171` (guard)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: rebuild-per-value builds keyed by the `INFERNO_PF_DIST` env var (a *compilation input* via `option_env!` — rustc records the env dependency, so cargo rebuilds `inferno-kernels` automatically when the value changes). `0` = no prefetch instruction. Task 5's gate script relies on exactly this behavior. `pub(crate) const fn parse_pf_dist(&str) -> usize` in `crate::pf`.

- [ ] **Step 1: Write the failing tests**

Create `crates/inferno-kernels/src/pf.rs`:

```rust
//! Compile-time parsing for the `INFERNO_PF_DIST` prefetch-distance
//! override (M4b.7 quiet-hardware sweep). `option_env!` makes the env var
//! a compilation input, so a rebuild with a different value re-evaluates
//! the kernel consts — no runtime cost at any value.

/// Parse a decimal `usize` at const-eval time. Panics — a compile error
/// in const context — on anything but a plain decimal integer.
pub(crate) const fn parse_pf_dist(s: &str) -> usize {
    let b = s.as_bytes();
    assert!(!b.is_empty(), "INFERNO_PF_DIST must be a decimal integer");
    let mut v = 0usize;
    let mut i = 0;
    while i < b.len() {
        assert!(
            b[i].is_ascii_digit(),
            "INFERNO_PF_DIST must be a decimal integer"
        );
        v = v * 10 + (b[i] - b'0') as usize;
        i += 1;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::parse_pf_dist;

    #[test]
    fn parses_decimal() {
        assert_eq!(parse_pf_dist("0"), 0);
        assert_eq!(parse_pf_dist("4"), 4);
        assert_eq!(parse_pf_dist("12"), 12);
    }

    #[test]
    #[should_panic(expected = "decimal integer")]
    fn rejects_empty() {
        parse_pf_dist("");
    }

    #[test]
    #[should_panic(expected = "decimal integer")]
    fn rejects_non_digit() {
        parse_pf_dist("4x");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo nextest run -p inferno-kernels pf::`
Expected: FAIL to build — `mod pf` is not declared yet.

- [ ] **Step 3: Declare the module**

In `crates/inferno-kernels/src/lib.rs`, next to the other private mods (`mod attention;` etc., lines 19-27), add:

```rust
mod pf;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo nextest run -p inferno-kernels pf::`
Expected: 3 tests PASS.

- [ ] **Step 5: Wire the q8_0 const and guard**

In `crates/inferno-kernels/src/q8_0.rs`, replace lines 14-19 (the `PF_DIST` doc comment + const) with:

```rust
/// Weight groups to software-prefetch ahead in the AVX2 GEMV (M4b.4). A
/// strip's `nb` groups are contiguous (`nb × GROUP_BYTES`), so prefetching
/// `PF_DIST` groups ahead reaches cleanly across the block loop and into the
/// next strip. Pure hint, so it never affects output bits at any value.
/// Compile-time override via `INFERNO_PF_DIST` for the M4b.7 quiet-hardware
/// sweep (`0` = no prefetch emitted); default 4 pending the deferred M4b.4
/// sweep — see that spec's Amendment.
const PF_DIST: usize = match option_env!("INFERNO_PF_DIST") {
    Some(s) => crate::pf::parse_pf_dist(s),
    None => 4,
};
```

Then wrap the prefetch call site (lines 155-163: the comment, `pf_addr` binding, and `_mm_prefetch` call) in a const-folded guard, keeping the existing comment text inside:

```rust
                if PF_DIST != 0 {
                    // Prefetch a future weight group into L1 to overlap DRAM latency
                    // with this block's int8 dot. `wrapping_add` (not `add`) because
                    // the last strip's tail offsets point past the buffer end;
                    // `_mm_prefetch` never dereferences and never faults, so it stays
                    // a pure hint — output is unchanged.
                    let pf_addr = w
                        .wrapping_add((strip * nb + b + PF_DIST) * GROUP_BYTES)
                        .cast();
                    _mm_prefetch::<_MM_HINT_T0>(pf_addr);
                }
```

(Adopt the first comment line verbatim from the current file — the block above shows the shape; only the `if PF_DIST != 0 { }` wrapper and indentation change.)

- [ ] **Step 6: Wire the q4_k const and guard**

Same two edits in `crates/inferno-kernels/src/q4_k.rs`: replace the const at lines 16-21 with the same `match option_env!(...)` form (keep q4_k's own doc comment, adding the same two override sentences), and wrap the prefetch site (lines 164-171, `nsb`/`sb` variant) in `if PF_DIST != 0 { ... }` the same way.

- [ ] **Step 7: Full suite at default build**

Run: `mise run test`
Expected: all tests pass (baseline was 265 passed / 3 skipped); default build's `PF_DIST` is still 4.

- [ ] **Step 8: Build matrix — correctness is PF_DIST-independent**

```bash
for v in 0 2 8; do
  INFERNO_PF_DIST=$v cargo nextest run -p inferno-kernels || exit 1
done
```

Expected: each value triggers a rebuild of `inferno-kernels` (env change is a compilation input) and the kernel suite passes bitwise — prefetch is a pure hint. Also confirm the invalid value fails to *compile*:

Run: `INFERNO_PF_DIST=nope cargo check -p inferno-kernels 2>&1 | grep -c "decimal integer"`
Expected: ≥1 (const-eval panic surfaces as a compile error).

- [ ] **Step 9: Lint, tolerance-diff check, commit**

```bash
devenv shell -- mise run lint
git diff --exit-code -- crates/inferno-graph/src/tolerance.rs
git add crates/inferno-kernels/src/pf.rs crates/inferno-kernels/src/lib.rs \
        crates/inferno-kernels/src/q8_0.rs crates/inferno-kernels/src/q4_k.rs
git commit -m "kernels: INFERNO_PF_DIST compile-time prefetch knob (M4b.7)"
```

---

### Task 2: `scripts/quiet-hw/lib.sh` + selftest

**Files:**
- Create: `scripts/quiet-hw/lib.sh`
- Create: `scripts/quiet-hw/lib-selftest.sh` (chmod +x both; `git add --chmod=+x` works too)

**Interfaces:**
- Consumes: nothing.
- Produces (all later tasks): `median <v...>`; `pct <a> <b>` (= `100*(a/b − 1)`, 2 decimals); `crit_mid_ns <file> <id-regex>` (lines `"<bench-id> <mid-ns>"`); `decode_toks <file-or->` (decode tok/s from `inferno run` output); `cpu_vendor`; `machine_block`; `smoke_header <gate-name>`; `phys_cores`. Env seams: `QHW_SMOKE`, `QHW_PROC_ROOT`.

- [ ] **Step 1: Write the selftest (failing)**

Create `scripts/quiet-hw/lib-selftest.sh`:

```bash
#!/usr/bin/env bash
# Golden tests for lib.sh — pure text processing, no devenv/model needed.
# Run standalone or via verify.sh --smoke.
set -euo pipefail
. "$(dirname "$0")/lib.sh"

fail() { echo "SELFTEST FAIL: $*" >&2; exit 1; }
expect() { # <label> <got> <want>
  [ "$2" = "$3" ] || fail "$1: got '$2', want '$3'"
}

expect "median odd"  "$(median 3 1 2)" "2"
expect "median even" "$(median 4 1 2 3)" "2.5"
expect "median one"  "$(median 7)" "7"
expect "pct up"      "$(pct 110 100)" "10.00"
expect "pct down"    "$(pct 97 100)" "-3.00"

# criterion output: same-line and wrapped id/time forms, µs/ms units.
crit=$(mktemp)
cat > "$crit" <<'EOF'
gemv/Q8_0/inferno-avx2/896x896
                        time:   [10.000 µs 10.500 µs 11.000 µs]
gemv/Q8_0/inferno-avx2/151936x896 time:   [12.300 ms 12.400 ms 12.500 ms]
gemv/Q8_0/reduce-unpack/896x896
                        time:   [9.000 µs 9.500 µs 10.000 µs]
Benchmarking gemv/Q8_0/inferno-avx2/896x896: Collecting 100 samples
EOF
expect "crit wrapped" \
  "$(crit_mid_ns "$crit" 'inferno-avx2/896x896$')" \
  "gemv/Q8_0/inferno-avx2/896x896 10500"
expect "crit sameline" \
  "$(crit_mid_ns "$crit" '151936x896$')" \
  "gemv/Q8_0/inferno-avx2/151936x896 1.24e+07"
expect "crit two ids" \
  "$(crit_mid_ns "$crit" '896x896$' | wc -l)" \
  "2"
rm -f "$crit"

run_out='prefill: 6 tok in 0.4s (15.00 tok/s) | decode: 128 tok in 11.6s (11.03 tok/s)'
expect "decode_toks" "$(echo "$run_out" | decode_toks -)" "11.03"

QHW_SMOKE=1
expect "smoke stamp" \
  "$(smoke_header x | head -1)" \
  "### SMOKE — NON-RECORDABLE (plumbing check on unfit hardware; never paste into a spec) ###"
QHW_SMOKE=0
case "$(smoke_header x | head -1)" in
  '# x'*) ;;
  *) fail "non-smoke header should start with '# x'" ;;
esac
QHW_OVERRIDE=1
expect "override stamp" \
  "$(smoke_header x | head -1)" \
  "### UNFIT-OVERRIDE (preflight failed; operator forced the run — record the override alongside any data) ###"
QHW_OVERRIDE=0

# median with no args must fail loudly, not return 0.
if out=$(median 2>/dev/null); then fail "median with no args should return nonzero (got '$out')"; fi

# phys_cores must survive an lscpu that emits only comments (pipefail trap)
# and fall back to nproc. Stub lscpu via PATH.
stub=$(mktemp -d)
printf '#!/usr/bin/env bash\necho "# comment only"\n' > "$stub/lscpu"
chmod +x "$stub/lscpu"
n=$(PATH="$stub:$PATH" bash -euo pipefail -c ". '$(dirname "$0")/lib.sh'; phys_cores")
[ "${n:-0}" -ge 1 ] || fail "phys_cores with data-less lscpu should fall back to nproc (got '$n')"
rm -rf "$stub"

echo "lib-selftest: OK"
```

- [ ] **Step 2: Run to verify it fails**

Run: `bash scripts/quiet-hw/lib-selftest.sh`
Expected: FAIL — `lib.sh: No such file or directory`.

- [ ] **Step 3: Write lib.sh**

Create `scripts/quiet-hw/lib.sh`:

```bash
#!/usr/bin/env bash
# Shared helpers for the M4b.7 quiet-hardware verification pass (spec:
# docs/superpowers/specs/2026-07-09-m4b7-quiet-hw-verification-design.md).
# Sourced by preflight.sh, gate-*.sh, verify.sh. Tested by lib-selftest.sh.
# Discipline: per-rep ratios, medians of ratios — never ratios of medians.

# median <v1> [v2 ...] — median; even count = mean of the middle two.
median() {
  [ $# -ge 1 ] || { echo "nan"; return 1; }
  printf '%s\n' "$@" | sort -g | awk '
    { v[NR] = $1 }
    END {
      if (NR == 0) { print "nan"; exit 1 }
      if (NR % 2) printf "%g\n", v[(NR + 1) / 2]
      else printf "%g\n", (v[NR / 2] + v[NR / 2 + 1]) / 2
    }'
}

# pct <a> <b> — 100 * (a/b − 1), two decimals (positive = a above b).
pct() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f\n", 100 * (a / b - 1) }'; }

# crit_mid_ns <file> <id-regex> — for each criterion bench id matching the
# regex, print "<id> <middle-estimate-in-ns>". Handles the id and its
# "time: [lo mid hi]" on the same or consecutive lines; skips
# "Benchmarking <id>: ..." progress lines.
crit_mid_ns() {
  awk -v re="$2" '
    function tons(v, u) {
      if (u == "ns") return v
      if (u ~ /^(µs|us)$/) return v * 1e3
      if (u == "ms") return v * 1e6
      return v * 1e9  # "s"
    }
    $1 != "Benchmarking" {
      for (i = 1; i <= NF; i++) if ($i ~ re && $i !~ /:$/) id = $i
      for (i = 1; i <= NF; i++)
        if ($i == "time:" && id != "") {
          v = $(i + 3); u = $(i + 4)
          gsub(/[\[\]]/, "", v); gsub(/[\[\]]/, "", u)
          printf "%s %.6g\n", id, tons(v, u)
          id = ""
          break
        }
    }' "$1"
}

# decode_toks <file-or--> — decode tok/s from `inferno run` output.
decode_toks() { sed -n 's#.*decode: .*(\([0-9.]*\) tok/s).*#\1#p' "$1"; }

cpu_vendor() {
  awk -F': *' '/^vendor_id/ { print $2; exit }' "${QHW_PROC_ROOT:-/proc}/cpuinfo"
}

machine_block() {
  local ci="${QHW_PROC_ROOT:-/proc}/cpuinfo"
  local model vendor
  model=$(awk -F': *' '/^model name/ { print $2; exit }' "$ci")
  vendor=$(cpu_vendor)
  echo "machine: ${model:-unknown} (${vendor:-unknown}) | $(nproc) logical CPUs | kernel $(uname -r) | $(date -u +%Y-%m-%d)"
}

# smoke_header <gate-name> — every gate prints this first. The stamps are
# the no-accidental-data-point guarantee: they must be the FIRST lines.
smoke_header() {
  if [ "${QHW_SMOKE:-0}" = 1 ]; then
    echo "### SMOKE — NON-RECORDABLE (plumbing check on unfit hardware; never paste into a spec) ###"
  fi
  if [ "${QHW_OVERRIDE:-0}" = 1 ]; then
    echo "### UNFIT-OVERRIDE (preflight failed; operator forced the run — record the override alongside any data) ###"
  fi
  echo "# $1 — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# phys_cores — physical core count (sweep upper bound for gate-decode-cap).
# Never fails under `set -euo pipefail`: an lscpu that exists but emits no
# data rows (sandboxes, odd VMs) falls back to nproc.
phys_cores() {
  local n=0
  if command -v lscpu >/dev/null; then
    n=$( (lscpu -p=CORE,SOCKET 2>/dev/null || true) \
         | awk '!/^#/ && NF && !seen[$0]++ { n++ } END { print n + 0 }')
  fi
  if [ "${n:-0}" -ge 1 ]; then echo "$n"; else nproc; fi
}
```

- [ ] **Step 4: Run the selftest to verify it passes**

Run: `bash scripts/quiet-hw/lib-selftest.sh`
Expected: `lib-selftest: OK`. If a `crit_mid_ns` case fails, debug the awk against the heredoc — do not weaken the expected values.

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/quiet-hw/lib.sh scripts/quiet-hw/lib-selftest.sh
mise run test && devenv shell -- mise run lint
git add scripts/quiet-hw/lib.sh scripts/quiet-hw/lib-selftest.sh
git commit -m "scripts: quiet-hw shared helpers + selftest (M4b.7)"
```

---

### Task 3: `preflight.sh` + selftest

**Files:**
- Create: `scripts/quiet-hw/preflight.sh`
- Create: `scripts/quiet-hw/preflight-selftest.sh`

**Interfaces:**
- Consumes: `lib.sh` (`machine_block`, `cpu_vendor`).
- Produces: exit 0 + `PREFLIGHT: FIT` line and machine block on fit hardware; exit 1 + `PREFLIGHT: UNFIT` + one ` - reason` line per failed probe. `verify.sh` (Task 7) keys off the exit code. Test seams: `QHW_CGROUP_ROOT`, `QHW_PROC_ROOT`, `QHW_NPROC`, `QHW_MIN_CPUS`, `QHW_PSI_MAX`, `QHW_CALIB_SECS`.

- [ ] **Step 1: Write the selftest (failing)**

Create `scripts/quiet-hw/preflight-selftest.sh`:

```bash
#!/usr/bin/env bash
# Preflight FIT/UNFIT paths against fake cgroup/proc trees — deterministic
# on any box (the real-devpod UNFIT observation is a manual exit-criterion
# step, not this test). Run standalone or via verify.sh --smoke.
set -euo pipefail
PF="$(dirname "$0")/preflight.sh"
fail() { echo "SELFTEST FAIL: $*" >&2; exit 1; }

mktree() { # <cpu.max content> <psi avg10> [cgroup rel path, default /podX]
  local root rel="${3:-/podX}"; root=$(mktemp -d)
  mkdir -p "$root/cg${rel}" "$root/proc/pressure"
  echo "0::${rel}" > "$root/proc/self_cgroup"   # see QHW_CGROUP_FILE below
  printf '%s\n' "$1" > "$root/cg${rel}/cpu.max"
  printf 'nr_periods 100\nnr_throttled 7\nthrottled_usec 0\n' > "$root/cg${rel}/cpu.stat"
  printf 'some avg10=%s avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n' \
    "$2" > "$root/proc/pressure/cpu"
  grep -m1 . /proc/cpuinfo >/dev/null  # sanity: real /proc exists
  printf 'vendor_id\t: FakeVendor\nmodel name\t: Fake CPU\n' > "$root/proc/cpuinfo"
  echo "$root"
}

run_pf() { # <root> — runs preflight against the fake tree, fast calibration
  QHW_CGROUP_ROOT="$1/cg" QHW_PROC_ROOT="$1/proc" \
  QHW_CGROUP_FILE="$1/proc/self_cgroup" \
  QHW_NPROC=16 QHW_CALIB_SECS=1 bash "$PF"
}

# FIT: unquota'd, quiet, enough cores, static cpu.stat (delta 0).
root=$(mktree "max 100000" "0.10")
out=$(run_pf "$root") || fail "expected FIT, got exit $? on: $out"
echo "$out" | grep -q "PREFLIGHT: FIT" || fail "missing FIT line: $out"
echo "$out" | grep -q "FakeVendor"    || fail "missing machine block: $out"

# UNFIT: the devpod signature — quota + pressure + too few cores.
root=$(mktree "800000 100000" "12.50")
if out=$(run_pf "$root" 2>&1); then fail "expected UNFIT to exit nonzero"; fi
out=$(QHW_CGROUP_ROOT="$root/cg" QHW_PROC_ROOT="$root/proc" \
      QHW_CGROUP_FILE="$root/proc/self_cgroup" \
      QHW_NPROC=8 QHW_CALIB_SECS=1 bash "$PF" 2>&1) && fail "UNFIT exited 0"
echo "$out" | grep -q "PREFLIGHT: UNFIT"      || fail "missing UNFIT line: $out"
echo "$out" | grep -q "cgroup quota"          || fail "quota probe silent: $out"
echo "$out" | grep -q "cpu pressure"          || fail "PSI probe silent: $out"
echo "$out" | grep -q "cores: 8"              || fail "core probe silent: $out"

# UNFIT: flat/root cgroup topology (0::/) — the walk must visit the single
# cpu.max exactly once, not double-count it (regression: 2x quota line and
# 2x-inflated throttled delta when rel="/" wasn't normalized to "").
root=$(mktree "800000 100000" "0.10" "/")
out=$(QHW_CGROUP_ROOT="$root/cg" QHW_PROC_ROOT="$root/proc" \
      QHW_CGROUP_FILE="$root/proc/self_cgroup" \
      QHW_NPROC=16 QHW_CALIB_SECS=1 bash "$PF" 2>&1) && fail "flat-root UNFIT exited 0"
echo "$out" | grep -q "PREFLIGHT: UNFIT" || fail "flat-root: missing UNFIT line: $out"
count=$(echo "$out" | grep -c "cgroup quota")
[ "$count" -eq 1 ] || fail "flat-root: expected exactly 1 quota line, got $count: $out"

# UNFIT: missing PSI file must be a reason, not a crash — the machine block
# (and a PREFLIGHT line) must still print.
root=$(mktree "max 100000" "0.10")
rm "$root/proc/pressure/cpu"
out=$(run_pf "$root" 2>&1) && fail "missing-PSI expected nonzero exit"
echo "$out" | grep -q "PREFLIGHT: UNFIT"            || fail "missing-PSI: no UNFIT line: $out"
echo "$out" | grep -qE "cpu pressure: .* missing"   || fail "missing-PSI: no missing-PSI reason: $out"
echo "$out" | grep -q "FakeVendor"                  || fail "missing-PSI: no machine block (crashed before printing?): $out"

echo "preflight-selftest: OK"
```

- [ ] **Step 2: Run to verify it fails**

Run: `bash scripts/quiet-hw/preflight-selftest.sh`
Expected: FAIL — `preflight.sh` does not exist.

- [ ] **Step 3: Write preflight.sh**

Create `scripts/quiet-hw/preflight.sh`:

```bash
#!/usr/bin/env bash
# M4b.7 environment-fitness preflight (spec §The Preflight): automates the
# probes M4b.1's amendment ran by hand, so fitness is asserted BEFORE any
# data exists. FIT → exit 0 + machine block; UNFIT → exit 1 listing every
# failed probe. Tunables (calibration points from the M4b.1 devpod, which
# must fail all three: cpu.max=800000 100000, PSI some avg10 11–15, +164
# throttled periods during one prefill):
#   QHW_MIN_CPUS (12)  QHW_PSI_MAX (1.0)  QHW_CALIB_SECS (10)
# Test seams: QHW_CGROUP_ROOT QHW_PROC_ROOT QHW_CGROUP_FILE QHW_NPROC.
set -euo pipefail
. "$(dirname "$0")/lib.sh"

CG="${QHW_CGROUP_ROOT:-/sys/fs/cgroup}"
PROC="${QHW_PROC_ROOT:-/proc}"
CGFILE="${QHW_CGROUP_FILE:-/proc/self/cgroup}"
MIN_CPUS="${QHW_MIN_CPUS:-12}"
PSI_MAX="${QHW_PSI_MAX:-1.0}"
CALIB_SECS="${QHW_CALIB_SECS:-10}"
NPROC="${QHW_NPROC:-$(nproc)}"

fails=()

# Probe 1 — core count.
if [ "$NPROC" -lt "$MIN_CPUS" ]; then
  fails+=("cores: $NPROC < required $MIN_CPUS")
fi

# Probe 2 — cgroup-v2 CPU quota anywhere up the hierarchy (the check that
# catches the devpod's 800000 100000).
rel=$(awk -F: '/^0::/ { print $3; exit }' "$CGFILE")
rel="${rel%/}"   # "/" → "" so the flat/root topology is walked exactly once
quota_summary="unquota'd"
path="$rel"
while :; do
  f="$CG$path/cpu.max"
  if [ -f "$f" ]; then
    read -r quota period < "$f"
    if [ "$quota" != "max" ]; then
      fails+=("cgroup quota: $f = '$quota ${period:-}' (must be 'max')")
      quota_summary="$quota/${period:-}"
    fi
  fi
  [ -z "$path" ] && break
  path="${path%/*}"
done

# Probe 3 — external CPU pressure (PSI).
psi=""
if [ -f "$PROC/pressure/cpu" ]; then
  psi=$(awk '/^some/ { sub(/.*avg10=/, ""); sub(/ .*/, ""); print; exit }' \
        "$PROC/pressure/cpu")
  if ! awk -v p="$psi" -v m="$PSI_MAX" 'BEGIN { exit !(p + 0 <= m + 0) }'; then
    fails+=("cpu pressure: some avg10 = $psi > $PSI_MAX")
  fi
else
  fails+=("cpu pressure: $PROC/pressure/cpu missing (cannot verify quiet)")
fi

# Probe 4 — throttling delta across an all-core calibration load (the
# direct version of M4b.1's +164-periods observation).
throttled_now() {
  local total=0 p="$rel" f n
  while :; do
    f="$CG$p/cpu.stat"
    if [ -f "$f" ]; then
      n=$(awk '/^nr_throttled/ { print $2; exit }' "$f")
      total=$((total + ${n:-0}))
    fi
    [ -z "$p" ] && break
    p="${p%/*}"
  done
  echo "$total"
}
before=$(throttled_now)
for _ in $(seq "$NPROC"); do
  (end=$((SECONDS + CALIB_SECS)); while [ "$SECONDS" -lt "$end" ]; do :; done) &
done
wait
after=$(throttled_now)
if [ "$after" -ne "$before" ]; then
  fails+=("throttling: nr_throttled +$((after - before)) during ${CALIB_SECS}s calibration load")
fi

machine_block
echo "probes: cpus=$NPROC quota=$quota_summary psi_some_avg10=${psi:-?} throttled_delta=$((after - before)) calib=${CALIB_SECS}s"

if [ "${#fails[@]}" -eq 0 ]; then
  echo "PREFLIGHT: FIT"
else
  echo "PREFLIGHT: UNFIT"
  printf ' - %s\n' "${fails[@]}"
  exit 1
fi
```

- [ ] **Step 4: Run the selftest to verify it passes**

Run: `bash scripts/quiet-hw/preflight-selftest.sh`
Expected: `preflight-selftest: OK`.

- [ ] **Step 5: Negative test on this very devpod (exit-criterion evidence)**

Run: `bash scripts/quiet-hw/preflight.sh; echo "exit=$?"`
Expected: `PREFLIGHT: UNFIT`, `exit=1`, with at least the `cgroup quota` reason showing `800000/100000` (pressure/throttling reasons may or may not fire depending on the moment — the quota one is deterministic here). Paste the output into the task report; the controller records it in the ledger as the exit-criterion demonstration. It is NOT a spec data point.

- [ ] **Step 6: Commit**

```bash
chmod +x scripts/quiet-hw/preflight.sh scripts/quiet-hw/preflight-selftest.sh
mise run test && devenv shell -- mise run lint
git add scripts/quiet-hw/preflight.sh scripts/quiet-hw/preflight-selftest.sh
git commit -m "scripts: quiet-hw fitness preflight + selftest (M4b.7)"
```

---

### Task 4: `gate-prefill-scaling.sh` (M4b.1) + `gate-decode-cap.sh` (M4b.5)

**Files:**
- Create: `scripts/quiet-hw/gate-prefill-scaling.sh`
- Create: `scripts/quiet-hw/gate-decode-cap.sh`

**Interfaces:**
- Consumes: `lib.sh` (`median`, `pct`, `decode_toks`, `machine_block`, `smoke_header`, `phys_cores`); `inferno bench --pp --tg --reps --threads --json` (JSON fields `inferno_pp_tok_s`, `inferno_tg_tok_s`, `llama_pp_tok_s`, `llama_tg_tok_s`); `inferno run --threads --max-tokens -p` (stdout `... | decode: N tok in T s (D tok/s)`); `INFERNO_DECODE_THREADS` (decode-cap override read at pool init, `inferno-core/src/lib.rs:111`).
- Produces: standalone gates invoked by `verify.sh` as `gate-<name>.sh <model.gguf>` with `QHW_OUT`/`QHW_SMOKE` honored.

- [ ] **Step 1: Write gate-prefill-scaling.sh**

```bash
#!/usr/bin/env bash
# M4b.7 gate 1 — M4b.1 exit criterion: prefill scaling ≥6x at t=12 vs t=1.
# `inferno bench` always emits llama-bench rows; they are recorded as
# environment corroboration (M4b.1-amendment style) — the evaluation uses
# the inferno rows only. Verdict destination: M4b.1 spec §Amendments
# (docs/superpowers/specs/2026-07-06-m4b1-threading-design.md).
# Usage: gate-prefill-scaling.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }
command -v jq >/dev/null || { echo "missing jq" >&2; exit 2; }

MODEL="${1:?usage: gate-prefill-scaling.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  THREADS="1 2"; PP=32; TG=8; REPS=1
else
  THREADS="1 2 4 8 12"; PP=512; TG=128; REPS=5
fi

smoke_header "gate-prefill-scaling (M4b.1 ≥6x @ t=12)"
machine_block
echo

for t in $THREADS; do
  cargo run --release -q -p inferno -- bench "$MODEL" \
    --pp "$PP" --tg "$TG" --reps "$REPS" --threads "$t" --json \
    > "$OUT/prefill-t$t.json"
done

ipp1=$(jq -r .inferno_pp_tok_s "$OUT/prefill-t1.json")
itg1=$(jq -r .inferno_tg_tok_s "$OUT/prefill-t1.json")
case "$ipp1$itg1" in *null*|"") ipp1=0 ;; esac
awk -v a="$ipp1" -v b="$itg1" 'BEGIN { exit !(a + 0 > 0 && b + 0 > 0) }' \
  || { echo "FATAL: t=1 baseline missing/zero in $OUT/prefill-t1.json" >&2; exit 1; }
echo "| t | pp tok/s | pp scale | tg tok/s | tg scale | llama pp (corrob.) | llama tg (corrob.) |"
echo "|---|---|---|---|---|---|---|"
scale12=""
for t in $THREADS; do
  j="$OUT/prefill-t$t.json"
  ipp=$(jq -r .inferno_pp_tok_s "$j"); itg=$(jq -r .inferno_tg_tok_s "$j")
  lpp=$(jq -r .llama_pp_tok_s "$j");   ltg=$(jq -r .llama_tg_tok_s "$j")
  spp=$(awk -v a="$ipp" -v b="$ipp1" 'BEGIN { printf "%.2f", a / b }')
  stg=$(awk -v a="$itg" -v b="$itg1" 'BEGIN { printf "%.2f", a / b }')
  echo "| $t | $ipp | ${spp}x | $itg | ${stg}x | $lpp | $ltg |"
  [ "$t" = 12 ] && scale12="$spp"
done
echo
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: evaluation skipped"
elif awk -v s="$scale12" 'BEGIN { exit !(s + 0 >= 6.0) }'; then
  echo "gate: prefill scale @ t=12 = ${scale12}x (target ≥6x) -> MET"
else
  echo "gate: prefill scale @ t=12 = ${scale12}x (target ≥6x) -> NOT MET"
  echo "note: on a MET=no result, take the M4b.1 spec's attribution fork (serial attention vs memory bandwidth) — see its Amendments."
fi
```

- [ ] **Step 2: Write gate-decode-cap.sh**

```bash
#!/usr/bin/env bash
# M4b.7 gate 2 — M4b.5 exit-criterion leg 2: decode-thread sweep. The
# shipped default cap clamp(active/3, 2, active) must meet-or-beat the best
# fixed cap, remove the high-thread regression, and leave t=1 decode
# unchanged. Sweeps INFERNO_DECODE_THREADS with rounds interleaved (rep-
# outer, cap-inner) per the standing M4b discipline. Verdict destination:
# M4b.5 spec §Amendments
# (docs/superpowers/specs/2026-07-08-m4b5-phase-aware-decode-threading-design.md).
# Usage: gate-decode-cap.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-decode-cap.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
PROMPT="The capital of France is"
PHYS=$(phys_cores)
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  CAPS="1 2"; REPS=1; MAXTOK=8
else
  CAPS=$(seq 1 "$PHYS" | tr '\n' ' '); REPS=3; MAXTOK=128
fi

smoke_header "gate-decode-cap (M4b.5 default-vs-best sweep)"
machine_block
echo "sweep: caps={$CAPS} + default + t1 | reps=$REPS (interleaved rounds) | max-tokens=$MAXTOK"
echo

one_run() { # <cap: number|default|t1> -> decode tok/s on stdout
  local threads=0 envset=()
  case "$1" in
    default) ;;                       # heuristic path: env unset
    t1)      threads=1 ;;             # t=1 decode-unchanged row
    *)       envset=(INFERNO_DECODE_THREADS="$1") ;;
  esac
  env "${envset[@]}" cargo run --release -q -p inferno -- run "$MODEL" \
    -p "$PROMPT" --max-tokens "$MAXTOK" --threads "$threads" 2>&1 \
    | tee -a "$OUT/decode-cap-runs.log" | decode_toks -
}

declare -A samples
for rep in $(seq "$REPS"); do
  for cap in $CAPS default t1; do
    tgs=$(one_run "$cap")
    [ -n "$tgs" ] || { echo "FATAL: no decode tok/s parsed (rep $rep cap $cap) — see $OUT/decode-cap-runs.log" >&2; exit 1; }
    samples[$cap]="${samples[$cap]:-} $tgs"
  done
done

echo "| cap | decode tok/s (median of $REPS) | per-rep |"
echo "|---|---|---|"
best_cap=""; best=0
for cap in $CAPS default t1; do
  med=$(median ${samples[$cap]})
  echo "| $cap | $med |${samples[$cap]} |"
  case "$cap" in default|t1) ;; *)
    if awk -v m="$med" -v b="$best" 'BEGIN { exit !(m > b) }'; then
      best="$med"; best_cap="$cap"
    fi ;;
  esac
done
def_med=$(median ${samples[default]})
echo
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: evaluation skipped"
else
  echo "knee (best fixed cap): $best_cap ($best tok/s median)"
  # Discipline: per-rep default/best ratios (same interleaved round), THEN median.
  read -ra def_arr <<< "${samples[default]}"
  read -ra best_arr <<< "${samples[$best_cap]}"
  ratios=""
  for i in $(seq 0 $((REPS - 1))); do
    ratios="$ratios $(pct "${def_arr[$i]}" "${best_arr[$i]}")"
  done
  echo "default clamp(active/3,2,active): $def_med tok/s median -> $(median $ratios)% vs best fixed (median of per-rep ratios)"
  echo "gate inputs (human verdict to M4b.5 Amendments): default meets-or-beats"
  echo "best-fixed? high-thread regression gone (compare cap=$PHYS row vs knee)?"
  echo "t=1 decode unchanged (t1 row vs prior recorded t=1)?"
fi
```

- [ ] **Step 3: Smoke both gates end-to-end on this box**

```bash
chmod +x scripts/quiet-hw/gate-prefill-scaling.sh scripts/quiet-hw/gate-decode-cap.sh
MODEL=$(devenv shell -- bash scripts/fetch-qwen-gguf.sh | tail -1)
QHW_SMOKE=1 devenv shell -- bash scripts/quiet-hw/gate-prefill-scaling.sh "$MODEL"
QHW_SMOKE=1 devenv shell -- bash scripts/quiet-hw/gate-decode-cap.sh "$MODEL"
```

Expected: each prints the SMOKE stamp first, the machine block, a filled table (2 thread counts / 2 caps + default + t1), and `SMOKE: evaluation skipped`. Exit 0. This is slow-ish (compiled-path runs); minutes, not seconds.

- [ ] **Step 4: Commit**

```bash
mise run test && devenv shell -- mise run lint
git add scripts/quiet-hw/gate-prefill-scaling.sh scripts/quiet-hw/gate-decode-cap.sh
git commit -m "scripts: quiet-hw gates 1-2, prefill scaling + decode cap (M4b.7)"
```

---

### Task 5: `gate-pf-dist.sh` (M4b.4)

**Files:**
- Create: `scripts/quiet-hw/gate-pf-dist.sh`

**Interfaces:**
- Consumes: Task 1's `INFERNO_PF_DIST` compile-time knob; `lib.sh` (`crit_mid_ns`, `median`, `pct`, `smoke_header`, `machine_block`); criterion bench ids `gemv/Q8_0/inferno-avx2/{rows}x{k}` and `gemv/Q4_K/inferno-avx2/{rows}x{k}` from `crates/inferno-kernels/benches/gemv.rs`.
- Produces: standalone gate for `verify.sh`.

- [ ] **Step 1: Write gate-pf-dist.sh**

```bash
#!/usr/bin/env bash
# M4b.7 gate 3 — M4b.4 deferred verdicts: PF_DIST keep/revert (0 vs 4) and
# sweep {2,4,8} for the q8_0 and q4_k AVX2 GEMV prefetch. Builds one bench
# binary per value up front (INFERNO_PF_DIST is a compile-time input), then
# interleaves runs rep-outer/value-inner so A/B pairs sit close in time.
# Ratios are per-rep vs the shipped v=4 binary, medianed across reps.
# Verdict destination: M4b.4 spec §Amendments
# (docs/superpowers/specs/2026-07-08-m4b4-decode-gemv-mlp-design.md) — the
# keep/revert call and the Task-3 (interleave) go/no-go stay human.
# Usage: gate-pf-dist.sh   (no model needed; env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }
command -v jq >/dev/null || { echo "missing jq" >&2; exit 2; }

OUT="${QHW_OUT:-$(mktemp -d)}"
VALUES="0 2 4 8"
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  REPS=1; FILTER='gemv/Q8_0/inferno-avx2/896x896$'; EXTRA=(--quick)
else
  REPS=3; FILTER='gemv/(Q8_0|Q4_K)/inferno-avx2/'; EXTRA=()
fi

smoke_header "gate-pf-dist (M4b.4 keep/revert + {2,4,8} sweep)"
machine_block
echo "values={$VALUES} reps=$REPS (interleaved; per-rep ratios vs v=4) filter=$FILTER"
echo

for v in $VALUES; do
  bin=$(INFERNO_PF_DIST=$v cargo bench -p inferno-kernels --bench gemv \
          --no-run --message-format=json 2>"$OUT/pf-build-$v.log" \
        | jq -r 'select(.reason == "compiler-artifact"
                        and .target.name == "gemv") | .executable' | tail -1)
  [ -n "$bin" ] && [ "$bin" != null ] || { echo "FATAL: no bench binary for INFERNO_PF_DIST=$v — see $OUT/pf-build-$v.log" >&2; exit 1; }
  cp "$bin" "$OUT/gemv-pf$v"
done

for rep in $(seq "$REPS"); do
  for v in $VALUES; do
    "$OUT/gemv-pf$v" --bench "${EXTRA[@]}" "$FILTER" \
      > "$OUT/pf$v-rep$rep.out" 2>&1
  done
done

shapes=$(crit_mid_ns "$OUT/pf4-rep1.out" 'inferno-avx2/' | awk '{ print $1 }')
[ -n "$shapes" ] || { echo "FATAL: no criterion times parsed — see $OUT/pf4-rep1.out" >&2; exit 1; }

echo "| bench id | v=0 vs 4 (median %) | v=2 vs 4 | v=8 vs 4 | (negative = faster than shipped v=4) |"
echo "|---|---|---|---|---|"
for id in $shapes; do
  row="| $id |"
  for v in 0 2 8; do
    diffs=""
    for rep in $(seq "$REPS"); do
      t4=$(crit_mid_ns "$OUT/pf4-rep$rep.out" "^${id}$" | awk '{ print $2 }')
      tv=$(crit_mid_ns "$OUT/pf$v-rep$rep.out" "^${id}$" | awk '{ print $2 }')
      [ -n "$t4" ] && [ -n "$tv" ] || { echo "FATAL: missing time for $id v=$v rep=$rep" >&2; exit 1; }
      diffs="$diffs $(pct "$tv" "$t4")"
    done
    row="$row $(median $diffs)% |"
  done
  echo "$row (n/a) |"
done
echo
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: evaluation skipped"
else
  echo "gate inputs (human verdict to M4b.4 Amendments): v=0 column decides"
  echo "keep/revert (v=0 faster => revert prefetch); best of {2,4,8} decides the"
  echo "distance; a >=5%-class win on any DRAM-bound shape is the signal that"
  echo "would authorize M4b.4 Task 3 (interleave)."
fi
```

- [ ] **Step 2: Smoke it end-to-end on this box**

```bash
chmod +x scripts/quiet-hw/gate-pf-dist.sh
QHW_SMOKE=1 devenv shell -- bash scripts/quiet-hw/gate-pf-dist.sh
```

Expected: four builds (each `INFERNO_PF_DIST` value rebuilds `inferno-kernels`), then one `--quick` rep of the single smoke shape per binary, then a table with one row (`gemv/Q8_0/inferno-avx2/896x896`) and three percentage columns, `SMOKE: evaluation skipped`, exit 0.

- [ ] **Step 3: Commit**

```bash
mise run test && devenv shell -- mise run lint
git add scripts/quiet-hw/gate-pf-dist.sh
git commit -m "scripts: quiet-hw gate 3, PF_DIST keep/revert + sweep (M4b.7)"
```

---

### Task 6: `gate-bench-protocol.sh` (M4a / v1 win) + `gate-intel-ab.sh` (M4b.6)

**Files:**
- Create: `scripts/quiet-hw/gate-bench-protocol.sh`
- Create: `scripts/quiet-hw/gate-intel-ab.sh`

**Interfaces:**
- Consumes: `lib.sh`; `inferno bench` (table and `--json` forms; JSON fields as in Task 4); `cpu_vendor` for Intel routing; git objects: arm commit `092b191` reachable from GitHub ref `refs/pull/11/head`; criterion ids `gemv/Q8_0/inferno-avx2/{shape}` and `gemv/Q8_0/reduce-unpack/{shape}` (the latter exists only in the cherry-picked tree).
- Produces: standalone gates for `verify.sh`. `gate-intel-ab.sh` exit codes: 0 = completed, **3 = SKIPPED (non-Intel)**, else failed. Flags: `--force-vendor` (smoke the cherry-pick plumbing on AMD), `--reps N` (default 3; use 6 on a deciding-shape straddle).

- [ ] **Step 1: Write gate-bench-protocol.sh**

```bash
#!/usr/bin/env bash
# M4b.7 gate 4 — the official M4a comparison protocol; the ONLY place the
# v1 win criterion ("beat llama.cpp prefill AND decode tok/s at its best
# thread count") can be judged. Runs the table form for the human record
# and the --json form for the evaluation, defaults pp=512 tg=128 reps=5
# threads=0 (physical cores), per the M4a spec. Verdict destination: M4a
# spec §Amendments (docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md).
# Usage: gate-bench-protocol.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }
command -v jq >/dev/null || { echo "missing jq" >&2; exit 2; }
command -v llama-bench >/dev/null || { echo "missing llama-bench (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-bench-protocol.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PP=32; TG=8; REPS=1; else PP=512; TG=128; REPS=5; fi

smoke_header "gate-bench-protocol (M4a protocol / v1 win criterion)"
machine_block
echo

cargo run --release -q -p inferno -- bench "$MODEL" \
  --pp "$PP" --tg "$TG" --reps "$REPS" --threads 0 \
  | tee "$OUT/bench-table.txt"
cargo run --release -q -p inferno -- bench "$MODEL" \
  --pp "$PP" --tg "$TG" --reps "$REPS" --threads 0 --json \
  > "$OUT/bench.json"

rpp=$(jq -r '.inferno_pp_tok_s / .llama_pp_tok_s' "$OUT/bench.json")
rtg=$(jq -r '.inferno_tg_tok_s / .llama_tg_tok_s' "$OUT/bench.json")
echo
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: evaluation skipped"
else
  printf "ratios (inferno/llama.cpp, from the independent --json run): pp %.2fx | tg %.2fx\n" "$rpp" "$rtg"
  if awk -v a="$rpp" -v b="$rtg" 'BEGIN { exit !(a > 1.0 && b > 1.0) }'; then
    echo "gate: v1 win criterion (pp > 1x AND tg > 1x) -> MET"
  else
    echo "gate: v1 win criterion (pp > 1x AND tg > 1x) -> NOT MET"
  fi
fi
```

- [ ] **Step 2: Write gate-intel-ab.sh**

```bash
#!/usr/bin/env bash
# M4b.7 gate 5 — M4b.6's deferred cross-vendor verdict: re-run the
# reduce-unpack A/B on an Intel box (the SKL µop model said wash, not the
# Zen 2 loss) before declaring the op-reduction lever dead cross-vendor.
# Restores the bench arm by cherry-picking 092b191 (lives only in PR #11's
# pre-squash history: refs/pull/11/head) into a scratch worktree, runs the
# per-process bitwise pre-check, then N interleaved reps, and applies the
# M4b.6 ship-gate arithmetic (fixed weights and conditions from that spec's
# amendments — never re-derived). Verdict destination: M4b.6 spec
# §Amendments (docs/superpowers/specs/2026-07-09-m4b6-decode-gemv-op-reduction-design.md).
# Exit: 0 completed, 3 SKIPPED (non-Intel), else failure.
# Usage: gate-intel-ab.sh [--force-vendor] [--reps N]   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

ARM=092b191
REPS=3; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --force-vendor) FORCE=1 ;;
    --reps) shift; REPS="${1:?--reps needs a value}" ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
OUT="${QHW_OUT:-$(mktemp -d)}"

smoke_header "gate-intel-ab (M4b.6 reduce-unpack cross-vendor A/B)"
machine_block

vendor=$(cpu_vendor)
if [ "$vendor" != GenuineIntel ] && [ "$FORCE" != 1 ]; then
  echo "SKIPPED: vendor is '$vendor', gate needs GenuineIntel (--force-vendor to smoke the plumbing)"
  exit 3
fi

REPO=$(git rev-parse --show-toplevel)
git -C "$REPO" fetch origin refs/pull/11/head
git -C "$REPO" rev-parse --verify --quiet "$ARM^{commit}" >/dev/null \
  || { echo "FATAL: $ARM not found after fetching refs/pull/11/head — if GitHub dropped the PR ref, re-transcribe the arm from the M4b.6 plan's Task 1 (see runbook)" >&2; exit 1; }

WT=$(mktemp -d)/m4b7-ab
git -C "$REPO" worktree add --detach "$WT" HEAD >/dev/null
trap 'git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true' EXIT
git -C "$WT" cherry-pick --no-commit "$ARM"
export CARGO_TARGET_DIR="$REPO/target"   # reuse dep builds across worktrees

if [ "${QHW_SMOKE:-0}" = 1 ]; then FILTER='gemv/Q8_0/(inferno-avx2|reduce-unpack)/896x896$'; EXTRA=(--quick); else FILTER='gemv/Q8_0/(inferno-avx2|reduce-unpack)/'; EXTRA=(); fi

echo "bitwise pre-check (arm vs library kernel, --test mode)…"
(cd "$WT" && cargo bench -p inferno-kernels --bench gemv -- 'gemv/Q8_0' --test) \
  > "$OUT/ab-test-mode.out" 2>&1 \
  || { echo "FATAL: bitwise pre-check failed — do not measure; see $OUT/ab-test-mode.out" >&2; exit 1; }

for rep in $(seq "$REPS"); do
  (cd "$WT" && cargo bench -p inferno-kernels --bench gemv -- "${EXTRA[@]}" "$FILTER") \
    > "$OUT/ab-rep$rep.out" 2>&1
done

MID_SHAPES="896x896 4864x896 896x4864"
declare -A wmed wall
straddle=0
shapes=$(crit_mid_ns "$OUT/ab-rep1.out" 'reduce-unpack/' | awk -F/ '{ print $NF }' | awk '{ print $1 }')
echo
echo "| shape | w per rep (%) | median w (%) | (w = 1 − t_unpack/t_base; positive = arm wins) |"
echo "|---|---|---|---|"
for shape in $shapes; do
  ws=""; pos=0; neg=0
  for rep in $(seq "$REPS"); do
    tb=$(crit_mid_ns "$OUT/ab-rep$rep.out" "inferno-avx2/${shape}\$" | awk '{ print $2 }')
    tu=$(crit_mid_ns "$OUT/ab-rep$rep.out" "reduce-unpack/${shape}\$" | awk '{ print $2 }')
    [ -n "$tb" ] && [ -n "$tu" ] || { echo "FATAL: missing time for $shape rep $rep" >&2; exit 1; }
    w=$(awk -v b="$tb" -v u="$tu" 'BEGIN { printf "%.2f", 100 * (1 - u / b) }')
    ws="$ws $w"
    awk -v w="$w" 'BEGIN { exit !(w > 0) }' && pos=$((pos + 1)) || neg=$((neg + 1))
  done
  wmed[$shape]=$(median $ws); wall[$shape]="$ws"
  [ "$pos" -gt 0 ] && [ "$neg" -gt 0 ] && straddle=1
  echo "| $shape |$ws | ${wmed[$shape]} | |"
done
echo
if [ "${QHW_SMOKE:-0}" = 1 ]; then
  echo "SMOKE: evaluation skipped"
  exit 0
fi

# Ship-gate arithmetic — fixed by the M4b.6 amendments, never re-derived.
c1=0
for shape in $MID_SHAPES; do
  allpos=1
  for w in ${wall[$shape]}; do awk -v w="$w" 'BEGIN { exit !(w > 0) }' || allpos=0; done
  c1=$((c1 + allpos))
done
c2=PASS
for shape in $shapes; do
  awk -v m="${wmed[$shape]}" 'BEGIN { exit !(m < -3.0) }' && c2=FAIL
done
proj=$(awk -v a="${wmed[151936x896]:-0}" -v b="${wmed[896x4864]:-0}" \
           -v c="${wmed[4864x896]:-0}"   -v d="${wmed[896x896]:-0}" \
       'BEGIN { printf "%.2f", 0.270*a + 0.211*b + 0.407*c + 0.087*d }')
echo "condition 1 (w_r>0 every rep on >=2 of 3 mid shapes): $c1 of 3 -> $([ "$c1" -ge 2 ] && echo MET || echo FAILED)"
echo "condition 2 (no shape median w < -3%): $c2"
echo "projected_decode_win = ${proj}% (weights .270/.211/.407/.087 per M4b.6 amendment)"
[ "$straddle" = 1 ] && echo "WARNING: a shape's w_r straddles 0 — if it is a deciding shape, re-run with --reps 6 before recording."
echo "verdict (human, to M4b.6 Amendments): SHIP iff condition 1 MET and condition 2 PASS."
```

- [ ] **Step 3: Smoke both on this box**

```bash
chmod +x scripts/quiet-hw/gate-bench-protocol.sh scripts/quiet-hw/gate-intel-ab.sh
MODEL=$(devenv shell -- bash scripts/fetch-qwen-gguf.sh | tail -1)
QHW_SMOKE=1 devenv shell -- bash scripts/quiet-hw/gate-bench-protocol.sh "$MODEL"
# Vendor routing (this box is AMD): must print SKIPPED and exit 3.
devenv shell -- bash scripts/quiet-hw/gate-intel-ab.sh; echo "exit=$?"
# Cherry-pick plumbing, vendor check bypassed:
QHW_SMOKE=1 devenv shell -- bash scripts/quiet-hw/gate-intel-ab.sh --force-vendor
```

Expected: protocol gate prints its table + `SMOKE: evaluation skipped`; the bare intel-ab run prints `SKIPPED: vendor is 'AuthenticAMD'…` with `exit=3`; the forced run fetches the PR ref, cherry-picks `092b191` cleanly (it reverts cleanly because `benches/gemv.rs` is byte-identical to the pre-arm state), passes the bitwise `--test` pre-check, runs one `--quick` rep on the smoke shape, prints the w table, `SMOKE: evaluation skipped`, exit 0, and removes its worktree.

- [ ] **Step 4: Commit**

```bash
mise run test && devenv shell -- mise run lint
git add scripts/quiet-hw/gate-bench-protocol.sh scripts/quiet-hw/gate-intel-ab.sh
git commit -m "scripts: quiet-hw gates 4-5, bench protocol + Intel A/B (M4b.7)"
```

---

### Task 7: `verify.sh` orchestrator + mise task + runbook + full smoke

**Files:**
- Create: `scripts/quiet-hw/verify.sh`
- Modify: `mise.toml` (append after `[tasks.bench]`, line ~67)
- Create: `docs/runbooks/quiet-hw-verification.md`

**Interfaces:**
- Consumes: everything above; exit-code contract (preflight nonzero = UNFIT; intel-ab 3 = SKIPPED).
- Produces: `mise run verify-quiet-hw -- <model.gguf> [--smoke] [--out-dir D] [--force-vendor]`; results tree `target/quiet-hw/<timestamp>/{preflight.out,gate-*.out,summary.md,…}`.

- [ ] **Step 1: Write verify.sh**

```bash
#!/usr/bin/env bash
# M4b.7 orchestrator: fitness preflight, then the five deferred-verdict
# gates, everything tee'd under one timestamped results dir. UNFIT preflight
# is a hard stop unless --smoke (which stamps every output NON-RECORDABLE).
# A gate that fails is recorded FAILED and the pass continues — on rare
# quiet hardware, partial data beats an aborted run. Scripts never write to
# docs/; paste verdicts into the owning spec's Amendments by hand (see
# docs/runbooks/quiet-hw-verification.md).
# Usage: verify.sh <model.gguf> [--smoke] [--out-dir D] [--force-vendor]
#                                [--i-know-what-im-doing]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (run inside 'devenv shell')" >&2; exit 2; }

MODEL="${1:?usage: verify.sh <model.gguf> [--smoke] [--out-dir D] [--force-vendor] [--i-know-what-im-doing]}"
shift
SMOKE=0; OUTDIR=""; OVERRIDE=0; AB_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --out-dir) shift; OUTDIR="${1:?--out-dir needs a value}" ;;
    --force-vendor) AB_ARGS+=(--force-vendor) ;;
    --i-know-what-im-doing) OVERRIDE=1 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
REPO=$(git rev-parse --show-toplevel)
OUT="${OUTDIR:-$REPO/target/quiet-hw/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
export QHW_OUT="$OUT" QHW_SMOKE="$SMOKE"

# Selftests first — cheap, and smoke is the pass that runs them (spec §Testing).
bash "$HERE/lib-selftest.sh"
bash "$HERE/preflight-selftest.sh"

declare -A status
if bash "$HERE/preflight.sh" 2>&1 | tee "$OUT/preflight.out"; then
  status[preflight]=FIT
else
  status[preflight]=UNFIT
  if [ "$SMOKE" = 1 ]; then
    echo "(smoke mode: continuing on unfit hardware; all output NON-RECORDABLE)"
  elif [ "$OVERRIDE" = 1 ]; then
    # Spec §Risks escape hatch for a preflight false-positive the operator
    # has judged wrong: every output gets the UNFIT-OVERRIDE stamp so
    # provenance survives into whatever gets recorded.
    export QHW_OVERRIDE=1
    status[preflight]=UNFIT-OVERRIDE
    echo "(UNFIT-OVERRIDE: operator forced the run; every output is stamped)"
  else
    echo "ABORT: preflight UNFIT and not --smoke; no gate may run (spec: UNFIT = hard stop)." >&2
    exit 1
  fi
fi

run_gate() { # <name> <cmd...>
  local name="$1"; shift
  echo "=== gate: $name ==="
  local rc=0
  "$@" 2>&1 | tee "$OUT/gate-$name.out" || rc=$?
  case "$rc" in
    0) status[$name]=PASS ;;
    3) status[$name]=SKIPPED ;;
    *) status[$name]=FAILED ;;
  esac
}

run_gate prefill-scaling bash "$HERE/gate-prefill-scaling.sh" "$MODEL"
run_gate decode-cap      bash "$HERE/gate-decode-cap.sh" "$MODEL"
run_gate pf-dist         bash "$HERE/gate-pf-dist.sh"
run_gate bench-protocol  bash "$HERE/gate-bench-protocol.sh" "$MODEL"
run_gate intel-ab        bash "$HERE/gate-intel-ab.sh" "${AB_ARGS[@]}"

{
  [ "$SMOKE" = 1 ] && echo "### SMOKE — NON-RECORDABLE (plumbing check on unfit hardware; never paste into a spec) ###"
  [ "${QHW_OVERRIDE:-0}" = 1 ] && echo "### UNFIT-OVERRIDE (preflight failed; operator forced the run — record the override alongside any data) ###"
  echo "# quiet-hw verification pass — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  machine_block
  echo
  echo "| stage | status |"
  echo "|---|---|"
  echo "| preflight | ${status[preflight]} |"
  for g in prefill-scaling decode-cap pf-dist bench-protocol intel-ab; do
    echo "| $g | ${status[$g]} |"
  done
  echo
  echo "PASS = script completed and printed its table — the VERDICTS are human;"
  echo "paste each gate's output into its owning spec's Amendments per"
  echo "docs/runbooks/quiet-hw-verification.md."
} | tee "$OUT/summary.md"
echo "results: $OUT"

for g in prefill-scaling decode-cap pf-dist bench-protocol; do
  [ "${status[$g]}" = PASS ] || exit 1
done
[ "${status[intel-ab]}" = FAILED ] && exit 1
exit 0
```

- [ ] **Step 2: Add the mise task**

Append to `mise.toml` after the `[tasks.bench]` block:

```toml
[tasks.verify-quiet-hw]
description = "M4b.7 quiet-hardware verification pass: fitness preflight + the five deferred-verdict gates (run inside devenv shell on quiet hardware; --smoke = plumbing check on any box; see docs/runbooks/quiet-hw-verification.md): mise run verify-quiet-hw -- <model.gguf> [--smoke]"
run = "bash scripts/quiet-hw/verify.sh"
```

- [ ] **Step 3: Write the runbook**

Create `docs/runbooks/quiet-hw-verification.md`:

```markdown
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
```

- [ ] **Step 4: Full smoke pass end-to-end (exit-criterion demonstration)**

```bash
chmod +x scripts/quiet-hw/verify.sh
MODEL=$(devenv shell -- bash scripts/fetch-qwen-gguf.sh | tail -1)
devenv shell -- mise run verify-quiet-hw -- "$MODEL" --smoke --force-vendor
echo "exit=$?"; ls target/quiet-hw/*/
```

Expected: selftests print OK; preflight prints UNFIT (this devpod) and the pass continues under smoke; all five gates run (intel-ab under `--force-vendor` exercises the cherry-pick); `summary.md` opens with the SMOKE stamp and lists preflight UNFIT + five gate rows; every `gate-*.out` file exists; exit 0. Also verify the non-smoke hard stop:

```bash
devenv shell -- mise run verify-quiet-hw -- "$MODEL"; echo "exit=$?"
```

Expected: preflight UNFIT → `ABORT: preflight UNFIT…`, `exit=1`, no `gate-*.out` files in the new results dir.

- [ ] **Step 5: Commit**

```bash
mise run test && devenv shell -- mise run lint
git add scripts/quiet-hw/verify.sh mise.toml docs/runbooks/quiet-hw-verification.md
git commit -m "scripts: quiet-hw orchestrator, mise task, runbook (M4b.7)"
```

---

## Milestone Close (controller, after final review)

- Confirm `git diff <branch-base> -- crates/inferno-graph/src/tolerance.rs` is empty.
- The exit-criterion evidence (devpod-UNFIT preflight output, smoke-pass summary, PF_DIST build-matrix results) goes in the PR description and ledger — NOT into any spec's Amendments (no data point exists; this was a tooling milestone).
- The five verdicts remain open; the runbook is their pointer. M4b.7 closes as tooling-complete.
