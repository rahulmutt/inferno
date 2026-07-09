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
