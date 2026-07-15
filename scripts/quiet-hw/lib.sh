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

# llama_bench_pp_tg <llama-bench -o json file> — "pp_tok_s tg_tok_s" from
# the -p row (n_prompt>0, n_gen==0) and -n row (n_gen>0). -e + error():
# schema drift or a missing row fails the caller loudly (same discipline as
# cli/src/llama_bench.rs, which parses the same schema strictly).
llama_bench_pp_tg() {
  jq -er '
    def one(f): [.[] | select(f) | .avg_ts]
      | if length == 1 then .[0] else error("expected exactly one matching row") end;
    "\(one(.n_prompt > 0 and .n_gen == 0)) \(one(.n_prompt == 0 and .n_gen > 0))"' "$1"
}

# fmax <a> <b> — the larger of two floats, verbatim (no reformatting).
fmax() { awk -v a="$1" -v b="$2" 'BEGIN { print (a + 0 > b + 0) ? a : b }'; }

# _cores_from_lscpu_p [node] — count unique (CORE,SOCKET) pairs from
# `lscpu -p=CORE,SOCKET,NODE` fed on stdin, optionally restricted to a NODE
# id. No stdin data rows (only/no comment lines) prints 0 — the caller falls
# back to nproc. Pure text processing so it is testable offline with
# synthetic input (see lib-selftest.sh).
_cores_from_lscpu_p() {
  local node="${1:-}"
  awk -F, -v node="$node" '
    !/^#/ && NF {
      if (node != "" && $3 != node) next
      key = $1 "," $2
      if (!seen[key]++) n++
    }
    END { print n + 0 }'
}

# phys_cores — physical core count (sweep upper bound for gate-decode-cap).
# When QHW_NUMA_NODE is set, counts only cores on that node — this is what
# makes a NUMA-pinned session's phys_cores (and every provenance line
# derived from it) honest instead of describing the whole machine while only
# half of it is bound. Never fails under `set -euo pipefail`: an lscpu that
# exists but emits no data rows (sandboxes, odd VMs) falls back to nproc.
phys_cores() {
  local n=0
  if command -v lscpu >/dev/null; then
    n=$( (lscpu -p=CORE,SOCKET,NODE 2>/dev/null || true) \
         | _cores_from_lscpu_p "${QHW_NUMA_NODE:-}")
  fi
  if [ "${n:-0}" -ge 1 ]; then echo "$n"; else nproc; fi
}

# cap_grid <max> — decode-cap sweep values: every cap up to 16, then step 4.
# Bounds session time on many-core boxes (M4b.10) while keeping full
# resolution where every recorded knee has landed (8..16). `max` always
# appears, exactly once.
cap_grid() {
  local max="$1" i out=""
  for i in $(seq 1 "$max"); do
    if [ "$i" -le 16 ] || [ $((i % 4)) -eq 0 ] || [ "$i" -eq "$max" ]; then
      out="$out $i"
    fi
  done
  echo "${out# }"
}

# numa_wrap — the numactl prefix pinning CPUs *and* memory to QHW_NUMA_NODE,
# or nothing when unset. Used to take a NUMA-free single-socket point on a
# dual-socket box (M4b.10: d2.c5.large is 2x32c).
numa_wrap() {
  [ -n "${QHW_NUMA_NODE:-}" ] || return 0
  echo "numactl --cpunodebind=${QHW_NUMA_NODE} --membind=${QHW_NUMA_NODE}"
}

# numa_require — a pinned session must be able to pin. Call this at gate start,
# NOT from inside numa_wrap: the gates expand `$(numa_wrap)` unquoted into the
# command line, so a numa_wrap that failed would expand to nothing and the gate
# would measure UNPINNED while still printing "numa: pinned to node N" — a
# silently mislabeled data point, which is worse than a crash. Fail before a
# single token is measured instead.
numa_require() {
  [ -n "${QHW_NUMA_NODE:-}" ] || return 0
  command -v numactl >/dev/null 2>&1 || {
    echo "FATAL: QHW_NUMA_NODE=$QHW_NUMA_NODE but numactl is not on PATH — run inside 'devenv shell'. A pinned session must never fall back to an unpinned run." >&2
    exit 2
  }
  numactl --hardware 2>/dev/null | grep -q "^node ${QHW_NUMA_NODE} cpus:" || {
    echo "FATAL: QHW_NUMA_NODE=$QHW_NUMA_NODE but this box has no NUMA node ${QHW_NUMA_NODE} (see 'numactl --hardware')" >&2
    exit 2
  }
  # membind is a capability, not just a binary: set_mempolicy is gated behind
  # CAP_SYS_NICE under Docker's default seccomp, so in a container without it
  # --membind dies mid-sweep with "set_mempolicy: Operation not permitted"
  # (M4b.10 session C, 2026-07-15) — after the box is paid for. Prove it here.
  numactl --membind="${QHW_NUMA_NODE}" true 2>/dev/null || {
    echo "FATAL: QHW_NUMA_NODE=$QHW_NUMA_NODE but 'numactl --membind' is denied — set_mempolicy needs CAP_SYS_NICE (add '\"runArgs\": [\"--cap-add=SYS_NICE\"]' to .devcontainer/devcontainer.json). A pinned session must not silently drop membind." >&2
    exit 2
  }
}
