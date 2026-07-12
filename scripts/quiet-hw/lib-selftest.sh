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

# llama_bench_pp_tg: pp/tg extraction from llama-bench -o json (golden
# fixture shared with cli/src/llama_bench.rs — one schema, two parsers).
fixture="$(dirname "$0")/../../cli/tests/fixtures/llama-bench.json"
expect "llama_bench_pp_tg" "$(llama_bench_pp_tg "$fixture")" "486.4 84.0"
# A json missing one of the two rows must fail loudly, not emit a blank.
if out=$(echo '[]' | llama_bench_pp_tg - 2>/dev/null); then
  fail "llama_bench_pp_tg on empty json should fail (got '$out')"
fi

expect "fmax a wins" "$(fmax 403.53 310.30)" "403.53"
expect "fmax b wins" "$(fmax 84 486.4)" "486.4"

# median with no args must fail loudly, not return 0.
if out=$(median 2>/dev/null); then fail "median with no args should return nonzero (got '$out')"; fi

# phys_cores must survive an lscpu that emits only comments (pipefail trap)
# and fall back to nproc. Stub lscpu via PATH.
stub=$(mktemp -d)
printf '#!/usr/bin/env bash\necho "# comment only"\n' > "$stub/lscpu"
chmod +x "$stub/lscpu"
n=$(PATH="$stub:$PATH" bash -euo pipefail -c ". '$(dirname "$0")/lib.sh'; phys_cores")
[ "${n:-0}" -ge 1 ] || fail "phys_cores with data-less lscpu should fall back to nproc (got '$n')"
# Same data-less lscpu with QHW_NUMA_NODE set — the fallback must still fire
# (no data rows to filter in the first place).
n=$(PATH="$stub:$PATH" QHW_NUMA_NODE=0 bash -euo pipefail -c ". '$(dirname "$0")/lib.sh'; phys_cores")
[ "${n:-0}" -ge 1 ] || fail "phys_cores(node set) with data-less lscpu should fall back to nproc (got '$n')"
rm -rf "$stub"

# _cores_from_lscpu_p — node-aware core counting, the helper phys_cores
# derives from. This box is single-node (phys_cores() above already proves
# the unfiltered, real-hardware path returns this box's true physical core
# count), so the NODE filter itself can only be exercised with synthetic
# lscpu -p output: a dual-socket/dual-node machine, 4 physical cores per
# node, each core listed twice (hyperthread siblings) to also prove the
# (CORE,SOCKET) de-dup still holds under a NODE filter.
lscpu_dual=$(mktemp)
cat > "$lscpu_dual" <<'EOF'
# Core,Socket,Node
0,0,0
1,0,0
2,0,0
3,0,0
0,0,0
1,0,0
2,0,0
3,0,0
0,1,1
1,1,1
2,1,1
3,1,1
0,1,1
1,1,1
2,1,1
3,1,1
EOF
expect "cores unfiltered is the whole machine" \
  "$(_cores_from_lscpu_p < "$lscpu_dual")" "8"
expect "cores filtered to node 0" \
  "$(_cores_from_lscpu_p 0 < "$lscpu_dual")" "4"
expect "cores filtered to node 1" \
  "$(_cores_from_lscpu_p 1 < "$lscpu_dual")" "4"

# phys_cores end to end: QHW_NUMA_NODE must change the answer against the
# same stubbed dual-node lscpu (this is Finding 1 — a pinned session must
# report the pinned node's core count, not the whole machine's).
stub2=$(mktemp -d)
{
  echo '#!/usr/bin/env bash'
  echo 'cat <<"DATA"'
  cat "$lscpu_dual"
  echo 'DATA'
} > "$stub2/lscpu"
chmod +x "$stub2/lscpu"
expect "phys_cores unpinned matches whole-machine count" \
  "$(PATH="$stub2:$PATH" bash -euo pipefail -c ". '$(dirname "$0")/lib.sh'; phys_cores")" "8"
expect "phys_cores pinned to node 0" \
  "$(PATH="$stub2:$PATH" QHW_NUMA_NODE=0 bash -euo pipefail -c ". '$(dirname "$0")/lib.sh'; phys_cores")" "4"
expect "phys_cores pinned to node 1" \
  "$(PATH="$stub2:$PATH" QHW_NUMA_NODE=1 bash -euo pipefail -c ". '$(dirname "$0")/lib.sh'; phys_cores")" "4"
rm -rf "$stub2"
rm -f "$lscpu_dual"

# cap_grid — fine-grained to 16, step 4 above; always includes 1 and max.
expect "cap_grid 8"  "$(cap_grid 8)"  "1 2 3 4 5 6 7 8"
expect "cap_grid 16" "$(cap_grid 16)" "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
expect "cap_grid 32" "$(cap_grid 32)" "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 20 24 28 32"
expect "cap_grid 1"  "$(cap_grid 1)"  "1"
# 18 is not a multiple of 4 above 16 — max must still appear, exactly once.
expect "cap_grid 18" "$(cap_grid 18)" "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18"

# numa_wrap — empty unless QHW_NUMA_NODE is set.
expect "numa_wrap unset" "$(numa_wrap)" ""
expect "numa_wrap set"   "$(QHW_NUMA_NODE=0 numa_wrap)" "numactl --cpunodebind=0 --membind=0"
expect "numa_wrap node1" "$(QHW_NUMA_NODE=1 numa_wrap)" "numactl --cpunodebind=1 --membind=1"

echo "lib-selftest: OK"
