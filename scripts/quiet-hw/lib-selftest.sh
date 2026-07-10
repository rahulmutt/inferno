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
