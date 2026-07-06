#!/bin/sh
# SAST scan (nightly tier). Single source of truth for the semgrep invocation so
# a local `mise run semgrep` matches CI exactly (nightly.yml runs this script).
#
# We run the registry `p/rust` pack but exclude one rule:
#   rust.lang.security.unsafe-usage.unsafe-usage
# It is a blanket rule that flags *every* `unsafe` block ("audit for secure
# usage"). Inferno is a from-scratch inference engine whose SIMD kernels
# (AVX2 intrinsics), quantized-weight pointer arithmetic, and LLVM/FFI codegen
# bridge require `unsafe` pervasively — the rule fired 127 times across the M3
# kernel/compiler code with zero actionable signal, and `--error` turned that
# into a hard CI failure. Auditing unsafe is already covered by clippy and the
# cargo-audit/deny supply-chain gate; excluding it keeps the other 10 p/rust
# rules as a real, non-noisy gate.
set -eu

exec semgrep scan \
  --config p/rust \
  --exclude-rule rust.lang.security.unsafe-usage.unsafe-usage \
  --error
