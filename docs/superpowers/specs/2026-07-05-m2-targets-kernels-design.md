# M2 — Targets + Kernels Design

**Date:** 2026-07-05
**Status:** Approved design, pre-implementation
**Parent:** [Inferno v1 design](2026-07-04-inferno-v1-design.md), milestone M2

## Goal

Prove the kernel bet before the compiler exists. Two standalone crates:
`inferno-target` (what machine is this?) and `inferno-kernels` (hand-tuned
AVX2 quantized-GEMV microkernels behind the fixed C ABI that M3 codegen will
call by symbol). A reference-comparison rig ties every kernel to the M1
scalar oracle, and criterion benches compare throughput side by side with the
devenv-pinned llama.cpp's ggml routines — the early signal, demanded by the
v1 risk section, that tells us whether the performance bet is failing while
it is still cheap to change course.

## Scope decisions

| Decision | Choice |
|---|---|
| Crates created | `inferno-target` and `inferno-kernels`; no changes to graph/runtime/cli beyond docs |
| ISA levels | AVX2+FMA (x86-64-v3) only. Neither the dev machine (Zen 2) nor `ubuntu-latest` runners reliably execute AVX-512; no kernels we cannot run. `Isa::X86_64v4` is defined but nothing populates it |
| Weight dtypes | Q4_K, Q8_0 (the quant formats real GGUF models use) + F32 as the trivial baseline that validates the rig itself. F16/BF16 kernels wait for a model that needs them |
| Activations | Quantized on the fly at the kernel boundary, mirroring ggml's pairings: Q4_K×Q8_K, Q8_0×Q8_0, integer SIMD dot products inside. The only realistic path to ggml-parity throughput; the per-format tolerances absorb the added numeric error |
| Integration | None. Kernels have no consumer until M3; the interpreter stays a pure scalar oracle. M2 ships crates + rig + benches only |
| Hardware detection | Linux (`cpuid` + `/sys`) + named TOML profiles. macOS `sysctl` waits for the v2 Apple Silicon work when it can be tested |
| llama.cpp comparison | Bench-only FFI against the devenv-pinned `libggml`, same buffers, same criterion groups |
| Out of scope | AVX-512, macOS detection, GEMM/prefill tiles, attention/norm/rope kernels (M3 generates those as LLVM IR), interpreter or CLI changes, threading inside kernels |

## Structural approach

GEMV-first microkernels with repacked weights. Batch = 1 decode — the metric
v1 wins on — makes every matmul a GEMV, and the win available at this layer
is layout: weights repacked at pack time into each kernel's preferred
block-interleaved strip layout so the inner loop streams contiguously. This
is the project's core thesis (layout specialization as a compile-time
transformation) exercised at the smallest possible surface, and the
pack-function + symbol ABI is exactly the durable M3 interface.

Alternatives rejected:

- *Mirror ggml's `vec_dot` over file-layout weights:* simplest and perfectly
  apples-to-apples, but forfeits the layout advantage that is the whole bet —
  it would measure whether we can rewrite ggml's kernels as well as ggml, and
  M3 would need a second ABI round anyway.
- *Full BLIS-style GEMM tiles now:* most future-proof for prefill, but the
  machinery idles until M3/M4 while decode is GEMV-shaped. YAGNI.

## `inferno-target`

`TargetDesc` is plain serde-able data — the same struct whether detected or
loaded from a profile; that equivalence is the future cross-compile
interface.

Fields:

- `isa: Isa` — enum at kernel-dispatch granularity: `X86_64v3` (AVX2+FMA),
  `X86_64v4` (defined, unused in M2), later aarch64 variants.
- `features: BTreeSet<Feature>` — flags outside the level (e.g. `f16c`) so
  profiles can describe real machines precisely.
- `caches: Vec<CacheLevel>` — level, size, line size, shared-by-cores.
- `topology: CoreTopology` — physical/logical core counts, SMT flag.
  P/E-core split deferred until a machine that has it.
- `page_size: u64`
- `memory_bw_class: Option<BwClass>` — profile-only; nothing detects it,
  the M3 planner may consume it.

**Detection is layered for testability** (v1 testing strategy: "target
detection parsing against captured cpuid/sysctl fixtures"). Pure functions
parse captured inputs — the `/sys/devices/system/cpu` tree is passed as a
root path so tests point at checked-in fixture trees from real machines — and
a thin live layer gathers those inputs (`is_x86_feature_detected!` for ISA,
`sysconf` for page size). Unit tests cover the parsers against a fixture tree
captured from the dev Ryzen 9 3900; one integration test asserts live
detection succeeds and yields a coherent struct.

**Named profiles** are TOML files embedded via `include_str!`, starting with
`ryzen-3900.toml` captured from the dev machine. `TargetDesc::detect()` and
`TargetDesc::from_profile("ryzen-3900")` return the same type. The
equivalence invariant is tested two ways: TOML round-trip (serialize any
detected desc → parse → equal) in the blocking tier, and
detect-on-this-machine == shipped profile as a nightly-only, machine-specific
test.

**Errors:** unknown profile → error listing available profiles; unsupported
OS/arch → error suggesting a named profile; malformed `/sys` → error naming
the offending path, never silent defaults.

## `inferno-kernels`

**ABI.** One `#[no_mangle] extern "C"` symbol per (op × dtype × ISA) — the
contract M3 codegen calls by name. Each weight dtype ships a triple:

- `pack_*` — repack file-order weight bytes (from `read_tensor_bytes`) into
  the kernel's preferred layout: strips of 8 rows interleaved at block
  granularity so the GEMV inner loop streams contiguously. The layout has a
  documented ID baked into the symbol name (`..._rs8`); M3's planner packs at
  compile time and the artifact records which layout it holds. Pack is safe,
  bounds-checked Rust.
- `quantize_row_*` — f32 activations → the integer format the weight dtype
  pairs with (Q8_K for Q4_K, Q8_0 for Q8_0). **Q8_K lives in
  `inferno-kernels`, never in `inferno-formats::DType`** — activation-side
  quantization is a kernel implementation detail, not a weight file dtype.
- `gemv_*` — computes output rows `[row_start, row_end)` of
  `y = W · x_quantized`. The row range is deliberate: M3 partitions rows
  across threads by calling the same symbol with disjoint ranges. No
  threading inside kernels.

F32 gets only `pack` (trivial strip reorder) + `gemv` (pure FMA, no
activation quant) — the rig's baseline.

**ISA variants.** Every kernel ships AVX2
(`#[target_feature(enable = "avx2,fma")]`) and a scalar fallback
implementing the *same* block-accumulation semantics: exact i32 dot products
within blocks, per-block scale-and-add in the same order. Scalar-vs-AVX2
comparisons therefore use a tight ~1e-6 relative tolerance, while
kernel-vs-oracle uses the per-format tolerances. Scalar fallbacks also let
the whole suite run on non-AVX2 hardware.

**Dispatch.** A safe registry, `kernels_for(dtype, isa) -> Option<KernelSet>`
(pack + quantize_row + gemv + layout ID), refuses to hand out AVX2 kernels
unless the CPU has them — the only place runtime feature detection happens.

**Error handling split.** The raw `extern "C"` symbols are unchecked by
design: documented pointer/length/alignment preconditions, no validation in
the hot path (M3 codegen guarantees the contracts by construction). Every
other caller — tests, benches, M3's planner — goes through the safe
`KernelSet` wrappers, which validate buffer lengths, block alignment of K,
and the 32-byte alignment of packed buffers, returning `Result`.

**Unsafe policy.** `unsafe` confined to intrinsics bodies and the C-ABI
boundary; every unsafe fn documents its contract; packed buffers are 32-byte
aligned by construction; `deny(unsafe_op_in_unsafe_fn)` crate-wide.

## Testing — the reference-comparison rig

Follows the M1 pattern: proptest invariants against the trusted oracle,
hand-computed unit anchors, fast enough for the blocking tier. The oracle
chain is `inferno-formats::dequant` + the f32 scalar reference matmul
(`inferno-graph` as dev-dependency).

Properties per dtype:

1. **Pack inverse:** a test-only `unpack` proves repacking is a pure block
   permutation — localizes layout bugs away from math bugs.
2. **`quantize_row` round-trip:** dequantized activation error bounded like
   the existing `roundtrip_rel_tol`; Q8_K block sums verified exactly (the
   Q4_K kernel's min-term correction depends on them).
3. **GEMV vs oracle:** random weights (packed from random f32) × random
   activations; kernel output within a new `gemv_rel_tol(dtype)` added to
   `inferno-graph/src/tolerance.rs` — the single tolerance home — tuned from
   observed error distributions (activation quantization adds error beyond
   weight round-trip), per the existing observed-data-only tolerance rule.
4. **Range partitioning:** GEMV over `[0, M)` bit-identical to concatenated
   GEMVs over any split of the range — the property M3's thread partitioning
   relies on.
5. **Scalar vs AVX2:** same random inputs, ~1e-6 relative tolerance.

Edge cases pinned by unit tests: row counts not a multiple of the 8-row
strip, K at exactly one block, empty row range as a no-op, zero and max-scale
blocks. NaN/Inf activations are a documented precondition violation — kernels
don't check; the hot path stays branch-free.

AVX2 properties run whenever the CPU supports it — true on the dev machine
and `ubuntu-latest`, so the blocking tier exercises real SIMD. Proptest case
counts are sized to keep PR wall-clock inside the existing ≤5-minute budget;
nightly runs enlarged case counts plus the machine-specific detect==profile
test.

## Benchmarks & the llama.cpp comparison

Criterion benches in `inferno-kernels/benches/`, one group per
(dtype × kernel), reporting GB/s of packed weights streamed (GEMV is
memory-bound; that is the honest metric) alongside GFLOPS. Shapes are real
Llama-family GEMV shapes — hidden×hidden, hidden×ffn, and vocab-logits rows
for Qwen2.5-0.5B (the nightly differential model) and Llama-3-8B dims — the
shapes M3 will generate calls for.

The ggml comparison is a cargo feature `ggml-compare` on `inferno-kernels`,
enabled only by benches — never by default, so shipping builds contain zero
FFI. A build script locates the devenv-pinned llama.cpp's `libggml` via an
env var exported by `devenv.nix`; the bench declares `extern "C"` prototypes
for the matching routines (`ggml_vec_dot_q4_K_q8_K`, `ggml_vec_dot_q8_0_q8_0`,
`quantize_row_q8_K`) and runs them on identical buffers in the same criterion
groups. Known risk: the pinned build may not export those internal symbols;
fallback is timing a `ggml_mul_mat` graph on identical shapes — slightly more
overhead attributed to ggml, still per-shape and repeatable.

Task and protocol: `mise run bench-kernels`, manual/local only — no CI
benching; shared-runner perf numbers are noise, and noisy numbers are worse
than none. Like the nightly-differential convention, the first real data
points are recorded in the milestone docs.

**Exit criterion for the risk check:** the AVX2 kernels are at or approaching
parity with ggml's equivalents on the bench shapes. The v1 thesis says the
*win* comes from M3's specialization — kernels must be competitive, not
miraculous. Falling well short of parity is the early failing-bet signal and
must be understood before M3 starts.

## Repo integration

- `ARCHITECTURE.md`: `inferno-target` and `inferno-kernels` move from
  planned to present, with the boundary rules (Q8_K is kernel-internal;
  layout ID is part of the symbol name; kernels are single-threaded,
  row-range partitioned).
- `AGENTS.md` gains the non-obvious constraints: activation-quant formats
  never enter `inferno-formats::DType`; kernel perf numbers come only from
  `mise run bench-kernels` on quiet hardware; `gemv_rel_tol` follows the
  observed-data-only tolerance rule.
- `README.md` status → M2.
- `devenv.nix`: export the llama.cpp lib path env var for `ggml-compare`.
- CI: no new workflows — `mise run test` / `lint` are workspace-wide and
  pick up both crates; nightly gains the detect==profile test and enlarged
  proptest runs.

## Milestone exit

- `TargetDesc::detect()` returns a correct description of the dev machine,
  equal to the shipped `ryzen-3900.toml` profile.
- Q4_K, Q8_0, F32 AVX2 GEMV kernels pass the full rig in the blocking tier.
- `mise run bench-kernels` produces side-by-side inferno-vs-ggml numbers;
  first data points recorded in docs, at or approaching parity (see the risk
  exit criterion above).

## Amendments (2026-07-05, during planning and implementation)

- **ggml comparison mechanism:** the pinned llama.cpp exports the needed
  kernels from its per-arch CPU backends (`bin/libggml-cpu-<arch>.so`), not
  from `libggml.so`. The bench `dlopen`s `libggml-cpu-haswell.so` (AVX2+FMA)
  via `$INFERNO_GGML_CPU_LIB` instead of link-time FFI; the `ggml_mul_mat`
  fallback was unnecessary. Verified: all five symbols export.
- **detect==profile placement:** GitHub runners are not the dev machine, so
  the equivalence test is gated on `INFERNO_EXPECT_PROFILE` (vacuous when
  unset) rather than nightly-scheduled; nightly CI instead runs the full
  suite with `PROPTEST_CASES=1024`.
- **`pack_*` is safe Rust, not a C symbol** — its only caller (M3 planner)
  is Rust. `quantize_row_*`/`gemv_*` remain `extern "C"`.
- **ISA variants are bit-identical**, not ~1e-6-close: integer block dots are
  exact and the f32 combine order is fixed. The rig asserts exact equality.
- **`pack_q8_0_rs8` clamps weight bytes −128 → −127** so the AVX2 sign-trick
  stays exact on hostile files (ggml's quantizer never emits −128).
- **Benches report GB/s only** (criterion `Throughput::Bytes` on the weight
  stream — the metric that matters for a memory-bound GEMV). GFLOPS is
  derivable as `2·rows·k / time` and was dropped rather than double-reported.
- **Oracle rig determinism (commit cc072c9):** the spec/plan's kernel-vs-oracle
  comparison originally fed the oracle raw f32 activations; measured
  activation-quantization noise (Q8_0 6.25e-2 rel @500k seeds, Q4_K tail
  ~0.142 @3M seeds — small-k/small-|y| worst) exceeds any meaningful constant
  tolerance, making the rig proptests unfixably flaky. The oracle-match tests
  now decode the kernel's own q8a/q8k activation buffer and feed that to the
  oracle: both sides consume identical quantized weights AND activations, so
  `gemv_rel_tol` bounds only combine-order/fma rounding and was retuned
  ~2000× tighter from observed data (Q8_0 1e-5, observed 2.384e-6; Q4_K 4e-5,
  observed 9.239e-6). End-to-end activation-quant noise remains covered by
  act.rs round-trip tests and measured by the rig's ignored
  `observed_error_*` diagnostics (raw-f32 comparison, sweeping the property
  shape distribution).
- **Strip-parallel AVX2 tuning (commit 4d430d8):** initial bench parity vs
  ggml fell below the spec bar on k>=4096 shapes (Q8_0 0.35–0.74×, Q4_K
  0.25–0.61×); the plan-anticipated optimizations (process whole strips per
  pass; batch integer reductions — one transpose-reduction per 8 lanes /
  vector-domain scale accumulation) landed as a Task 8 follow-up, preserving
  the bitwise scalar/AVX2 contract. Result: Q8_0 1.09–1.72× (above parity),
  Q4_K 0.73–0.80× (remaining gap is shuffle-port-bound integer work; closing
  it needs VNNI/AVX-512, out of M2 scope).

### First bench data points (dev Ryzen 9 3900, 2026-07-05)

Criterion GB/s, midpoint estimates, `mise run bench-kernels` inside the
devenv shell on quiet hardware (raw log: `/tmp/bench-kernels-m2.txt`). Note:
inferno's byte basis is the packed weight image, ggml's is its own file
image (F32 identical; quantized dtypes ~5.6–5.9% apart), so the GB/s columns
across engines aren't directly comparable — the ratio column is computed
from wall-clock time (`t_ggml / t_avx2`; >1× means inferno-avx2 is faster)
on identical `rows × k`, which is basis-independent.

| dtype | shape (rows x k) | scalar GiB/s | avx2 GiB/s | ggml GiB/s | avx2:ggml (time) |
|-------|-----------------|-------------:|-----------:|-----------:|------------------:|
| F32  | 4096x4096   | 1.77 | 20.07 | 19.74 | 1.02x |
| Q8_0 | 896x896     | 10.34 | 43.74 | 23.99 | 1.72x |
| Q8_0 | 4864x896    | 10.27 | 43.28 | 24.01 | 1.70x |
| Q8_0 | 896x4864    | 10.94 | 43.57 | 25.16 | 1.64x |
| Q8_0 | 151936x896  | 5.62 | 16.43 | 13.35 | 1.16x |
| Q8_0 | 4096x4096   | 3.91 | 18.51 | 16.02 | 1.09x |
| Q8_0 | 14336x4096  | 3.65 | 16.57 | 13.75 | 1.14x |
| Q4_K | 4096x4096   | 2.46 | 14.90 | 18.98 | 0.74x |
| Q4_K | 14336x4096  | 2.28 | 10.14 | 12.65 | 0.76x |
| Q4_K | 4096x14336  | 1.98 | 10.71 | 12.71 | 0.80x |
| Q4_K | 128256x4096 | 2.29 | 9.63 | 12.57 | 0.73x |

F32 and Q8_0 (small-k, cache-resident, and large DRAM-bound shapes) are at
or above parity with ggml; Q4_K clears the ~0.7× approaching-parity bar on
every k>=4096 shape but remains compute-bound behind ggml's tuned
scale-folded accumulation (see the strip-parallel amendment above).
