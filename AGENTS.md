# Agent instructions — inferno

Everything derivable from code is not repeated here. Read
[ARCHITECTURE.md](ARCHITECTURE.md) for the crate map and
[docs/superpowers/specs/2026-07-04-inferno-v1-design.md](docs/superpowers/specs/2026-07-04-inferno-v1-design.md)
for the v1 design.

## Non-obvious constraints

- **Workflows are mise tasks** (`mise tasks`): use `mise run test` / `lint` /
  `audit` / `fuzz` — CI runs the same names. Don't hand-roll cargo invocations
  in docs or CI.
- **Toolchain:** rust + dev tools are mise-pinned (`mise.toml`); native deps
  (LLVM, llama.cpp) come ONLY from `devenv.nix`. The devenv LLVM
  major.minor there must match the `inkwell` feature flag in
  `inferno-codegen` (M3+) exactly — currently `llvm22-1` (`Cargo.toml`)
  against LLVM 22.1.8 (`devenv.nix`'s `pkgs.llvmPackages_22`,
  `LLVM_SYS_221_PREFIX`). Bumping one without the other breaks the build.
- **`inferno-formats` must stay `#![forbid(unsafe_code)]`** and every parser
  read bounded — model files are untrusted input (see
  [docs/threat-model.md](docs/threat-model.md)). Touching parser code means
  running `mise run fuzz -- gguf_parse` / `-- safetensors_parse` locally.
- **`ModelDesc` is format-agnostic:** never let a downstream crate learn
  which file format a model came from.
- **Tensor shapes are row-major, outermost first** everywhere in inferno;
  GGUF stores dims reversed and the GGUF parser normalizes them on ingest.
- **Snapshots (insta):** review with `cargo insta review`; never blind-accept.
- Fixture files under `tests/fixtures/` and `fuzz/corpus/` are generated —
  regenerate with `cargo run -p inferno-formats --example gen_fixtures`,
  don't hand-edit.
- **Rope style is coupled to weight layout:** GGUF llama-arch files carry
  *row-permuted* Q/K weights (Interleaved rope); MLX/HF files are unpermuted
  (HalfSplit). `HyperParams::rope_style` records which; the fixture
  differential (`inferno-graph/tests/differential.rs`) guards the coupling.
  Never "simplify" one side without the other.
- **Embedded and JSON tokenizer fixtures must stay equivalent:**
  `fixtures::tiny_vocab()` feeds both the GGUF metadata and
  `mlx/tokenizer.json`; the BPE equivalence property tests depend on it.
- **`LOGIT_TIE_EPSILON`** (`inferno-graph/src/tolerance.rs`) is tuned against
  the gap distributions printed by `mise run differential` — adjust it with
  observed data, never to make a red nightly green without understanding the
  divergence.
- **`inferno-kernels` and `inferno-core` are the only crates allowed
  `unsafe`** (`inferno-kernels`: intrinsics + the C ABI; `inferno-core`,
  M3+: mmap + `dlopen` + calling compiled entry points by raw fn pointer).
  Both opt out of the workspace's `unsafe_code = "deny"` lint with their own
  `[lints.rust]` table. Scalar and SIMD kernel variants must stay
  bit-identical — the rig asserts exact equality.
- **Kernel perf numbers come only from `mise run bench-kernels`** inside the
  devenv shell on quiet hardware; CI runners are noise. Record data points in
  the M2 spec's amendments section.
- **`inferno bench` (vs llama.cpp) is a manual protocol**, never a CI gate:
  quiet hardware, devenv shell, release build; record each report in the
  M4a spec's Amendments section
  (`docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md`) and
  never edit a recorded data point.
- **The nightly speedup gate (`bench-compiled`) is pinned to `--threads 1`
  on purpose**: it measures codegen quality vs the interpreter, and
  threading would hide codegen regressions behind parallelism. Never "fix"
  a red nightly by unpinning it.
- **`gemv_rel_tol`** follows the same rule as `LOGIT_TIE_EPSILON`: tuned
  against observed error distributions (the rig's ignored `observed_error_*`
  diagnostics), never to make a red test green.
- **Object-emit + `dlopen` is the only compile path (M3+):** `inferno
  compile`/`run` emit an LLVM module to a native object, link it to
  `model.so`, and `dlopen` it from the on-disk artifact cache — there is no
  in-memory JIT path.
- **KV is stored as `f32` in M3** (not F16), specifically to keep the
  compiled-vs-interpreter differential free of an extra rounding term; don't
  "optimize" it to F16 without re-deriving tolerances (M4 work).
- **After touching codegen op lowerings**, run
  `cargo test -p inferno-codegen --test differential` (the
  compiled-vs-interpreter correctness gate) and
  `cargo test -p inferno-core --test artifact` (the artifact-level
  differential); never loosen `logits_abs_tol` to make either green.
- **Decode threading is uncapped (M4b.10, was phase-capped in M4b.5):** the
  compiled decode path (`inferno_par_gemv`) shards over the full
  `active_threads`, same as prefill (`inferno_par_gemm`). Override with
  `INFERNO_DECODE_THREADS=N`. M4b.5 had capped decode at `clamp(active/3, 2,
  active)` on a bandwidth-knee hypothesis; three quiet-hw sessions (6336Y,
  E-2388G, socket-pinned 8352Y) found no cliff — uncapped within 0.9–3.2% of
  the best fixed cap — so M4b.10 removed the cap (see its design's Amendments).
  Row-sharding stays bit-neutral regardless of lane count (`shard_table` keeps
  each row on one lane); never treat a thread-count change as a numeric change. Prefill attention (M4b.8) dispatches per
  tile through `inferno_par_attention`, sharding the tile's tokens with
  align-1 shards at full `active` — the decode cap never applies to it,
  and `m <= 1` calls bypass the pool entirely. Since M4b.9 the serial tail
  (rmsnorm/rope/add/swiglu/bias/embed, KV-append, activation-quantize panel
  fill) is token-sharded too: codegen outlines each per-token body into a
  private `tok_body.*` function dispatched through `inferno_par_token_loop`
  (opaque-ctx ABI, align-1 shards, `m <= 1` direct) — outlined bodies must
  never reference caller SSA values or call the profiler, and the
  `kv_append` dispatch always joins before the attention read is issued.
  Decode attention is head-sharded through `inferno_par_attention_heads`
  (M4b.11) under the same `INFERNO_DECODE_THREADS` bound; the head-span
  kernels must stay bit-identical to the whole-call kernels (the rig's
  hspan tiling tests are the guard).
  The pool's `pool-profile` cargo feature is the M4b.12 dispatch-split
  instrument (off in every shipping/bench build; quiet-hw gate scripts
  build with it), and `INFERNO_ATTN_SHARDS` is its probe-only shard-count
  override — neither is a tuning surface.
  M4b.12's attribution closed all-STOP: publish/wake/scratch-alloc are each
  sub-0.2% of the decode wall on both quiet boxes — decode-attention time is
  in the hspan kernel itself (plus drain-side lane imbalance at 16c), so
  don't reach for dispatch-side levers there (spec §Amendments 2026-07-17).
  Prefill attention is query-blocked below `inferno_par_attention` (M4b.14):
  the pool's `run_attn_span` makes ONE `AttnBlockFn` call per lane shard
  (`inferno_attention_f32_{scalar,avx2}_qblock`, 14-arg ABI with
  `pos0`/`m_block`/row strides; `HOST_ABI_VERSION` "8"), streaming each
  visible K/V vector once per block instead of once per token. The block
  kernel is bit-identical to the per-token kernel for every block length —
  the rig's `attention_qblock_*` proptests are the guard, and
  `m_block == 1` bit-equals the per-token path (the `m == 1` prefill and
  decode routes depend on it). The per-token kernels stay (rig oracle).
  `inferno-kernels`' `attn-subprofile` cargo feature is the M4b.14
  scores/softmax/output rdtsc instrument (off in every shipping/bench
  build; only `gate-prefill-attn-split.sh` enables it) — not a tuning
  surface. M4b.14 closed all-STOP: post-blocking, t=1 prefill is
  matmul-shaped again (~69% at stream rate) and on the 8c box even
  deleting the whole attention bracket can't reach pp 1.0x — no further
  single-bracket prefill lever; see the M4b.14 spec §Amendments.
- **`mise run metal` spends real money** (PhoenixNAP bare metal, hourly):
  operator-driven only, never CI. After any interrupted session run
  `mise run metal-gc` — EXIT traps don't survive killed terminals. The
  ISA table (`scripts/metal/cpu-features.json`) is verified against
  `/proc/cpuinfo` on every provision; on drift, fix the table in a
  commit, never override (see docs/runbooks/metal.md).
