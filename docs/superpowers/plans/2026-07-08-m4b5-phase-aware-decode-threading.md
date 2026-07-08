# M4b.5 Phase-Aware Decode Threading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cap the compiled decode path's row-sharding at a tuned, bandwidth-saturating thread count while leaving prefill at full cores, removing the proven high-thread decode regression — bit-neutrally and with no ABI/codegen change.

**Architecture:** Add one `decode_cap` knob to the process-global `inferno-pool`. The decode dispatcher (`inferno_par_gemv`) shards over `min(active_threads, decode_cap)`; the prefill dispatcher (`inferno_par_gemm`) is untouched. `inferno-core` sets the cap default at pool init from a bandwidth-knee heuristic, overridable by `INFERNO_DECODE_THREADS`. Because the cap only regroups output rows into shards — never changes per-row math — every existing bit-identity guarantee holds.

**Tech Stack:** Rust (workspace crates `inferno-pool`, `inferno-core`), `cargo nextest`, `mise` tasks.

## Global Constraints

- **Bit-neutrality is mandatory.** Thread/shard count must never change output bits; `shard_table` computes each output row entirely within one lane. Assert exact equality (`to_bits`), never tolerance.
- **No tolerance constant may be loosened.** No `logits_abs_tol` / `gemv_rel_tol` edit, no `observed_error_*` sweep — the kernels produce identical bits.
- **No ABI/codegen change.** `inferno_par_gemv` / `inferno_par_gemm` `extern "C"` signatures stay identical; **no `HOST_ABI_VERSION` bump**; no recompile of cached artifacts required.
- **`inferno_par_gemm` (prefill) is untouched.** Only the decode dispatch site changes.
- **No new `unsafe`.** All new code (atomic load/store, `min`, env parsing, clamping) is safe.
- **Workflows are mise tasks.** Full suite: `mise run test`. Targeted runs use `cargo nextest run -p <crate> ...` as shown per task.
- **Heuristic default (verbatim from spec):** `decode_cap = clamp(active/3, 2, active)`, implemented as `(active / 3).max(2).min(active)` to avoid a `clamp` low>high panic when `active == 1`. Env override: `INFERNO_DECODE_THREADS=N`. Final default value is deferred to the quiet-hardware sweep (Task 6); this is the reversible starting hypothesis.

---

### Task 1: Pool `decode_cap` knob + capped decode dispatch

**Files:**
- Modify: `crates/inferno-pool/src/pool.rs` (`Shared` struct ~line 112; `Pool::new` ~line 154; new accessors after `active_threads` ~line 210; `par_gemv` shard-count line 236; unit tests in `mod tests` ~line 449)

**Interfaces:**
- Consumes: existing `Pool` / `Shared` / `shard_table`.
- Produces:
  - `Pool::set_decode_threads(&self, n: usize)` — stores `n.clamp(1, capacity)` into `decode_cap`.
  - `Pool::decode_threads(&self) -> usize` — loads `decode_cap`.
  - `decode_cap` defaults to `capacity` (i.e. no cap) so untouched behavior is byte- and thread-count-identical to today: `min(active, capacity) == active`.

- [ ] **Step 1: Write the failing test**

Add to `crates/inferno-pool/src/pool.rs` inside `mod tests` (after `zero_rows_is_a_noop`, ~line 497):

```rust
    #[test]
    fn decode_cap_defaults_to_capacity() {
        let pool = Pool::new(8);
        assert_eq!(pool.decode_threads(), 8);
    }

    #[test]
    fn set_decode_threads_clamps_to_1_capacity() {
        let pool = Pool::new(8);
        pool.set_decode_threads(0);
        assert_eq!(pool.decode_threads(), 1);
        pool.set_decode_threads(999);
        assert_eq!(pool.decode_threads(), 8);
        pool.set_decode_threads(4);
        assert_eq!(pool.decode_threads(), 4);
    }

    #[test]
    fn decode_cap_bounds_shard_count_but_not_result() {
        // Cap below active must still produce the exact serial expectation:
        // capping only regroups rows into fewer shards.
        let pool = Pool::new(8);
        pool.set_decode_threads(2);
        for rows in [1, 7, 8, 9, 64, 1000, 1024] {
            assert_eq!(dispatch(&pool, rows, 3), expected(rows, 3), "rows={rows}");
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo nextest run -p inferno-pool -E 'test(decode_cap) + test(set_decode_threads)'`
Expected: FAIL — `no method named decode_threads`/`set_decode_threads found for struct Pool` (compile error).

- [ ] **Step 3: Add the `decode_cap` field to `Shared`**

In `crates/inferno-pool/src/pool.rs`, in `struct Shared` (~line 112), add after the `active` field:

```rust
    /// Decode-phase parallelism cap (≤ capacity); `Pool::set_decode_threads`.
    /// `par_gemv` shards over `min(active, decode_cap)` so decode stops past
    /// its bandwidth knee while prefill (`par_gemm`) keeps full `active`.
    decode_cap: AtomicUsize,
```

In `Pool::new` (~line 154), in the `Shared { ... }` initializer, add after `active: AtomicUsize::new(capacity),`:

```rust
            decode_cap: AtomicUsize::new(capacity),
```

- [ ] **Step 4: Add the accessors**

In `crates/inferno-pool/src/pool.rs`, immediately after `active_threads` (~line 210), add:

```rust
    /// Cap decode-phase (`par_gemv`) parallelism to `n.clamp(1, capacity)`
    /// lanes. Prefill (`par_gemm`) is unaffected. Defaults to `capacity`
    /// (no cap); `inferno-core` lowers it to the bandwidth-knee heuristic.
    pub fn set_decode_threads(&self, n: usize) {
        self.shared
            .decode_cap
            .store(n.clamp(1, self.capacity), Ordering::Relaxed);
    }

    pub fn decode_threads(&self) -> usize {
        self.shared.decode_cap.load(Ordering::Relaxed)
    }
```

- [ ] **Step 5: Apply the cap at the decode dispatch site**

In `crates/inferno-pool/src/pool.rs`, in `par_gemv`, change line 236 from:

```rust
        let active = self.active_threads();
```

to:

```rust
        // Decode is bandwidth-bound: cap below prefill's full-core count so
        // sharding stops at its bandwidth knee (M4b.5). `par_gemm` is not capped.
        let active = self.active_threads().min(self.decode_threads());
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo nextest run -p inferno-pool`
Expected: PASS — the three new tests plus every existing pool test (default cap == capacity keeps all prior behavior identical).

- [ ] **Step 7: Commit**

```bash
git add crates/inferno-pool/src/pool.rs
git commit -m "feat(pool): decode_cap knob; par_gemv shards over min(active, decode_cap)"
```

---

### Task 2: Global decode-threads setter

**Files:**
- Modify: `crates/inferno-pool/src/lib.rs` (after `set_global_active_threads` ~line 55)
- Test: `crates/inferno-pool/tests/global.rs`

**Interfaces:**
- Consumes: `Pool::set_decode_threads` (Task 1), `GLOBAL` (`lib.rs`).
- Produces: `inferno_pool::set_global_decode_threads(n: usize) -> bool` — caps the global pool's decode parallelism; returns `false` (no-op) if `init_global` has not run. Mirrors `set_global_active_threads`.

- [ ] **Step 1: Write the failing test**

Inspect the existing style first: `sed -n '1,40p' crates/inferno-pool/tests/global.rs` (it initializes the global pool via `init_global`). Append this test to `crates/inferno-pool/tests/global.rs`:

```rust
#[test]
fn set_global_decode_threads_reports_init_state() {
    // Before any init in a fresh test process this could be either state
    // depending on test ordering within the binary; assert the post-init
    // contract explicitly by initializing first.
    inferno_pool::init_global(4).unwrap();
    assert!(inferno_pool::set_global_decode_threads(2));
}
```

Note: `global.rs` runs as its own test binary, so `init_global(4)` here does not race the other crates' pools. If the existing file already calls `init_global` with a different count, reuse that same count in this test instead of `4` (a mismatched re-init returns `AlreadyInitialized`). Check with `grep -n init_global crates/inferno-pool/tests/global.rs` and match it.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo nextest run -p inferno-pool -E 'test(set_global_decode_threads_reports_init_state)'`
Expected: FAIL — `cannot find function set_global_decode_threads in crate inferno_pool`.

- [ ] **Step 3: Add the global setter**

In `crates/inferno-pool/src/lib.rs`, after `set_global_active_threads` (~line 55), add:

```rust
/// Cap the global pool's decode-phase (`inferno_par_gemv`) parallelism
/// (clamped to `1..=capacity`). Prefill (`inferno_par_gemm`) is unaffected.
/// Returns `false` (and does nothing) if [`init_global`] has not run.
pub fn set_global_decode_threads(n: usize) -> bool {
    match GLOBAL.get() {
        Some(p) => {
            p.set_decode_threads(n);
            true
        }
        None => false,
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo nextest run -p inferno-pool -E 'test(set_global_decode_threads_reports_init_state)'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-pool/src/lib.rs crates/inferno-pool/tests/global.rs
git commit -m "feat(pool): set_global_decode_threads global setter"
```

---

### Task 3: Bit-identity across decode caps (real kernels)

**Files:**
- Modify: `crates/inferno-pool/tests/par_rig.rs` (append a test near the other `*_thread_count_is_bit_invisible` tests)

**Interfaces:**
- Consumes: `Pool::set_decode_threads` (Task 1), existing rig helpers `prep`, `serial`, `pooled`.
- Produces: locks the spec's core correctness claim — decode output is byte-identical for every `decode_cap`.

- [ ] **Step 1: Write the failing test**

Append to `crates/inferno-pool/tests/par_rig.rs`:

```rust
/// M4b.5: decode_cap must be bit-invisible. Fix a 12-lane pool, sweep the
/// decode cap 1..=12, and require every capped dispatch to match one direct
/// serial kernel call exactly — capping only regroups rows into shards.
#[test]
fn q8_0_decode_cap_is_bit_invisible() {
    let dtype = DType::Q8_0;
    let kernel = inferno_kernels::inferno_gemv_q8_0_rs8_scalar;
    let (rows, k) = (1003usize, 64usize);
    let (w, xq) = prep(&dtype, rows, k, 0xfeed_beef);
    let want = serial(kernel, &w, &xq, rows, k);
    let pool = Pool::new(12);
    for cap in 1..=12usize {
        pool.set_decode_threads(cap);
        let got = pooled(&pool, kernel, &w, &xq, rows, k);
        for (i, (g, s)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                s.to_bits(),
                "cap={cap} row {i}: {g} != {s}"
            );
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo nextest run -p inferno-pool -E 'test(q8_0_decode_cap_is_bit_invisible)'`
Expected: FAIL — compile error `no method named set_decode_threads` **only if Task 1 is absent**. With Task 1 present this test compiles and PASSES immediately (the mechanism is already bit-neutral). That is the intended outcome: it is a regression lock, not a red-then-green cycle. Record that it passes.

- [ ] **Step 3: Confirm it passes (regression lock)**

Run: `cargo nextest run -p inferno-pool -E 'test(q8_0_decode_cap_is_bit_invisible)'`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/inferno-pool/tests/par_rig.rs
git commit -m "test(pool): decode_cap is bit-invisible (q8_0, cap 1..=12)"
```

---

### Task 4: Core wiring — heuristic default + env override

**Files:**
- Modify: `crates/inferno-core/src/lib.rs` (`compiled_backend` ~line 83; add a private helper + `#[cfg(test)]` unit tests)

**Interfaces:**
- Consumes: `inferno_pool::set_global_decode_threads` (Task 2), `self.threads` (already resolved to `physical_cores.max(1)` in `Engine::new`).
- Produces: private `fn decode_cap(active: usize, override_env: Option<&str>) -> usize` — returns a parsed positive `INFERNO_DECODE_THREADS` value, else `(active / 3).max(2).min(active)`. Called from `compiled_backend` with `std::env::var("INFERNO_DECODE_THREADS").ok()`.

- [ ] **Step 1: Write the failing test**

At the bottom of `crates/inferno-core/src/lib.rs`, add (or extend an existing) test module:

```rust
#[cfg(test)]
mod decode_cap_tests {
    use super::decode_cap;

    #[test]
    fn heuristic_is_third_clamped_to_2_and_active() {
        assert_eq!(decode_cap(12, None), 4); // 12/3 = 4
        assert_eq!(decode_cap(32, None), 10); // 32/3 = 10
        assert_eq!(decode_cap(2, None), 2); // 2/3=0 -> max(2) -> min(2) = 2
        assert_eq!(decode_cap(1, None), 1); // 1/3=0 -> max(2) -> min(1) = 1
    }

    #[test]
    fn env_override_wins_when_a_positive_integer() {
        assert_eq!(decode_cap(12, Some("1")), 1);
        assert_eq!(decode_cap(12, Some("8")), 8);
        assert_eq!(decode_cap(12, Some(" 6 ")), 6); // trimmed
    }

    #[test]
    fn env_override_falls_back_to_heuristic_when_invalid() {
        assert_eq!(decode_cap(12, Some("garbage")), 4);
        assert_eq!(decode_cap(12, Some("0")), 4); // 0 rejected
        assert_eq!(decode_cap(12, Some("")), 4);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo nextest run -p inferno-core -E 'test(decode_cap)'`
Expected: FAIL — `cannot find function decode_cap in this scope` (compile error).

- [ ] **Step 3: Add the helper**

In `crates/inferno-core/src/lib.rs`, add this free function near the top-level items (module scope, not inside `impl`):

```rust
/// Resolve the decode-phase thread cap. An explicit `INFERNO_DECODE_THREADS`
/// override wins when it parses to a positive integer (the pool re-clamps it
/// to `[1, capacity]`); otherwise the bandwidth-knee heuristic
/// `clamp(active/3, 2, active)` — written `.max(2).min(active)` so `active==1`
/// yields `1` instead of panicking. `active` is the engine's resolved thread
/// count (physical cores by default). Final default is deferred to the
/// M4b.5 quiet-hardware sweep; this is the reversible starting hypothesis.
fn decode_cap(active: usize, override_env: Option<&str>) -> usize {
    if let Some(v) = override_env.and_then(|s| s.trim().parse::<usize>().ok()) {
        if v >= 1 {
            return v;
        }
    }
    (active / 3).max(2).min(active)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo nextest run -p inferno-core -E 'test(decode_cap)'`
Expected: PASS.

- [ ] **Step 5: Call it at pool init**

In `crates/inferno-core/src/lib.rs`, in `compiled_backend` (~line 87-88), after the existing two pool lines:

```rust
        inferno_pool::init_global(self.threads)?;
        inferno_pool::set_global_active_threads(self.threads);
```

add:

```rust
        // M4b.5: decode is bandwidth-bound — cap its row-sharding below full
        // cores so it stops at its knee; prefill keeps every core. Env
        // `INFERNO_DECODE_THREADS` overrides the heuristic.
        let env = std::env::var("INFERNO_DECODE_THREADS").ok();
        inferno_pool::set_global_decode_threads(decode_cap(self.threads, env.as_deref()));
```

- [ ] **Step 6: Run the crate's tests + full suite**

Run: `cargo nextest run -p inferno-core`
Expected: PASS (including the artifact differential — the cap is bit-neutral).

Run: `mise run test`
Expected: PASS — whole workspace green.

- [ ] **Step 7: Commit**

```bash
git add crates/inferno-core/src/lib.rs
git commit -m "feat(core): set decode_cap at pool init (heuristic + INFERNO_DECODE_THREADS)"
```

---

### Task 5: Document the knob

**Files:**
- Modify: `README.md` (near existing run/threads docs), `AGENTS.md` (non-obvious constraints list)

**Interfaces:**
- Consumes: nothing. Produces: user- and agent-facing documentation of `INFERNO_DECODE_THREADS` and the phase-aware-cap invariant.

- [ ] **Step 1: Add the AGENTS.md constraint**

First read the surrounding style: `sed -n '1,40p' AGENTS.md`. Add a bullet to the "Non-obvious constraints" list:

```markdown
- **Decode threading is phase-capped (M4b.5):** the compiled decode path
  (`inferno_par_gemv`) shards over `min(active_threads, decode_cap)`, not
  full cores — decode is bandwidth-bound and regresses past its knee.
  Prefill (`inferno_par_gemm`) is uncapped. Default `clamp(active/3, 2,
  active)`, override with `INFERNO_DECODE_THREADS=N`. The cap is
  bit-neutral (`shard_table` keeps each row on one lane); never treat a
  cap change as a numeric change.
```

- [ ] **Step 2: Add the README note**

Read the current run section: `grep -n "threads\|--threads\|Environment\|## " README.md | head`. Under the run/threads documentation, add:

```markdown
`INFERNO_DECODE_THREADS=N` caps the number of threads the *decode* phase
shards across (prefill still uses all `--threads`). Decode is
memory-bandwidth-bound, so more threads than saturate DRAM bandwidth only
add overhead; the default is a fraction of cores. Output is identical for
any value.
```

- [ ] **Step 3: Verify no build/doc breakage**

Run: `mise run test`
Expected: PASS (docs-only change; confirms nothing else regressed).

- [ ] **Step 4: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: document INFERNO_DECODE_THREADS + phase-aware decode cap"
```

---

### Task 6: Diagnostic sweep + exit-criterion recording

**Files:**
- Modify: `docs/superpowers/specs/2026-07-08-m4b5-phase-aware-decode-threading-design.md` (Amendments section)

**Interfaces:**
- Consumes: the shipped mechanism (Tasks 1-4). Produces: the recorded exit-criterion evidence. Leg 1 (correctness) is load-bearing and provable on this box; Leg 2 (performance) is directional-only here and deferred to quiet hardware.

- [ ] **Step 1: Prove Leg 1 — correctness gates green, no tolerance touched**

Run:
```bash
cargo nextest run -p inferno-codegen -E 'binary(differential)'
cargo nextest run -p inferno-core -E 'binary(artifact)'
cargo nextest run -p inferno-pool -E 'test(decode_cap) + test(bit_invisible) + test(bit_invariant)'
```
Expected: differential 5/5 PASS, artifact 4/4 PASS, pool cap tests PASS.

Confirm no tolerance/ABI edit on the branch:
```bash
git diff main..HEAD -- crates/inferno-graph/src/tolerance.rs crates/inferno-codegen/src/lib.rs | head
```
Expected: empty (no tolerance constant, no `HOST_ABI_VERSION` change).

- [ ] **Step 2: Capture the directional decode-thread sweep (this box — directional only)**

Run the decode sweep against the pinned model, varying the cap via the env knob at a fixed high `--threads`, so only the decode cap moves:

```bash
MODEL=/home/dev/.cache/inferno-tests/qwen2.5-0.5b-instruct-q8_0.gguf
for cap in 1 2 4 8 12; do
  echo "== decode_cap=$cap =="
  INFERNO_DECODE_THREADS=$cap cargo run --release -p inferno -- \
    run "$MODEL" --prompt "The quick brown fox" --max-tokens 64 --threads 12 --profile 2>&1 \
    | grep -iE "decode|tok/s|matmul" | head
done
```

This is **directional only** on the shared/quota'd devpod — not a verdict, exactly as M4b.1-M4b.4. It shows the cap knob takes effect and the direction of the decode-GEMV change; it does not finalize the default.

- [ ] **Step 3: Record the amendment**

Append a dated entry to the spec's Amendments section documenting: the correctness gates (Leg 1, PASS, load-bearing), the directional sweep numbers from Step 2 (clearly labelled directional/deferred), and that the **final default constant + Leg 2 performance verdict are deferred to a quiet-hardware re-run** (unquota'd bare metal, per the M4b.1 environment finding). Follow the exact tone of the M4b.4 spec's Task 6 entry: separate the load-bearing correctness result from the deferred performance verdict. Never edit a recorded data point.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-07-08-m4b5-phase-aware-decode-threading-design.md
git commit -m "docs(specs): M4b.5 exit — correctness PROVEN, perf verdict DEFERRED"
```

---

## Self-Review Notes

- **Spec coverage:** Mechanism (§Design/Mechanism) → Tasks 1-2. Default value + env (§Design/Default) → Task 4. Correctness & bit-neutrality (§Correctness) → Tasks 1 & 3, verified in Task 6 Step 1. Exit criterion two legs (§Exit Criterion) → Task 6. Docs (§Tasks item 5) → Task 5. Out-of-scope items (prefill threading, tg gap, auto-tune, `memory_bw_class`, CLI flag) → deliberately absent, matching the spec.
- **Type consistency:** `decode_cap`/`set_decode_threads`/`decode_threads` (pool), `set_global_decode_threads` (lib), `decode_cap(active, override_env)` (core) are used identically wherever they appear across tasks.
- **No-placeholder check:** every code step shows the exact code; every run step shows the exact command and expected result.
