# M4b.8 — Parallel Prefill Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shard each prefill tile's attention across the thread pool so the serial-attention fraction (~70% of prefill cycles) stops capping prefill scaling at ~4.1x @ t=12.

**Architecture:** A new `inferno_par_attention` C-ABI dispatcher in `inferno-pool` (mirroring `inferno_par_gemm`) shards a tile's `m` tokens across pool lanes with align-1 contiguous shards; each lane calls the **unchanged** per-token kernel `inferno_attention_f32_{isa}` over its token sub-range with its own heap-allocated `scores` scratch. `inferno-codegen`'s prefill tile driver splits `Step::Attention` into a serial whole-tile KV-append loop followed by one dispatcher call. Decode (`lower_decode` → `lower_step` → `lower_attention`) is untouched.

**Tech Stack:** Rust workspace; LLVM IR emission via inkwell (`inferno-codegen`); the fork-join pool in `inferno-pool`; insta/tempfile/libloading in tests.

**Spec:** `docs/superpowers/specs/2026-07-11-m4b8-parallel-attention-design.md`

## Global Constraints

- **Prefill only.** Decode attention stays serial; `lower_decode`/`lower_body`/`lower_step` and the per-token `lower_attention` remain the decode path.
- **Kernel ABI unchanged**: `inferno_attention_f32_{scalar,avx2}` in `crates/inferno-kernels/src/attention.rs` are not modified in any way.
- **No tolerance edits**: `attn_rel_tol`, `logits_abs_tol`, `gemv_rel_tol` untouched; all differential/artifact tests must pass as-is (AGENTS.md standing rule).
- **Bit-identity**: thread count must never change output bits (each token's out row is computed entirely by one lane with unchanged kernel math); the existing tiling bit-gate (`differential.rs`) must stay green.
- **Thread budget**: `active_threads` (prefill, uncapped). The M4b.5 `decode_cap` must NOT be applied to attention dispatch.
- **Align-1 sharding** for attention (never `SHARD_ALIGN = 8` — it would cap m=64 at 8 shards); GEMV/GEMM keep their 8-row alignment.
- **`m == 1` guard**: `inferno_par_attention` must call the kernel directly with no CAS and no job publish when `m <= 1`.
- **`HOST_ABI_VERSION` bumps "4" → "5"** (host-call shape gains a symbol; M4b.1/2/3 precedent).
- **`inferno-pool` and `inferno-codegen` stay `unsafe`-clean per workspace policy** — `inferno-pool` already has its own FFI-boundary allowances in the existing style; follow the file-local SAFETY-comment discipline exactly as `par_gemm` does. `inferno-formats` is irrelevant here (no parser changes).
- Run `mise run test` / `mise run lint` (CI names) — never hand-rolled cargo invocations for the final gate; clippy runs with `-D warnings` in `mise run lint`.
- After touching codegen op lowerings: `cargo test -p inferno-codegen --test differential` and `cargo test -p inferno-core --test artifact` (AGENTS.md).

---

### Task 1: Align-1 sharding — `shard_table_aligned`

**Files:**
- Modify: `crates/inferno-pool/src/shard.rs`
- Modify: `crates/inferno-pool/src/lib.rs:15` (re-export)

**Interfaces:**
- Produces: `pub fn shard_table_aligned(rows: usize, threads: usize, align: usize) -> Vec<(usize, usize)>` — same contract as `shard_table` but with a caller-chosen boundary alignment (`align.max(1)`). `shard_table(rows, threads)` becomes a thin wrapper calling it with `SHARD_ALIGN`. Task 2 consumes `shard_table_aligned(m, active, 1)`.

- [ ] **Step 1: Write the failing tests** — append to the `tests` module in `crates/inferno-pool/src/shard.rs`:

```rust
    #[test]
    fn align1_uses_all_threads_on_a_prefill_tile() {
        // The M4b.8 motivating case: a 64-token tile across 12 lanes must
        // yield 12 shards (8-aligned sharding would collapse it to 8).
        let s = shard_table_aligned(64, 12, 1);
        assert_eq!(s.len(), 12);
        assert_eq!(s[0], (0, 6)); // 64 = 12*5 + 4 → first 4 shards get 6
        assert_eq!(s.last().unwrap().1, 64);
    }

    #[test]
    fn align1_threads_exceeding_rows_collapses_to_rows() {
        let s = shard_table_aligned(3, 16, 1);
        assert_eq!(s, vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn shard_table_wrapper_is_align8() {
        for rows in [1, 7, 20, 64, 1000] {
            for threads in 1..=16 {
                assert_eq!(
                    shard_table(rows, threads),
                    shard_table_aligned(rows, threads, SHARD_ALIGN)
                );
            }
        }
    }

    /// The structural grid, generalized over align ∈ {1, 8}: shards tile
    /// `0..rows` contiguously, internal boundaries are align-multiples,
    /// shard count is `min(threads, ceil(rows/align))`, deterministic.
    #[test]
    fn structural_properties_hold_on_grid_for_align1() {
        for rows in (0..2048usize).step_by(7) {
            for threads in 1..=16usize {
                let s = shard_table_aligned(rows, threads, 1);
                assert_eq!(s, shard_table_aligned(rows, threads, 1), "determinism");
                if rows == 0 {
                    assert!(s.is_empty());
                    continue;
                }
                assert_eq!(s.len(), threads.min(rows));
                assert_eq!(s[0].0, 0);
                assert_eq!(s.last().unwrap().1, rows);
                for w in s.windows(2) {
                    assert_eq!(w[0].1, w[1].0, "contiguous");
                }
                for &(a, b) in &s {
                    assert!(a < b, "non-empty shard");
                }
            }
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p inferno-pool shard`
Expected: FAIL to compile — `shard_table_aligned` not found.

- [ ] **Step 3: Implement** — in `crates/inferno-pool/src/shard.rs`, generalize the existing body (keep the module doc, add a line noting attention uses align-1):

```rust
/// Split `0..rows` into at most `threads` contiguous shards whose internal
/// boundaries are multiples of `align.max(1)`. Strips are distributed as
/// evenly as possible (earlier shards get the remainder strips); the final
/// (possibly partial) strip lands in the last shard. `rows == 0` yields no
/// shards; `threads == 0` is treated as 1. GEMV/GEMM shard with
/// [`SHARD_ALIGN`] (AVX2 strips must not split); attention shards whole
/// tokens with `align = 1` (M4b.8 — 8-alignment would cap a 64-token tile
/// at 8 shards).
pub fn shard_table_aligned(rows: usize, threads: usize, align: usize) -> Vec<(usize, usize)> {
    let align = align.max(1);
    if rows == 0 {
        return Vec::new();
    }
    let strips = rows.div_ceil(align);
    let n = threads.max(1).min(strips);
    let base = strips / n;
    let extra = strips % n;
    let mut out = Vec::with_capacity(n);
    let mut strip = 0usize;
    for i in 0..n {
        let take = base + usize::from(i < extra);
        let start = strip * align;
        strip += take;
        out.push((start, (strip * align).min(rows)));
    }
    out
}

/// Split `0..rows` on [`SHARD_ALIGN`] boundaries — the GEMV/GEMM shard map.
pub fn shard_table(rows: usize, threads: usize) -> Vec<(usize, usize)> {
    shard_table_aligned(rows, threads, SHARD_ALIGN)
}
```

Update the re-export in `crates/inferno-pool/src/lib.rs`:

```rust
pub use shard::{SHARD_ALIGN, shard_table, shard_table_aligned};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p inferno-pool shard`
Expected: PASS (new tests + all pre-existing shard tests, which pin the wrapper's align-8 behavior).

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-pool/src/shard.rs crates/inferno-pool/src/lib.rs
git commit -m "pool: parameterize shard_table alignment (attention shards whole tokens, align-1)"
```

---

### Task 2: Pool attention job — `AttnFn`, `AttnJob`, `JobKind::Attention`, `Pool::par_attention`

**Files:**
- Modify: `crates/inferno-pool/src/pool.rs`
- Modify: `crates/inferno-pool/src/lib.rs:14` (re-export)

**Interfaces:**
- Consumes: `shard_table_aligned(m, active, 1)` from Task 1.
- Produces (Task 3 and tests rely on these exact shapes):

```rust
/// The M4b.3 attention kernel ABI: `(out, q, kv, scores, kv_base, v_off,
/// pos, kv_dim, n_heads, n_kv_heads, head_dim)`. Must match
/// `inferno-kernels`' `inferno_attention_f32_*` symbols exactly.
pub type AttnFn = unsafe extern "C" fn(
    *mut f32, *const f32, *mut f32, *mut f32,
    usize, usize, usize, usize, usize, usize, usize,
);

/// Per-tile invariants of one parallel-attention dispatch (M4b.8). The
/// token index `t in 0..m` is the sharded axis; per-token args derive as
/// `out + t*out_stride`, `q + t*q_stride`, `pos0 + t`.
#[derive(Clone, Copy)]
pub struct AttnJob {
    pub kernel: AttnFn,
    pub out: *mut f32,
    pub q: *const f32,
    pub kv: *mut f32,
    pub pos0: usize,
    pub kv_base: usize,
    pub v_off: usize,
    pub kv_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub q_stride: usize,
    pub out_stride: usize,
}

pub(crate) unsafe fn run_attn_span(j: &AttnJob, start: usize, end: usize); // serial loop over tokens [start, end)
impl Pool { pub unsafe fn par_attention(&self, job: &AttnJob, m: usize); }
```

- [ ] **Step 1: Write the failing tests** — append to the `tests` module in `crates/inferno-pool/src/pool.rs`:

```rust
    /// Fake attention kernel with the real ABI: deterministic function of
    /// (q row, pos), writes the whole out row, and touches `scores[pos]`
    /// to prove each lane's scratch really covers `pos + 1` entries.
    unsafe extern "C" fn stamp_attn(
        out: *mut f32,
        q: *const f32,
        _kv: *mut f32,
        scores: *mut f32,
        _kv_base: usize,
        _v_off: usize,
        pos: usize,
        _kv_dim: usize,
        n_heads: usize,
        _n_kv_heads: usize,
        head_dim: usize,
    ) {
        // SAFETY: run_attn_span sizes scores to max pos + 1 for its span.
        unsafe { *scores.add(pos) = pos as f32 };
        for i in 0..n_heads * head_dim {
            // SAFETY: out/q rows are n_heads*head_dim per the AttnFn contract.
            unsafe { *out.add(i) = *q.add(i) + (pos * 31 + i) as f32 };
        }
    }

    const ATTN_NH: usize = 3;
    const ATTN_HD: usize = 4;
    const ATTN_STRIDE: usize = ATTN_NH * ATTN_HD;

    fn attn_dispatch(pool: &Pool, m: usize, pos0: usize) -> Vec<f32> {
        let q: Vec<f32> = (0..m * ATTN_STRIDE).map(|i| i as f32).collect();
        let mut out = vec![f32::NAN; m * ATTN_STRIDE];
        let mut kv = [0f32; 1];
        let job = AttnJob {
            kernel: stamp_attn,
            out: out.as_mut_ptr(),
            q: q.as_ptr(),
            kv: kv.as_mut_ptr(),
            pos0,
            kv_base: 0,
            v_off: 0,
            kv_dim: 0,
            n_heads: ATTN_NH,
            n_kv_heads: 1,
            head_dim: ATTN_HD,
            q_stride: ATTN_STRIDE,
            out_stride: ATTN_STRIDE,
        };
        // SAFETY: buffers sized per stamp_attn's expectations, live for the call.
        unsafe { pool.par_attention(&job, m) };
        out
    }

    fn attn_expected(m: usize, pos0: usize) -> Vec<f32> {
        (0..m * ATTN_STRIDE)
            .map(|j| {
                let (t, i) = (j / ATTN_STRIDE, j % ATTN_STRIDE);
                j as f32 + ((pos0 + t) * 31 + i) as f32
            })
            .collect()
    }

    #[test]
    fn attention_parallel_matches_serial_expectation() {
        let pool = Pool::new(4);
        for m in [1, 2, 7, 63, 64, 100] {
            assert_eq!(attn_dispatch(&pool, m, 5), attn_expected(m, 5), "m={m}");
        }
    }

    #[test]
    fn attention_threads_exceeding_tokens_collapses() {
        let pool = Pool::new(16);
        assert_eq!(attn_dispatch(&pool, 3, 0), attn_expected(3, 0));
    }

    #[test]
    fn attention_capacity_one_runs_inline() {
        let pool = Pool::new(1);
        assert_eq!(attn_dispatch(&pool, 64, 9), attn_expected(64, 9));
    }

    #[test]
    fn attention_ignores_decode_cap() {
        // The decode cap applies to par_gemv only; attention (prefill work)
        // shards over full active. Result must be identical either way.
        let pool = Pool::new(8);
        pool.set_decode_threads(1);
        assert_eq!(attn_dispatch(&pool, 64, 0), attn_expected(64, 0));
    }

    #[test]
    fn attention_zero_tokens_is_a_noop() {
        let pool = Pool::new(4);
        assert!(attn_dispatch(&pool, 0, 0).is_empty());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p inferno-pool attention`
Expected: FAIL to compile — `AttnJob` / `par_attention` not found.

- [ ] **Step 3: Implement** — in `crates/inferno-pool/src/pool.rs`:

3a. Add `AttnFn` and `AttnJob` (exact code from the Interfaces block above) below the `GemmFn` typedef. Import `shard_table_aligned` alongside `shard_table` at the top:

```rust
use crate::shard::{shard_table, shard_table_aligned};
```

3b. Add the variant to `JobKind` (it stays `Clone, Copy`; raw pointers are `Copy`):

```rust
    Attention(AttnJob),
```

3c. Add the span runner below `run_shard` (NOT inside it — the serial fallbacks in `lib.rs` call it too):

```rust
/// Run attention for tokens `[start, end)` of a dispatch: one serial
/// per-token kernel call per token, with one lane-local `scores` scratch
/// sized for the span's largest `pos + 1`. Heap-allocating here (one Vec
/// per lane per tile per layer) is noise next to the attention math; it
/// keeps the kernel ABI scratch-free of threading concerns.
///
/// # Safety
/// The dispatcher's caller contract must cover tokens `[start, end)`:
/// `out`/`q` valid for `m` rows of their strides (out rows disjoint per
/// token — each token is computed by exactly one lane), `kv` fully
/// appended for positions `< pos0 + end` and read-only for the duration,
/// `kernel` a valid `AttnFn`.
pub(crate) unsafe fn run_attn_span(j: &AttnJob, start: usize, end: usize) {
    let mut scores = vec![0f32; j.pos0 + end];
    for t in start..end {
        // SAFETY: forwarding the caller's contract for token t; scores is
        // sized pos0 + end >= (pos0 + t) + 1.
        unsafe {
            (j.kernel)(
                j.out.add(t * j.out_stride),
                j.q.add(t * j.q_stride),
                j.kv,
                scores.as_mut_ptr(),
                j.kv_base,
                j.v_off,
                j.pos0 + t,
                j.kv_dim,
                j.n_heads,
                j.n_kv_heads,
                j.head_dim,
            );
        }
    }
}
```

3d. Add the arm to `run_shard`'s match:

```rust
        // SAFETY: forwarding the caller's contract for the disjoint token span.
        JobKind::Attention(job) => unsafe { run_attn_span(&job, start, end) },
```

3e. Add `Pool::par_attention` after `par_gemm`, mirroring its shape exactly (publish job → store `remaining` → bump packed epoch → wake shard-holding workers → run shard 0 → spin). The only deltas from `par_gemm`: align-1 sharding over `m`, full `active_threads()` (no decode cap), and the job's generic pointer fields are null (the `Attention` variant carries its own):

```rust
    /// Tiled prefill attention across up to `active_threads()` lanes
    /// (M4b.8): splits the tile's `m` tokens into align-1 contiguous
    /// shards. Each token's out row is computed entirely by one lane with
    /// the unchanged per-token kernel, so thread count never changes
    /// output bits. The M4b.5 decode cap does NOT apply — attention here
    /// is prefill work.
    ///
    /// # Safety
    /// As [`run_attn_span`] over `0..m`; calls must not overlap (one job
    /// at a time).
    pub unsafe fn par_attention(&self, job: &AttnJob, m: usize) {
        if m == 0 {
            return;
        }
        let active = self.active_threads();
        let shards = shard_table_aligned(m, active, 1);
        if shards.len() == 1 {
            // SAFETY: caller contract covers the full token range.
            unsafe { run_attn_span(job, 0, m) };
            return;
        }
        let n_worker = shards.len() - 1;
        let (s0, e0) = shards[0];
        let kind = JobKind::Attention(*job);
        // SAFETY (job write): the previous dispatch ended with
        // `remaining == 0`, and shardless workers never read `job`
        // (packed-epoch protocol) — no reader exists here.
        unsafe {
            *self.shared.job.get() = Job {
                kind: Some(kind),
                y: std::ptr::null_mut(),
                xq: std::ptr::null(),
                w: std::ptr::null(),
                k: 0,
                shards,
            };
        }
        self.shared.remaining.store(n_worker, Ordering::SeqCst);
        let counter =
            (self.shared.epoch.load(Ordering::SeqCst) >> PACKED_SHARD_BITS).wrapping_add(1);
        self.shared.epoch.store(
            (counter << PACKED_SHARD_BITS) | (n_worker + 1),
            Ordering::SeqCst,
        );
        // Wake exactly the workers that hold a shard (same handshake as
        // par_gemv; see there for the lost-wakeup argument).
        for slot in &self.shared.slots[..n_worker] {
            if slot.parked.load(Ordering::SeqCst) {
                slot.thread
                    .get()
                    .expect("worker registered in Pool::new")
                    .unpark();
            }
        }
        // SAFETY: caller contract; shard 0's tokens are disjoint from
        // worker shards.
        unsafe { run_attn_span(job, s0, e0) };
        let mut spins = 0u32;
        while self.shared.remaining.load(Ordering::Acquire) != 0 {
            if spins < SPIN_ITERS {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
```

3f. Update the `run_shard` doc comment's first paragraph to mention the third kind, and update the re-export in `crates/inferno-pool/src/lib.rs`:

```rust
pub use pool::{AttnFn, AttnJob, GemmFn, GemvFn, Pool};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p inferno-pool`
Expected: PASS — new attention tests plus every pre-existing pool/shard/integration test.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-pool/src/pool.rs crates/inferno-pool/src/lib.rs
git commit -m "pool: JobKind::Attention + Pool::par_attention (align-1 token shards, M4b.8)"
```

---

### Task 3: C-ABI entry `inferno_par_attention` + fallback tests + rig coercion

**Files:**
- Modify: `crates/inferno-pool/src/lib.rs`
- Create: `crates/inferno-pool/tests/par_attention_fallback.rs`
- Modify: `crates/inferno-pool/tests/par_rig.rs` (ABI-drift coercion)

**Interfaces:**
- Consumes: `AttnJob`, `Pool::par_attention`, `run_attn_span` (Task 2).
- Produces: the host symbol generated code calls (Task 4 declares and calls it):

```rust
pub unsafe extern "C" fn inferno_par_attention(
    kernel: AttnFn, out: *mut f32, q: *const f32, kv: *mut f32,
    pos0: usize, m: usize, kv_base: usize, v_off: usize, kv_dim: usize,
    n_heads: usize, n_kv_heads: usize, head_dim: usize,
    q_stride: usize, out_stride: usize,
);
```

- [ ] **Step 1: Write the failing tests**

Create `crates/inferno-pool/tests/par_attention_fallback.rs`. This file must NEVER call `init_global` — an integration test file is its own process, so the global pool is guaranteed uninitialized here, pinning the serial-fallback and `m == 1` paths:

```rust
//! `inferno_par_attention` without an initialized global pool: the entry
//! point must degrade to the serial full-range loop (and take the m == 1
//! direct path) — this file never calls `init_global`, and an integration
//! test binary is its own process, so the pool is guaranteed absent.
#![allow(unsafe_code)] // FFI entry-point tests; same justification as the sibling test files.

use inferno_pool::inferno_par_attention;

unsafe extern "C" fn stamp_attn(
    out: *mut f32,
    q: *const f32,
    _kv: *mut f32,
    scores: *mut f32,
    _kv_base: usize,
    _v_off: usize,
    pos: usize,
    _kv_dim: usize,
    n_heads: usize,
    _n_kv_heads: usize,
    head_dim: usize,
) {
    // SAFETY: the dispatcher sizes scores to max pos + 1 for its span.
    unsafe { *scores.add(pos) = pos as f32 };
    for i in 0..n_heads * head_dim {
        // SAFETY: out/q rows are n_heads*head_dim per the AttnFn contract.
        unsafe { *out.add(i) = *q.add(i) + (pos * 31 + i) as f32 };
    }
}

const NH: usize = 3;
const HD: usize = 4;
const STRIDE: usize = NH * HD;

fn dispatch(m: usize, pos0: usize) -> Vec<f32> {
    let q: Vec<f32> = (0..m * STRIDE).map(|i| i as f32).collect();
    let mut out = vec![f32::NAN; m * STRIDE];
    let mut kv = [0f32; 1];
    // SAFETY: buffers sized per stamp_attn's expectations, live for the call.
    unsafe {
        inferno_par_attention(
            stamp_attn,
            out.as_mut_ptr(),
            q.as_ptr(),
            kv.as_mut_ptr(),
            pos0,
            m,
            0,
            0,
            0,
            NH,
            1,
            HD,
            STRIDE,
            STRIDE,
        );
    }
    out
}

fn expected(m: usize, pos0: usize) -> Vec<f32> {
    (0..m * STRIDE)
        .map(|j| {
            let (t, i) = (j / STRIDE, j % STRIDE);
            j as f32 + ((pos0 + t) * 31 + i) as f32
        })
        .collect()
}

#[test]
fn uninitialized_pool_falls_back_to_serial_full_range() {
    for m in [2, 7, 64] {
        assert_eq!(dispatch(m, 3), expected(m, 3), "m={m}");
    }
}

#[test]
fn m1_takes_the_direct_path() {
    // Decode-shaped call (and the T=1 prefill tile): one token, computed
    // correctly with no pool involvement by construction.
    assert_eq!(dispatch(1, 41), expected(1, 41));
}

#[test]
fn m0_is_a_noop() {
    assert!(dispatch(0, 0).is_empty());
}
```

Add the ABI-drift coercion to `crates/inferno-pool/tests/par_rig.rs`, next to the existing `GemvFn`/`GemmFn` coercions (exact placement: near the `SHARD_ALIGN == STRIP` assert at the top of the file):

```rust
// AttnFn must match the real attention kernel ABI — a drift is a compile
// error here, same trick as the GemvFn/GemmFn coercions.
let _: inferno_pool::AttnFn = inferno_kernels::inferno_attention_f32_scalar;
let _: inferno_pool::AttnFn = inferno_kernels::inferno_attention_f32_avx2;
```

(Put the two lines inside the existing test fn that asserts `SHARD_ALIGN == STRIP`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p inferno-pool --test par_attention_fallback`
Expected: FAIL to compile — `inferno_par_attention` not found.

- [ ] **Step 3: Implement** — in `crates/inferno-pool/src/lib.rs`, after `inferno_par_gemm`:

```rust
/// Host dispatcher for tiled prefill attention (M4b.8). Same
/// single-dispatcher guard + serial fallback as [`inferno_par_gemv`];
/// shares `DISPATCH_CLAIMED` deliberately — within one forward pass GEMV,
/// GEMM and attention dispatches are issued serially and never overlap,
/// so one guard suffices. `m <= 1` (decode-shaped calls, T=1 prefill
/// tiles) takes a direct serial path with no CAS and no job publish, so
/// decode never touches the pool for attention. On the CAS-loss (or
/// uninitialized-pool) path this runs the serial full-range token loop,
/// bit-identical to the pooled path since each token's out row is
/// computed by a single kernel invocation either way.
///
/// A panic inside the dispatcher or kernel aborts the process at this
/// `extern "C"` boundary — there is no unwind across FFI.
///
/// # Safety
/// Same contract as [`Pool::par_attention`] over tokens `0..m`;
/// additionally `kernel` must be a valid non-null function pointer with
/// the M4b.3 attention ABI, and the KV cache must already contain every
/// position `< pos0 + m` (the tile's append loop runs before this call).
/// Generated code guarantees all of this by construction (M3 trust model).
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_par_attention(
    kernel: AttnFn,
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    pos0: usize,
    m: usize,
    kv_base: usize,
    v_off: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_stride: usize,
    out_stride: usize,
) {
    if m == 0 {
        return;
    }
    let job = pool::AttnJob {
        kernel,
        out,
        q,
        kv,
        pos0,
        kv_base,
        v_off,
        kv_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        q_stride,
        out_stride,
    };
    if m == 1 {
        // SAFETY: forwarding the caller's contract for the single token.
        unsafe { pool::run_attn_span(&job, 0, 1) };
        return;
    }
    match GLOBAL.get() {
        Some(p) => {
            if DISPATCH_CLAIMED
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: forwarding the caller's contract unchanged.
                unsafe { p.par_attention(&job, m) };
                DISPATCH_CLAIMED.store(false, Ordering::Release);
            } else {
                // Lost the race for the pool's single dispatcher slot: run
                // serially over the full token range instead of overlapping
                // another thread's in-flight pool dispatch.
                // SAFETY: forwarding the caller's contract for the full range.
                unsafe { pool::run_attn_span(&job, 0, m) };
            }
        }
        // SAFETY: forwarding the caller's contract for the full range.
        None => unsafe { pool::run_attn_span(&job, 0, m) },
    }
}
```

Also update the crate doc comment at the top of `lib.rs` (currently "the `inferno_par_gemv` dispatcher") to say the crate hosts the three dispatchers `inferno_par_{gemv,gemm,attention}`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p inferno-pool`
Expected: PASS — the two new fallback tests, the rig coercion (compile-time), and everything pre-existing.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-pool/src/lib.rs crates/inferno-pool/tests/par_attention_fallback.rs crates/inferno-pool/tests/par_rig.rs
git commit -m "pool: inferno_par_attention C-ABI entry (m<=1 direct path, shared dispatch guard)"
```

---

### Task 4: Codegen — declare the dispatcher, split KV-append, parallel tile arm

**Files:**
- Modify: `crates/inferno-codegen/src/llvm/mod.rs` (extern decl ~line 146 block; IR-contains test ~line 336)
- Modify: `crates/inferno-codegen/src/llvm/ops.rs` (`lower_tile` ~710; `lower_attention` ~1203)
- Modify: `crates/inferno-codegen/tests/differential.rs:55-78` (`retain_kernel_symbols`)

**Interfaces:**
- Consumes: the `inferno_par_attention` symbol (Task 3) — 4 ptr + 10 i64 params, argument order exactly as in Task 3's signature.
- Produces: `fn lower_kv_append(&self, frame: &Frame<'c>, k: usize, v: usize, layer: usize)`, `fn module_isa(&self) -> inferno_kernels::KernelIsa`, `fn lower_tile_attention(&self, env: &TileEnv<'c>, step: &Step, tile_start: IntValue<'c>, m: IntValue<'c>)` — all private to `ops.rs`; no cross-task consumers.

- [ ] **Step 1: Declare the extern** — in `crates/inferno-codegen/src/llvm/mod.rs`, immediately after the `inferno_par_gemm` declaration block:

```rust
        // void inferno_par_attention(ptr kernel, ptr out, ptr q, ptr kv,
        //     i64 pos0, i64 m, i64 kv_base, i64 v_off, i64 kv_dim,
        //     i64 n_heads, i64 n_kv_heads, i64 head_dim,
        //     i64 q_stride, i64 out_stride)
        // — the M4b.8 prefill-attention dispatcher; the attention kernel
        // chosen by `attention_symbol` is passed as a function pointer, so
        // the ISA selection logic is unchanged.
        let par_attn_ty = void.fn_type(
            &[
                ptr.into(),
                ptr.into(),
                ptr.into(),
                ptr.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
            ],
            false,
        );
        self.module
            .add_function("inferno_par_attention", par_attn_ty, Some(Linkage::External));
```

And in the module test near line 336, add alongside the existing asserts:

```rust
        assert!(ir.contains("inferno_par_attention"));
```

- [ ] **Step 2: Run the module test to verify the new assert fails before the arm exists**

Run: `cargo test -p inferno-codegen --lib`
Expected: the IR-contains test FAILS if the assert checks for a *call* — it checks `ir.contains`, and the declaration alone satisfies it, so expect PASS. The real failing gate for this task is Step 6; proceed.

- [ ] **Step 3: Split `lower_attention`** — in `crates/inferno-codegen/src/llvm/ops.rs`, extract the KV-append half (currently lines ~1224-1238 plus the constants they use) into:

```rust
    /// Append this token's k/v rows into the f32 KV cache at `frame.pos` —
    /// the write half of `Step::Attention`. Called per token by the decode
    /// path (`lower_attention`) and by the prefill tile arm
    /// (`lower_tile_attention`), which appends the WHOLE tile before its
    /// parallel attention read. That reordering is bit-safe: token i's
    /// causal read never reaches rows past `pos_i`.
    fn lower_kv_append(&self, frame: &Frame<'c>, k: usize, v: usize, layer: usize) {
        let kv_dim = self.plan.kv.kv_dim as u64;
        let seq_len = self.plan.max_seq_len as u64;
        let kv_base = layer as u64 * seq_len * kv_dim * 2;
        let k_region = kv_base;
        let v_region = kv_base + seq_len * kv_dim;
        let kv_dim_c = self.const_i64(kv_dim);
        let pos_kv = self
            .builder
            .build_int_mul(frame.pos, kv_dim_c, "poskv")
            .unwrap();
        let k_row = self.row_base(frame, k);
        let v_row = self.row_base(frame, v);
        let k_dst = self.add(self.const_i64(k_region), pos_kv);
        let v_dst = self.add(self.const_i64(v_region), pos_kv);
        self.range_loop(kv_dim_c, |cg, c| {
            let kval = cg.load_f32(cg.arena_ptr(frame, k_row, c));
            cg.store_f32(cg.elem_ptr(frame.kv, cg.add(k_dst, c)), kval);
            let vval = cg.load_f32(cg.arena_ptr(frame, v_row, c));
            cg.store_f32(cg.elem_ptr(frame.kv, cg.add(v_dst, c)), vval);
        });
    }

    /// Every PackedWeight carries the same target-derived ISA; use it as
    /// the module ISA (attention has no PackedWeight of its own).
    fn module_isa(&self) -> inferno_kernels::KernelIsa {
        self.plan
            .weights
            .weights
            .first()
            .map(|w| w.isa)
            .unwrap_or(inferno_kernels::KernelIsa::Scalar)
    }
```

Rewrite `lower_attention` to call `self.lower_kv_append(frame, k, v, layer)` for its append half and `self.module_isa()` for its ISA pick; the kernel-call half is otherwise byte-for-byte what it is today (recompute `kv_dim`/`seq_len`/`kv_base` locals it still needs). Delete the now-duplicated inline code.

- [ ] **Step 4: Add the parallel tile arm** — still in `ops.rs`:

4a. New method after `lower_gemm`:

```rust
    /// Tiled prefill attention (M4b.8): append the whole tile's k/v
    /// serially (same per-token order as before), then ONE
    /// `inferno_par_attention` call shards the tile's `m` tokens across
    /// pool lanes. Each token's out row is computed entirely by one lane
    /// with the unchanged per-token kernel, so thread count never changes
    /// output bits, and token i's causal read never reaches KV rows past
    /// `pos_i`, so appending the whole tile first is bit-neutral.
    fn lower_tile_attention(
        &self,
        env: &TileEnv<'c>,
        step: &Step,
        tile_start: IntValue<'c>,
        m: IntValue<'c>,
    ) {
        let Step::Attention {
            q,
            k,
            v,
            layer,
            n_heads,
            n_kv_heads,
            head_dim,
            out,
        } = step
        else {
            unreachable!("lower_tile_attention called on non-Attention step")
        };
        self.range_loop(m, |cg, ti| {
            let row = cg.add(tile_start, ti);
            let frame = cg.tile_frame(env, row);
            cg.lower_kv_append(&frame, *k, *v, *layer);
        });
        let kv_dim = self.plan.kv.kv_dim as u64;
        let seq_len = self.plan.max_seq_len as u64;
        let kv_base = *layer as u64 * seq_len * kv_dim * 2;
        let q_ptr = self.arena_row_ptr_at(env.arena, *q, tile_start);
        let out_ptr = self.arena_row_ptr_at(env.arena, *out, tile_start);
        let pos0 = self.add(env.pos_off, tile_start);
        let sym = crate::loopir::attention_symbol(self.module_isa());
        let afn = self
            .module
            .get_function(&sym)
            .expect("attention kernel declared (Task 6)");
        let pfn = self
            .module
            .get_function("inferno_par_attention")
            .expect("par attention dispatcher declared");
        self.builder
            .build_call(
                pfn,
                &[
                    afn.as_global_value().as_pointer_value().into(),
                    out_ptr.into(),
                    q_ptr.into(),
                    env.kv.into(),
                    pos0.into(),
                    m.into(),
                    self.const_i64(kv_base).into(),
                    self.const_i64(seq_len * kv_dim).into(),
                    self.const_i64(kv_dim).into(),
                    self.const_i64(*n_heads as u64).into(),
                    self.const_i64(*n_kv_heads as u64).into(),
                    self.const_i64(*head_dim as u64).into(),
                    self.const_i64(self.row_len(*q)).into(),
                    self.const_i64(self.row_len(*out)).into(),
                ],
                "par_attention",
            )
            .unwrap();
    }
```

4b. In `lower_tile`'s match (ops.rs ~720), add an arm above the `_` catch-all:

```rust
                    Step::Attention { .. } => {
                        self.profiled(&label, |cg| {
                            cg.lower_tile_attention(env, step, tile_start, m)
                        });
                    }
```

4c. Update `lower_tile`'s doc comment (~702-709): the sentence about "per-token KV-append order within the m-loop" becomes a statement that attention appends the whole tile's k/v in token order, then dispatches one parallel attention call over the tile (bit-identical and T-invariant because each token's result is computed by one lane with the unchanged kernel and its causal read stops at its own position).

- [ ] **Step 5: Retain the symbol in the differential harness** — in `crates/inferno-codegen/tests/differential.rs`, `retain_kernel_symbols` (~line 76), add:

```rust
    p(inferno_pool::inferno_par_attention as *const ());
```

(Without this the new `model.so` fails `dlopen` on the undefined symbol; also update the fn's doc comment, which enumerates the dispatchers.)

- [ ] **Step 6: Run the codegen gates**

Run: `cargo test -p inferno-codegen`
Expected: PASS — including `differential_tiny_gguf` (tolerance vs interpreter unchanged), `prefill_tiling_is_bit_invariant_to_tile_size` (T=1 exercises the `m == 1` direct path; T=4 exercises the uninitialized-pool serial fallback), and `profiling_does_not_change_logits`. No tolerance was touched; if anything is red, the lowering is wrong — fix the code, never the tolerance.

- [ ] **Step 7: Run the artifact-level differential (AGENTS.md rule for op-lowering changes)**

Run: `cargo test -p inferno-core --test artifact`
Expected: FAIL at link/dlopen — `inferno-core`'s own retention list doesn't export the new symbol yet. That is Task 5's job; if it happens to pass (linker kept the symbol anyway), fine — proceed either way, Task 5 makes it explicit.

- [ ] **Step 8: Commit**

```bash
git add crates/inferno-codegen/src/llvm/mod.rs crates/inferno-codegen/src/llvm/ops.rs crates/inferno-codegen/tests/differential.rs
git commit -m "codegen: prefill attention via inferno_par_attention (whole-tile KV append + one dispatch per tile)"
```

---

### Task 5: Host plumbing — symbol retention + `HOST_ABI_VERSION` bump

**Files:**
- Modify: `crates/inferno-core/src/artifact.rs:550` (retention list)
- Modify: `crates/inferno-codegen/src/lib.rs:11-17` (`HOST_ABI_VERSION`)

**Interfaces:**
- Consumes: `inferno_pool::inferno_par_attention` (Task 3).
- Produces: nothing new — this task makes the production loader export the symbol and stale-cache keys rotate.

- [ ] **Step 1: Add retention** — in `crates/inferno-core/src/artifact.rs`, next to the existing line 550:

```rust
    p(inferno_pool::inferno_par_attention as *const ());
```

- [ ] **Step 2: Bump the ABI version** — in `crates/inferno-codegen/src/lib.rs`, update the doc comment and constant:

```rust
/// Version of the host-symbol surface generated code links against
/// (kernel symbols + `inferno_par_{gemv,gemm,attention}` + the profiler
/// global). Folded into `inferno-core`'s artifact cache key. "5" = M4b.8's
/// `inferno_par_attention` dispatch; "4" was M4b.3's attention kernel
/// symbols (`inferno_attention_f32_{scalar,avx2}`); "3" was M4b.2's
/// GEMM dispatch + optional profiling (v2 was M4b.1's `inferno_par_gemv`).
pub const HOST_ABI_VERSION: &str = "5";
```

- [ ] **Step 3: Run the artifact gate**

Run: `cargo test -p inferno-core --test artifact`
Expected: PASS — the artifact-level compiled-vs-interpreter differential is green with unchanged tolerances, and cached artifacts from before this change recompile (new cache key).

- [ ] **Step 4: Run the full core + cache tests**

Run: `cargo test -p inferno-core`
Expected: PASS (the cache-key test at `cache.rs:122-138` asserts key *sensitivity*, not specific values, so no fixture updates).

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-core/src/artifact.rs crates/inferno-codegen/src/lib.rs
git commit -m "core/codegen: export inferno_par_attention to model.so; HOST_ABI_VERSION 4->5"
```

---

### Task 6: The threads bit-gate

**Files:**
- Modify: `crates/inferno-codegen/tests/differential.rs` (new test at the end)

**Interfaces:**
- Consumes: `run_compiled`, `Meta`, `retain_kernel_symbols` (existing test helpers in the same file); `inferno_pool::{init_global, set_global_active_threads}`.

**Note:** this is a regression *gate*, not a red-first TDD test — before Task 4 it would pass trivially (attention was serial). Its job is to pin the new dispatcher's bit-neutrality forever, the thread-axis analogue of the tiling gate.

- [ ] **Step 1: Write the gate**

```rust
/// THE THREADS GATE (M4b.8): prefill logits must be **bitwise** identical
/// across pool thread counts. Compile at T=4 (so a 10-token prompt spans
/// 3 tiles and each tile's `inferno_par_attention` dispatch shards m
/// tokens into align-1 shards), run the same prompt with the pool capped
/// at 1 lane and at 8 lanes, and compare bits. Each token's attention out
/// row is computed entirely by one lane with the unchanged per-token
/// kernel, and GEMM sharding was already bit-neutral (M4b.1/2), so any
/// difference means the dispatcher partitioned wrongly — fix the pool,
/// never the tolerance.
///
/// This is the only test in this binary that initializes the global pool;
/// other tests in the same process then dispatch through it too, which is
/// harmless — bit-identical by construction (that is this crate's whole
/// invariant).
#[test]
fn prefill_is_bit_invariant_to_thread_count() {
    let fixture = "../inferno-formats/tests/fixtures/tiny.gguf";
    let desc = load_desc(Path::new(fixture)).unwrap();
    let graph = build_graph(&desc).unwrap();
    let target = TargetDesc::detect().unwrap();
    let tokens: Vec<u32> = vec![1, 5, 9, 3, 2, 7, 4, 6, 0, 8];

    let tmp = tempfile::tempdir().unwrap();
    let art = compile(
        &desc,
        &graph,
        &target,
        64,
        &CompileOptions {
            profile: false,
            prefill_tile: 4,
        },
        tmp.path(),
    )
    .unwrap();
    let meta: Meta =
        serde_json::from_slice(&std::fs::read(art.dir.join("meta.json")).unwrap()).unwrap();

    inferno_pool::init_global(8).unwrap();
    inferno_pool::set_global_active_threads(1);
    let l1 = unsafe { run_compiled(&art.dir, &tokens, &meta) };
    inferno_pool::set_global_active_threads(8);
    let l8 = unsafe { run_compiled(&art.dir, &tokens, &meta) };
    for (i, (a, b)) in l1.iter().zip(&l8).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit {i} differs between t=1 and t=8 ({a} vs {b})"
        );
    }
}
```

- [ ] **Step 2: Run the gate**

Run: `cargo test -p inferno-codegen --test differential prefill_is_bit_invariant_to_thread_count`
Expected: PASS. If it fails, the dispatcher has a sharding bug (overlapping token spans, wrong stride math, or a scores-scratch race) — debug with `set_global_active_threads(2)` to shrink the diff surface. Do NOT touch tolerances (the gate is bitwise; there is no tolerance to touch — that is the point).

- [ ] **Step 3: Run the whole differential suite again** (the pool is now initialized in this binary; every other test must still pass through pooled dispatch)

Run: `cargo test -p inferno-codegen --test differential`
Expected: PASS, all tests.

- [ ] **Step 4: Commit**

```bash
git add crates/inferno-codegen/tests/differential.rs
git commit -m "codegen: threads bit-gate — prefill logits bitwise invariant to pool thread count"
```

---

### Task 7: Docs, full verification, wrap-up

**Files:**
- Modify: `AGENTS.md` (threading bullet)
- Modify: `ARCHITECTURE.md` (only if it enumerates the pool's dispatchers — check first)
- Verify: whole workspace

- [ ] **Step 1: Update AGENTS.md** — in the M4b.5 decode-threading bullet (which names `inferno_par_gemv`/`inferno_par_gemm`), append one sentence:

> Prefill attention (M4b.8) dispatches per tile through `inferno_par_attention`, sharding the tile's tokens with align-1 shards at full `active` — the decode cap never applies to it, and `m <= 1` calls bypass the pool entirely.

- [ ] **Step 2: Check ARCHITECTURE.md** for a description of `inferno-pool` or its dispatcher list:

Run: `grep -n "par_gemm\|par_gemv\|inferno-pool" ARCHITECTURE.md`
If it enumerates dispatchers, add `inferno_par_attention` to the enumeration in the same style; if it only names the crate, leave it.

- [ ] **Step 3: Full test suite via the repo's named tasks**

Run: `mise run test`
Expected: PASS. (This is the CI-blocking tier; snapshot tests must not need review — this change adds no insta snapshots.)

- [ ] **Step 4: Lint (CI runs this; local `mise run test` does not)**

Run: `mise run lint`
Expected: PASS with zero clippy warnings (`-D warnings`). Likely nits to pre-empt: `#[allow(clippy::too_many_arguments)]` on `inferno_par_attention` and the test stamp kernels; no `unsafe` without a `// SAFETY:` comment in inferno-pool (the crate's existing discipline).

- [ ] **Step 5: Commit docs**

```bash
git add AGENTS.md ARCHITECTURE.md
git commit -m "docs: record the M4b.8 parallel prefill attention dispatch"
```

- [ ] **Step 6: Wrap-up note.** Implementation complete ≠ milestone complete: the M4b.8 verdict is the **manual quiet-hw protocol** (`mise run metal` re-running `gate-prefill-scaling`; gate ≥6x @ t=12), recorded as amendments in the M4b.1 spec and the M4b.8 spec per their §Verification protocol. That run is an operator action outside this plan — surface it as the explicit next step when handing back. Likewise the spec's verification item 1 (`mise run bench-compiled` stays green): it is the **nightly** t=1 codegen-quality gate and devpod numbers are untrusted (standing M4b discipline), so do not run it locally as a pass/fail signal — note that the next nightly covers it.
