# M4b.9 — Serial-Tail Parallelization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize every remaining serial per-token prefill op (rmsnorm, rope, add, swiglu, bias, embed, KV-append, activation quantize) by outlining their per-token loop bodies into emitted C-ABI functions dispatched through one new `inferno_par_token_loop` pool entry, and split the `kv_append`/`quantize` profile brackets.

**Architecture:** Codegen outlines each serial op-site's per-token `range_loop(m)` body into a private module function `void tok_body.<label>.<n>(ptr ctx, i64 t0, i64 t1)`; the ctx is an opaque 6-word stack pack (`tokens, pos_off, weights, kv, arena, tile_start`) whose layout only codegen knows. `inferno-pool` gains `JobKind::TokenLoop` + `Pool::par_token_loop` + the C-ABI `inferno_par_token_loop` dispatcher, mirroring `inferno_par_attention` exactly (align-1 token shards, `m == 1` direct path, `DISPATCH_CLAIMED` CAS, serial full-range fallback). Spec: `docs/superpowers/specs/2026-07-12-m4b9-serial-tail-parallelization-design.md`.

**Tech Stack:** Rust workspace; inkwell 0.6 / LLVM 18 (opaque pointers) in `inferno-codegen`; `mise run test` / `mise run lint` are the CI-equivalent gates.

## Global Constraints

- **No tolerance edits, ever:** `logits_abs_tol`, `attn_rel_tol`, `gemv_rel_tol`, `LOGIT_TIE_EPSILON` untouched; every differential passes as-is (AGENTS.md standing rule).
- **Thread count never changes output bits**; the tiling bit-gate (`prefill_tiling_is_bit_invariant_to_tile_size`) and threads bit-gate (`prefill_is_bit_invariant_to_thread_count`) must stay green through every task.
- **`HOST_ABI_VERSION` bumps `"5"` → `"6"`** (host-call shape gains `inferno_par_token_loop`).
- **Decode path untouched:** `lower_decode`/`lower_body` and `inferno-graph` (the oracle) are not modified. The dispatcher's `m <= 1` guard keeps decode-shaped work off the pool.
- **The outlined-body rule (malformed-IR hazard):** IR emitted inside an outlined `tok_body.*` function may reference ONLY (a) values derived from its own `ctx`/`t0`/`t1` params, (b) module-level globals and functions, (c) LLVM constants. Referencing an SSA instruction value from the calling function is malformed IR; the module-verify tests catch it — never "fix" that by widening ctx ad hoc without updating both pack and unpack sides.
- **Never call `profiled()` inside an outlined body** — the counter accumulation is a non-atomic load/add/store, safe only on the dispatcher thread (all dispatches join before the bracket closes).
- **Test runner:** use `mise run test` for full-workspace runs (CI parity, nextest). Known pre-existing quirk: plain `cargo test -p inferno-core` fails one pool-init test that nextest isolates correctly — not a regression, don't chase it.
- Run `mise run lint` before pushing (CI runs clippy `-D warnings`; `mise run test` does not).
- `mise run metal` (the quiet-hw verdict) is **operator-driven and paid** — NOT part of this plan's execution. The plan ends at green local gates + recorded follow-ups.

---

### Task 1: `inferno-pool` — `TokenBodyFn`, `JobKind::TokenLoop`, `Pool::par_token_loop`

**Files:**
- Modify: `crates/inferno-pool/src/pool.rs` (type + enum arm + method + unit tests; current landmarks: `AttnFn` at ~line 33, `JobKind` at ~71, `run_shard` at ~93, `Pool::par_attention` at ~477, tests mod at ~615)

**Interfaces:**
- Consumes: existing `shard_table_aligned(rows, threads, align)` (`crates/inferno-pool/src/shard.rs:20`), the `Job`/epoch/`remaining` publish protocol (copy `par_attention`'s shape verbatim).
- Produces: `pub type TokenBodyFn = unsafe extern "C" fn(*const u8, usize, usize)` and `pub unsafe fn Pool::par_token_loop(&self, body: TokenBodyFn, ctx: *const u8, m: usize)` — Task 2 and codegen (Task 4) rely on these exact signatures.

- [ ] **Step 1: Write the failing unit tests** — append to the `tests` mod at the bottom of `crates/inferno-pool/src/pool.rs`, after `attention_zero_tokens_is_a_noop` (~line 834), mirroring the `stamp_attn` family:

```rust
    /// Fake outlined token body with the real M4b.9 ABI: ctx is two usize
    /// words [out_ptr_bits, stride]; each token t writes its own disjoint
    /// out row — a deterministic function of (t, i), like the codegen
    /// bodies it stands in for.
    unsafe extern "C" fn stamp_tokens(ctx: *const u8, t0: usize, t1: usize) {
        let words = ctx as *const usize;
        // SAFETY: tests pass a 2-word ctx pack, live for the call.
        let out = unsafe { *words } as *mut f32;
        let stride = unsafe { *words.add(1) };
        for t in t0..t1 {
            for i in 0..stride {
                // SAFETY: out has m*stride elements and t < m per contract.
                unsafe { *out.add(t * stride + i) = (t * 31 + i) as f32 };
            }
        }
    }

    const TOK_STRIDE: usize = 5;

    fn tok_dispatch(pool: &Pool, m: usize) -> Vec<f32> {
        let mut out = vec![f32::NAN; m * TOK_STRIDE];
        let ctx = [out.as_mut_ptr() as usize, TOK_STRIDE];
        // SAFETY: ctx/out sized per stamp_tokens' expectations, live for the call.
        unsafe { pool.par_token_loop(stamp_tokens, ctx.as_ptr() as *const u8, m) };
        out
    }

    fn tok_expected(m: usize) -> Vec<f32> {
        (0..m * TOK_STRIDE)
            .map(|j| ((j / TOK_STRIDE) * 31 + j % TOK_STRIDE) as f32)
            .collect()
    }

    #[test]
    fn token_loop_parallel_matches_serial_expectation() {
        let pool = Pool::new(4);
        for m in [1, 2, 7, 63, 64, 100] {
            assert_eq!(tok_dispatch(&pool, m), tok_expected(m), "m={m}");
        }
    }

    #[test]
    fn token_loop_threads_exceeding_tokens_collapses() {
        let pool = Pool::new(16);
        assert_eq!(tok_dispatch(&pool, 3), tok_expected(3));
    }

    #[test]
    fn token_loop_capacity_one_runs_inline() {
        let pool = Pool::new(1);
        assert_eq!(tok_dispatch(&pool, 64), tok_expected(64));
    }

    #[test]
    fn token_loop_ignores_decode_cap() {
        // The decode cap applies to par_gemv only; token loops are prefill
        // work and shard over full active. Result identical either way.
        let pool = Pool::new(8);
        pool.set_decode_threads(1);
        assert_eq!(tok_dispatch(&pool, 64), tok_expected(64));
    }

    #[test]
    fn token_loop_zero_tokens_is_a_noop() {
        let pool = Pool::new(4);
        assert!(tok_dispatch(&pool, 0).is_empty());
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p inferno-pool token_loop 2>&1 | tail -20`
Expected: compile error E0599 — `no method named par_token_loop found for reference &Pool`.

- [ ] **Step 3: Implement.** Three edits to `crates/inferno-pool/src/pool.rs`:

(a) After the `AttnJob` struct (~line 65), add the ABI type:

```rust
/// The M4b.9 outlined token-body ABI: `(ctx, t0, t1)` runs tokens
/// `[t0, t1)` of a prefill tile. `ctx` is an opaque argument pack built by
/// the emitting module (codegen packs pointers + tile_start; only it knows
/// the layout — the pool just passes ctx through). Each token's writes are
/// disjoint rows, so thread count never changes output bits.
pub type TokenBodyFn = unsafe extern "C" fn(*const u8, usize, usize);
```

(b) Add the `JobKind` variant and `run_shard` arm:

```rust
    Attention(AttnJob),
    TokenLoop { body: TokenBodyFn, ctx: *const u8 },
```

and in `run_shard`'s match:

```rust
        // SAFETY: forwarding the caller's contract for the disjoint token span.
        JobKind::TokenLoop { body, ctx } => unsafe { body(ctx, start, end) },
```

(c) Add `Pool::par_token_loop` directly after `par_attention` (~line 533), copying its publish/join shape verbatim:

```rust
    /// Outlined token-span work across up to `active_threads()` lanes
    /// (M4b.9): splits the tile's `m` tokens into align-1 contiguous
    /// shards and calls `body(ctx, start, end)` once per shard. Each
    /// token's writes are disjoint rows computed by exactly one lane, so
    /// thread count never changes output bits. The M4b.5 decode cap does
    /// NOT apply — token loops are prefill work.
    ///
    /// # Safety
    /// `body` must be a valid `TokenBodyFn` whose contract holds for
    /// every token span within `0..m` given `ctx`; `ctx` and every buffer
    /// the body touches stay live and otherwise-untouched until this
    /// returns; per-token writes must be disjoint across tokens; calls
    /// must not overlap (one job at a time).
    pub unsafe fn par_token_loop(&self, body: TokenBodyFn, ctx: *const u8, m: usize) {
        if m == 0 {
            return;
        }
        let active = self.active_threads();
        let shards = shard_table_aligned(m, active, 1);
        if shards.len() == 1 {
            // SAFETY: caller contract covers the full token range.
            unsafe { body(ctx, 0, m) };
            return;
        }
        let n_worker = shards.len() - 1;
        let (s0, e0) = shards[0];
        let kind = JobKind::TokenLoop { body, ctx };
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
        unsafe { body(ctx, s0, e0) };
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

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p inferno-pool 2>&1 | tail -5`
Expected: all tests pass (the five new `token_loop_*` plus every existing test).

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-pool/src/pool.rs
git commit -m "M4b.9: JobKind::TokenLoop + Pool::par_token_loop (align-1 token shards)"
```

---

### Task 2: `inferno-pool` — C-ABI `inferno_par_token_loop` + fallback tests

**Files:**
- Modify: `crates/inferno-pool/src/lib.rs` (dispatcher after `inferno_par_attention` at ~line 261; re-export; crate doc)
- Create: `crates/inferno-pool/tests/par_token_loop_fallback.rs`

**Interfaces:**
- Consumes: `Pool::par_token_loop`, `TokenBodyFn` (Task 1); `GLOBAL`, `DISPATCH_CLAIMED` (existing statics in lib.rs).
- Produces: `#[unsafe(no_mangle)] pub unsafe extern "C" fn inferno_par_token_loop(body: TokenBodyFn, ctx: *const u8, m: usize)` — the symbol compiled code calls; Tasks 3–6 depend on the exact name and arg order `(body, ctx, m)`.

- [ ] **Step 1: Write the failing integration test** — create `crates/inferno-pool/tests/par_token_loop_fallback.rs`, mirroring `par_attention_fallback.rs` (a fresh test binary is its own process, so the global pool is guaranteed absent):

```rust
//! `inferno_par_token_loop` without an initialized global pool: the entry
//! point must degrade to the serial full-range body call (and take the
//! m == 1 direct path) — this file never calls `init_global`, and an
//! integration test binary is its own process, so the pool is guaranteed
//! absent. The stub's coercion to `TokenBodyFn` in these calls is also the
//! ABI drift guard: there is no inferno-kernels symbol to coerce (bodies
//! are codegen-emitted), so this stub plays that role.
#![allow(unsafe_code)] // FFI entry-point tests; same justification as the sibling test files.

use inferno_pool::{TokenBodyFn, inferno_par_token_loop};

unsafe extern "C" fn stamp_tokens(ctx: *const u8, t0: usize, t1: usize) {
    let words = ctx as *const usize;
    // SAFETY: tests pass a 2-word ctx pack, live for the call.
    let out = unsafe { *words } as *mut f32;
    let stride = unsafe { *words.add(1) };
    for t in t0..t1 {
        for i in 0..stride {
            // SAFETY: out has m*stride elements and t < m per contract.
            unsafe { *out.add(t * stride + i) = (t * 31 + i) as f32 };
        }
    }
}

const STRIDE: usize = 5;

fn dispatch(m: usize) -> Vec<f32> {
    let body: TokenBodyFn = stamp_tokens; // ABI coercion is part of the test
    let mut out = vec![f32::NAN; m * STRIDE];
    let ctx = [out.as_mut_ptr() as usize, STRIDE];
    // SAFETY: ctx/out sized per stamp_tokens' expectations, live for the call.
    unsafe { inferno_par_token_loop(body, ctx.as_ptr() as *const u8, m) };
    out
}

fn expected(m: usize) -> Vec<f32> {
    (0..m * STRIDE)
        .map(|j| ((j / STRIDE) * 31 + j % STRIDE) as f32)
        .collect()
}

#[test]
fn uninitialized_pool_falls_back_to_serial_full_range() {
    for m in [2, 7, 64] {
        assert_eq!(dispatch(m), expected(m), "m={m}");
    }
}

#[test]
fn m1_takes_the_direct_path() {
    // Decode-shaped span (and the T=1 prefill tile): one token, computed
    // correctly with no pool involvement by construction.
    assert_eq!(dispatch(1), expected(1));
}

#[test]
fn m0_is_a_noop() {
    assert!(dispatch(0).is_empty());
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p inferno-pool --test par_token_loop_fallback 2>&1 | tail -10`
Expected: compile error E0432 — `inferno_par_token_loop` not found in `inferno_pool`.

- [ ] **Step 3: Implement the dispatcher.** In `crates/inferno-pool/src/lib.rs`:

(a) Update the re-export line to include the new type:

```rust
pub use pool::{AttnFn, AttnJob, GemmFn, GemvFn, Pool, TokenBodyFn};
```

(b) Update the crate doc's first line — "the three `inferno_par_{gemv,gemm,attention}` dispatchers" becomes "the four `inferno_par_{gemv,gemm,attention,token_loop}` dispatchers".

(c) Append after `inferno_par_attention`:

```rust
/// Host dispatcher for outlined serial-tail token loops (M4b.9). Same
/// single-dispatcher guard + serial fallback as [`inferno_par_gemv`];
/// shares `DISPATCH_CLAIMED` deliberately — within one forward pass all
/// pool dispatches are issued serially and never overlap, so one guard
/// suffices. `m <= 1` (the T=1 prefill tile tail) takes a direct body
/// call with no CAS and no job publish — decode never calls this
/// dispatcher at all (its codegen lowers ops inline, single-token). On
/// the CAS-loss (or uninitialized-pool) path this runs the body once
/// over the full token range, bit-identical to the pooled path since
/// each token's rows are written by a single body invocation either way.
///
/// A panic inside the dispatcher or body aborts the process at this
/// `extern "C"` boundary — there is no unwind across FFI.
///
/// # Safety
/// Same contract as [`Pool::par_token_loop`]; additionally `body` must
/// be a valid non-null function pointer with the M4b.9 token-body ABI
/// and `ctx` the pack that body expects. Generated code guarantees all
/// of this by construction (M3 trust model).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_par_token_loop(body: TokenBodyFn, ctx: *const u8, m: usize) {
    if m == 0 {
        return;
    }
    if m == 1 {
        // SAFETY: forwarding the caller's contract for the single token.
        unsafe { body(ctx, 0, 1) };
        return;
    }
    match GLOBAL.get() {
        Some(p) => {
            if DISPATCH_CLAIMED
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: forwarding the caller's contract unchanged.
                unsafe { p.par_token_loop(body, ctx, m) };
                DISPATCH_CLAIMED.store(false, Ordering::Release);
            } else {
                // Lost the race for the pool's single dispatcher slot: run
                // the body serially over the full token range instead of
                // overlapping another thread's in-flight pool dispatch.
                // SAFETY: forwarding the caller's contract for the full range.
                unsafe { body(ctx, 0, m) };
            }
        }
        // SAFETY: forwarding the caller's contract for the full range.
        None => unsafe { body(ctx, 0, m) },
    }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p inferno-pool 2>&1 | tail -5`
Expected: all pass, including the three new fallback tests.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-pool/src/lib.rs crates/inferno-pool/tests/par_token_loop_fallback.rs
git commit -m "M4b.9: inferno_par_token_loop C-ABI dispatcher + fallback tests"
```

---

### Task 3: Declare the dispatcher in codegen, bump `HOST_ABI_VERSION`, extend both retention lists

**Files:**
- Modify: `crates/inferno-codegen/src/llvm/mod.rs` (`declare_kernels` ends ~line 195; `scaffold_verifies` test ~line 357)
- Modify: `crates/inferno-codegen/src/lib.rs:12-18` (`HOST_ABI_VERSION`)
- Modify: `crates/inferno-core/src/artifact.rs:528-552` (`ensure_kernels_linked`)
- Modify: `crates/inferno-codegen/tests/differential.rs:49-81` (`retain_kernel_symbols`)

**Interfaces:**
- Consumes: `inferno_pool::inferno_par_token_loop` (Task 2).
- Produces: the LLVM extern declaration `void inferno_par_token_loop(ptr body, ptr ctx, i64 m)` resolvable via `self.module.get_function("inferno_par_token_loop")` — Task 4's emitter depends on it; both retention lists keep the symbol exported for `dlopen`.

- [ ] **Step 1: Extend the scaffold test (failing first).** In `scaffold_verifies` (llvm/mod.rs), after the `inferno_par_attention` assertion add:

```rust
        assert!(ir.contains("inferno_par_token_loop"));
```

Run: `cargo test -p inferno-codegen scaffold_verifies 2>&1 | tail -5`
Expected: FAIL on the new assertion.

- [ ] **Step 2: Declare the extern.** At the end of `declare_kernels`, after the `inferno_par_attention` block:

```rust
        // void inferno_par_token_loop(ptr body, ptr ctx, i64 m)
        // — the M4b.9 serial-tail dispatcher; `body` is a module-emitted
        // outlined token-span function (`tok_body.*`) and `ctx` its opaque
        // argument pack (layout private to ops.rs). Shards the tile's m
        // tokens align-1, exactly like par_attention.
        let par_tok_ty = void.fn_type(&[ptr.into(), ptr.into(), i64_t.into()], false);
        self.module.add_function(
            "inferno_par_token_loop",
            par_tok_ty,
            Some(Linkage::External),
        );
```

- [ ] **Step 3: Bump the ABI version.** In `crates/inferno-codegen/src/lib.rs` replace the constant and extend its doc:

```rust
/// Version of the host-symbol surface generated code links against
/// (kernel symbols + `inferno_par_{gemv,gemm,attention,token_loop}` + the
/// profiler global). Folded into `inferno-core`'s artifact cache key.
/// "6" = M4b.9's `inferno_par_token_loop` dispatch; "5" = M4b.8's
/// `inferno_par_attention` dispatch; "4" was M4b.3's attention kernel
/// symbols (`inferno_attention_f32_{scalar,avx2}`); "3" was M4b.2's
/// GEMM dispatch + optional profiling (v2 was M4b.1's `inferno_par_gemv`).
pub const HOST_ABI_VERSION: &str = "6";
```

- [ ] **Step 4: Extend both retention lists.** Append to `ensure_kernels_linked` (artifact.rs) and to `retain_kernel_symbols` (differential.rs), in each case after the `inferno_par_attention` line:

```rust
    p(inferno_pool::inferno_par_token_loop as *const ());
```

Also update `retain_kernel_symbols`' doc comment list to mention `inferno_par_token_loop` (M4b.9).

- [ ] **Step 5: Run to verify pass**

Run: `cargo test -p inferno-codegen scaffold_verifies 2>&1 | tail -5 && cargo build -p inferno-core 2>&1 | tail -3`
Expected: scaffold test PASS; core builds clean.

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-codegen/src/llvm/mod.rs crates/inferno-codegen/src/lib.rs crates/inferno-core/src/artifact.rs crates/inferno-codegen/tests/differential.rs
git commit -m "M4b.9: declare inferno_par_token_loop, HOST_ABI_VERSION 5->6, retention lists"
```

---

### Task 4: Codegen outlining helper + generic elementwise arm through the dispatcher

**Files:**
- Modify: `crates/inferno-codegen/src/llvm/ops.rs` — `Codegen` struct (~line 112) gains a counter field; new `par_token_loop` method near the control-flow helpers (~line 465); `lower_tile`'s generic `_ =>` arm (~lines 742-750)

**Interfaces:**
- Consumes: `inferno_par_token_loop` declaration (Task 3); existing `TileEnv`, `tile_frame`, `range_loop`, `entry_alloca`, `byte_ptr`, `load_i64`.
- Produces: `fn par_token_loop(&self, env: &TileEnv<'c>, tile_start: IntValue<'c>, m: IntValue<'c>, label: &str, per_token: impl FnOnce(&Self, &TileEnv<'c>, IntValue<'c>, IntValue<'c>))` — Tasks 5 and 6 call it with `(cg, body_env, ti, row)` closure args. ctx word layout (fixed, private to ops.rs): `[ptrtoint(tokens), pos_off, ptrtoint(weights), ptrtoint(kv), ptrtoint(arena), tile_start]`.

- [ ] **Step 1: Add the counter field.** In the `Codegen` struct after `rope_freqs`:

```rust
    /// Monotonic counter naming outlined `tok_body.*` functions uniquely.
    outlined: RefCell<usize>,
```

and in `Codegen::new`'s struct literal, after `rope_freqs: RefCell::new(HashMap::new()),`:

```rust
            outlined: RefCell::new(0),
```

- [ ] **Step 2: Add the outlining helper.** Insert after `tile_loop` (before the `// ---- entry points` divider, ~line 574):

```rust
    /// Outline `per_token` into a private `void tok_body.<label>.<n>(ptr
    /// ctx, i64 t0, i64 t1)` function and emit ONE
    /// `inferno_par_token_loop(body, ctx, m)` dispatch sharding the tile's
    /// `m` tokens across pool lanes (M4b.9). The ctx pack is 6 i64 words on
    /// the caller's stack — ptrtoint(tokens), pos_off, ptrtoint(weights),
    /// ptrtoint(kv), ptrtoint(arena), tile_start — rebuilt per dispatch;
    /// the pool treats ctx as opaque, only this emitter knows the layout.
    ///
    /// `per_token(cg, env, ti, row)` is emitted POSITIONED INSIDE the
    /// outlined function, once, looped over the span: `ti` is the
    /// tile-local token index, `row = tile_start + ti` the arena row. It
    /// must derive every runtime value from the rebuilt `env`/`ti`/`row`,
    /// module-level globals/functions, or constants — referencing an SSA
    /// value from the calling function is malformed IR (module
    /// verification fails). Never call `profiled` inside `per_token`: the
    /// counter accumulation is non-atomic and belongs to the dispatcher
    /// thread; brackets wrap the dispatch call in the caller instead.
    ///
    /// Bit-neutrality: each token's writes are disjoint rows produced by
    /// the identical loop-body IR the serial `range_loop(m)` emitted, and
    /// exactly one lane runs each token — thread count and shard layout
    /// cannot change output bits (M4b.8 argument, verbatim).
    fn par_token_loop(
        &self,
        env: &TileEnv<'c>,
        tile_start: IntValue<'c>,
        m: IntValue<'c>,
        label: &str,
        per_token: impl FnOnce(&Self, &TileEnv<'c>, IntValue<'c>, IntValue<'c>),
    ) {
        // Caller side: pack ctx on the stack (6 stores per dispatch).
        let ctx = self.entry_alloca(self.i64_t.array_type(6), "tokctx");
        let p2i = |p: PointerValue<'c>| {
            self.builder
                .build_ptr_to_int(p, self.i64_t, "p2i")
                .unwrap()
        };
        let fields = [
            p2i(env.tokens),
            env.pos_off,
            p2i(env.weights),
            p2i(env.kv),
            p2i(env.arena),
            tile_start,
        ];
        for (i, v) in fields.iter().enumerate() {
            let slot = self.byte_ptr(ctx, self.const_i64((i * 8) as u64));
            self.builder.build_store(slot, *v).unwrap();
        }

        // Emit the outlined body function.
        let n = {
            let mut c = self.outlined.borrow_mut();
            *c += 1;
            *c
        };
        let sanitized: String = label
            .chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
            .collect();
        let fn_ty = self.ctx.void_type().fn_type(
            &[self.ptr_t.into(), self.i64_t.into(), self.i64_t.into()],
            false,
        );
        let f = self.module.add_function(
            &format!("tok_body.{sanitized}.{n}"),
            fn_ty,
            Some(Linkage::Private),
        );
        let saved = self.builder.get_insert_block().unwrap();
        let entry = self.ctx.append_basic_block(f, "entry");
        self.builder.position_at_end(entry);

        let ctx_p = f.get_nth_param(0).unwrap().into_pointer_value();
        let t0 = f.get_nth_param(1).unwrap().into_int_value();
        let t1 = f.get_nth_param(2).unwrap().into_int_value();
        let field = |i: usize| {
            let p = self.byte_ptr(ctx_p, self.const_i64((i * 8) as u64));
            self.load_i64(p)
        };
        let i2p = |v: IntValue<'c>| {
            self.builder
                .build_int_to_ptr(v, self.ptr_t, "i2p")
                .unwrap()
        };
        let body_env = TileEnv {
            tokens: i2p(field(0)),
            pos_off: field(1),
            weights: i2p(field(2)),
            kv: i2p(field(3)),
            arena: i2p(field(4)),
        };
        let ts = field(5);
        let span = self.builder.build_int_sub(t1, t0, "span").unwrap();
        self.range_loop(span, |cg, i| {
            let ti = cg.add(i, t0);
            let row = cg.add(ts, ti);
            per_token(cg, &body_env, ti, row);
        });
        self.builder.build_return(None).unwrap();
        self.builder.position_at_end(saved);

        // Caller side: one dispatch over the tile.
        let pfn = self
            .module
            .get_function("inferno_par_token_loop")
            .expect("par token-loop dispatcher declared");
        self.builder
            .build_call(
                pfn,
                &[
                    f.as_global_value().as_pointer_value().into(),
                    ctx.into(),
                    m.into(),
                ],
                "par_token_loop",
            )
            .unwrap();
    }
```

- [ ] **Step 3: Route the generic elementwise arm through it.** Replace `lower_tile`'s `_ =>` arm (currently the profiled `range_loop` over `tile_frame` + `lower_step`):

```rust
                    _ => {
                        self.profiled(&label, |cg| {
                            cg.par_token_loop(env, tile_start, m, &label, |cg, benv, _ti, row| {
                                let frame = cg.tile_frame(benv, row);
                                cg.lower_step(&frame, step);
                            });
                        });
                    }
```

- [ ] **Step 4: Run the codegen gates**

Run: `cargo test -p inferno-codegen 2>&1 | tail -8`
Expected: ALL PASS — `lowered_module_verifies_on_tiny` and `profiled_module_verifies_and_exports_counters` (catch any cross-function SSA reference immediately), `differential_tiny_gguf`/`_mlx`/`_bias`, `prefill_tiling_is_bit_invariant_to_tile_size`, `prefill_is_bit_invariant_to_thread_count`.

- [ ] **Step 5: Run the artifact-level differential**

Run: `cargo nextest run -p inferno-core --test artifact 2>&1 | tail -5`
Expected: PASS (plain `cargo test -p inferno-core` has the known pre-existing pool-init quirk; use nextest).

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-codegen/src/llvm/ops.rs
git commit -m "M4b.9: outline elementwise tile ops into tok_body fns dispatched via inferno_par_token_loop"
```

---

### Task 5: KV-append through the dispatcher + its own `kv_append` profile bracket

**Files:**
- Modify: `crates/inferno-codegen/src/profile.rs` (label const + `assign_slots` + test)
- Modify: `crates/inferno-codegen/src/llvm/ops.rs` (`lower_tile`'s `Step::Attention` arm ~line 737; `lower_tile_attention` ~line 857 loses its append loop; new `lower_tile_kv_append`)

**Interfaces:**
- Consumes: `par_token_loop` (Task 4), existing `lower_kv_append(frame, k, v, layer)` (~line 1283).
- Produces: `pub const KV_APPEND_LABEL: &str = "kv_append"` in `profile.rs`; `fn lower_tile_kv_append(&self, env, step, tile_start, m)` in ops.rs. Profile tables gain a `kv_append` row (intentional continuity change: the `attention` row shrinks by the split-out share from this milestone on).

- [ ] **Step 1: Failing profile test.** In `profile.rs`'s `slots_aggregate_matmuls_across_layers` test, extend the elementwise-kind loop:

```rust
        for kind in ["rmsnorm", "rope", "swiglu", "add", "attention", "kv_append"] {
```

Run: `cargo test -p inferno-codegen slots_aggregate 2>&1 | tail -5`
Expected: FAIL — `kv_append` count is 0.

- [ ] **Step 2: Intern the label.** In `profile.rs` add above `assign_slots`:

```rust
/// The prefill lowering brackets the tile's KV-append separately from the
/// attention read (M4b.9), so the label is interned alongside every
/// `Attention` step rather than derived from a `Step` kind.
pub const KV_APPEND_LABEL: &str = "kv_append";
```

and inside `assign_slots`'s inner loop, after `slots.intern(step_label(...));`:

```rust
            if matches!(step, Step::Attention { .. }) {
                slots.intern(KV_APPEND_LABEL.into());
            }
```

Run: `cargo test -p inferno-codegen slots_aggregate 2>&1 | tail -5` — Expected: PASS.

- [ ] **Step 3: Split the attention arm.** In `lower_tile`, replace the `Step::Attention` arm:

```rust
                    Step::Attention { .. } => {
                        self.profiled(crate::profile::KV_APPEND_LABEL, |cg| {
                            cg.lower_tile_kv_append(env, step, tile_start, m)
                        });
                        self.profiled(&label, |cg| {
                            cg.lower_tile_attention(env, step, tile_start, m)
                        });
                    }
```

Add the new method directly above `lower_tile_attention`:

```rust
    /// The write half of the tile's attention step (M4b.9): shard the
    /// tile's `m` KV-appends across pool lanes. Bit-safe by the same
    /// argument as the parallel read: each token writes only its own KV
    /// row (`pos0 + t`), rows are disjoint across tokens, and the
    /// dispatch joins before `lower_tile_attention` issues the parallel
    /// read — so every KV row `<= pos_i` is in place when token i's
    /// causal read runs, exactly as with the old serial append loop.
    fn lower_tile_kv_append(
        &self,
        env: &TileEnv<'c>,
        step: &Step,
        tile_start: IntValue<'c>,
        m: IntValue<'c>,
    ) {
        let Step::Attention { k, v, layer, .. } = step else {
            unreachable!("lower_tile_kv_append called on non-Attention step")
        };
        let (k, v, layer) = (*k, *v, *layer);
        self.par_token_loop(
            env,
            tile_start,
            m,
            crate::profile::KV_APPEND_LABEL,
            |cg, benv, _ti, row| {
                let frame = cg.tile_frame(benv, row);
                cg.lower_kv_append(&frame, k, v, layer);
            },
        );
    }
```

Then delete the serial append loop at the head of `lower_tile_attention` (the `self.range_loop(m, ...) { ... lower_kv_append ... }` block at ~lines 877-881) and update `lower_tile_attention`'s doc comment first sentence to: "Tiled prefill attention read (M4b.8/M4b.9): the tile's k/v was appended by `lower_tile_kv_append`'s dispatch (already joined), then ONE `inferno_par_attention` call shards the tile's `m` tokens across pool lanes." Also update the `lower_tile` doc comment (~lines 702-715): the attention sentence becomes "Attention appends the whole tile's k/v via one token-loop dispatch, then issues a single parallel read; both are bit-identical and T-invariant because each token's rows are written by exactly one lane and a token's causal read never reaches KV rows past its own position."

- [ ] **Step 4: Run the gates**

Run: `cargo test -p inferno-codegen 2>&1 | tail -8 && cargo nextest run -p inferno-core --test artifact 2>&1 | tail -5`
Expected: ALL PASS — in particular the tiling gate (append restructure is the risky spot) and both module-verify tests.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-codegen/src/profile.rs crates/inferno-codegen/src/llvm/ops.rs
git commit -m "M4b.9: parallel KV-append with its own kv_append profile bracket"
```

---

### Task 6: Activation-quantize panel fill through the dispatcher + its own `quantize` bracket

**Files:**
- Modify: `crates/inferno-codegen/src/llvm/ops.rs` (`lower_tile`'s `Step::Gemv` arm ~line 727; `lower_gemm` ~line 770)

**Interfaces:**
- Consumes: `par_token_loop` (Task 4); existing `arena_row_ptr_at`, `act_scratch_ptr_row0`, `packed_act_bytes`, `quantize_symbol`. The `"quantize"` profile label already has a slot (`Step::Quantize` steps exist in every quantized-weight loopir; `assign_slots` interns them).
- Produces: `lower_gemm(&self, env, step, tile_start, m, label: &str)` — signature gains the label; it now emits its own two profile brackets (`quantize` then the matmul label) instead of being wrapped by `lower_tile`.

- [ ] **Step 1: Re-plumb the call site.** In `lower_tile`, replace the `Step::Gemv` arm:

```rust
                    Step::Gemv { .. } => {
                        self.lower_gemm(env, step, tile_start, m, &label);
                    }
```

- [ ] **Step 2: Restructure `lower_gemm`.** Change its signature to take `label: &str` as the last parameter, and replace the quantized-panel branch and the dispatch call. The full new body after the existing `let src = ...; let k_c = ...; let rows_c = ...; let gemm_sym = ...;` prologue (`k_c`/`rows_c` stay — constants are usable from any function):

```rust
        let panel_ptr = if pw.dtype != inferno_formats::DType::F32 {
            // Quantize each token's source row into scratch[ti * act_len(k)],
            // sharded across pool lanes (M4b.9) — each token fills only its
            // own panel row, so shards write disjoint bytes. Bracketed as
            // "quantize" so the profile attributes the panel fill separately
            // from the matmul dispatch (it was folded into the matmul row
            // before M4b.9).
            let act_row = Self::packed_act_bytes(&pw.dtype, *k);
            let qsym = Self::quantize_symbol(&pw.dtype, pw.isa);
            let qfn = self
                .module
                .get_function(&qsym)
                .expect("quantize kernel declared (Task 8)");
            let kk = *k as u64;
            self.profiled("quantize", |cg| {
                cg.par_token_loop(env, tile_start, m, "quantize", |cg, benv, ti, row| {
                    let src_ptr = cg.arena_row_ptr_at(benv.arena, src, row);
                    let scratch = cg.act_scratch_ptr_row0(benv.arena);
                    let off = cg
                        .builder
                        .build_int_mul(ti, cg.const_i64(act_row), "actoff")
                        .unwrap();
                    let dst = cg.byte_ptr(scratch, off);
                    cg.builder
                        .build_call(
                            qfn,
                            &[src_ptr.into(), dst.into(), cg.const_i64(kk).into()],
                            "quantize",
                        )
                        .unwrap();
                });
            });
            self.act_scratch_ptr_row0(env.arena)
        } else {
            // F32 weight: the source rows are already a contiguous f32 panel.
            self.arena_row_ptr_at(env.arena, src, tile_start)
        };

        let y_ptr = self.arena_row_ptr_at(env.arena, *out, tile_start);
        let w_ptr = self.byte_ptr(env.weights, self.const_i64(pw.offset as u64));
        let gfn = self
            .module
            .get_function(&gemm_sym)
            .expect("gemm kernel declared (Task 5)");
        let pfn = self
            .module
            .get_function("inferno_par_gemm")
            .expect("par gemm dispatcher declared");
        self.profiled(label, |cg| {
            cg.builder
                .build_call(
                    pfn,
                    &[
                        gfn.as_global_value().as_pointer_value().into(),
                        y_ptr.into(),
                        panel_ptr.into(),
                        w_ptr.into(),
                        k_c.into(),
                        m.into(),
                        rows_c.into(),
                    ],
                    "par_gemm",
                )
                .unwrap();
        });
        // Bias (if any) is a separate Step handled in the elementwise m-loop.
```

Note the two deliberate details: the body recomputes `scratch` from `benv.arena` (the caller's `scratch` pointer would be a cross-function SSA reference — malformed IR), and the panel pointer returned to the caller is a fresh caller-side `act_scratch_ptr_row0(env.arena)`. Update `lower_gemm`'s doc comment to mention the sharded fill and the new brackets.

- [ ] **Step 3: Run the gates**

Run: `cargo test -p inferno-codegen 2>&1 | tail -8 && cargo nextest run -p inferno-core --test artifact 2>&1 | tail -5`
Expected: ALL PASS — `differential_tiny_gguf` exercises the quantized-panel path (tiny.gguf is quantized), the profiled-build bit-invariance test exercises the new brackets.

- [ ] **Step 4: Commit**

```bash
git add crates/inferno-codegen/src/llvm/ops.rs
git commit -m "M4b.9: parallel activation-quantize panel fill with its own quantize bracket"
```

---

### Task 7: Full-workspace gates, profile-surface check, AGENTS.md note

**Files:**
- Modify: `AGENTS.md` (the M4b threading bullet)
- Verify only: `crates/inferno-core` profile table plumbing

**Interfaces:**
- Consumes: everything above.
- Produces: a green workspace + the repo front-door note future sessions rely on.

- [ ] **Step 1: Confirm the profile table surfaces the new labels.** The CLI's `--profile` table is driven by `ProfileSlots.labels` from the same `assign_slots` codegen used — confirm no hardcoded label list exists:

Run: `grep -rn "assign_slots\|ProfileSlots\|kv_append" crates/inferno-core/src cli/ --include=*.rs | head -20`
Expected: consumers iterate `slots.labels` (no fixed label enumeration). If any hardcoded list of op labels turns up, add `kv_append` to it and note the file in the commit message.

- [ ] **Step 2: Full test suite**

Run: `mise run test 2>&1 | tail -10`
Expected: all green, zero tolerance edits anywhere in the diff (`git diff main -- '**/tolerance*' | wc -l` → 0 lines).

- [ ] **Step 3: Lint**

Run: `mise run lint 2>&1 | tail -10`
Expected: clean (clippy `-D warnings`).

- [ ] **Step 4: Update AGENTS.md.** In the M4b bullet that ends "…and `m <= 1` calls bypass the pool entirely.", append one sentence:

```
Since M4b.9 the serial tail (rmsnorm/rope/add/swiglu/bias/embed, KV-append,
activation-quantize panel fill) is token-sharded too: codegen outlines each
per-token body into a private `tok_body.*` function dispatched through
`inferno_par_token_loop` (opaque-ctx ABI, align-1 shards, `m <= 1` direct)
— outlined bodies must never reference caller SSA values or call the
profiler, and the `kv_append` dispatch always joins before the attention
read is issued.
```

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md
git commit -m "M4b.9: record the token-loop dispatch invariants in AGENTS.md"
```

- [ ] **Step 6: Record the follow-ups that are NOT in this plan** (in the PR description, not code): (1) `mise run bench-compiled` single-thread gate — pending the first post-merge nightly (M4b.8 precedent), it guards the outlining-regression risk; (2) the quiet-hw ≥6x @ t=12 verdict via `mise run metal` — operator-driven, paid, recorded as amendments in the M4b.1 ledger and the M4b.9 spec.

---

## Self-Review

- **Spec coverage:** op coverage (Task 4 generic arm = rmsnorm/rope/add/swiglu/bias/embed; Task 5 append; Task 6 quantize) ✓; mechanism/body ABI/pool entry (Tasks 1-2) ✓; kernel-ABI bump + retention (Task 3) ✓; profiling split (Tasks 5-6) ✓; testing plan (pool unit tests Task 1, fallback file Task 2, ABI-coercion-via-stub noted in the fallback file's doc since no inferno-kernels symbol exists to coerce, threads/tiling gates rerun Tasks 4-6) ✓; verification protocol items outside local execution recorded (Task 7 Step 6) ✓; interpreter/tolerances/decode untouched (Global Constraints) ✓.
- **Deviation from spec, recorded:** the spec's "extend par_rig.rs ABI-coercion to TokenBodyFn" is realized in `par_token_loop_fallback.rs` (the `let body: TokenBodyFn = stamp_tokens;` binding) because the body fns are codegen-emitted — there is no `inferno-kernels` symbol for par_rig to coerce.
- **Type consistency:** `TokenBodyFn = unsafe extern "C" fn(*const u8, usize, usize)` (Tasks 1/2), LLVM decl `(ptr, ptr, i64) -> void` with call args `(body, ctx, m)` (Tasks 3/4), ctx word order `[tokens, pos_off, weights, kv, arena, tile_start]` fixed in Task 4 and never re-declared elsewhere ✓; `par_token_loop` closure shape `(cg, benv, ti, row)` used identically in Tasks 4/5/6 ✓; `lower_gemm` label-parameter change is confined to Task 6 and its only call site is re-plumbed in the same task ✓.
