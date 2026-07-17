# M4b.14 Prefill Attention Query-Blocking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-query-token prefill attention kernel with a query-blocked kernel that streams each visible K and V vector once per block of query tokens instead of once per token, bit-identically, then gate a second attention lever on a fresh profile.

**Architecture:** Query blocking lives entirely below the `inferno_par_attention` dispatch boundary — in the kernel (`inferno-kernels`) and the pool's per-lane driver (`inferno-pool`). A new block kernel takes a shard of `m_block` query tokens plus row strides and, per head, runs a three-pass loop (scores / softmax+normalize / output) that reuses each K and V vector across the block's rows. Because blocking only reorders the *query axis* and never any within-token reduction, the output is bitwise-identical to the current per-token kernel for any block length — the same structural argument M4b.11 used for the head axis. Codegen changes only the kernel symbol it passes and bumps the host-ABI version so stale artifacts recompile.

**Tech Stack:** Rust (workspace crates `inferno-kernels`, `inferno-pool`, `inferno-codegen`, `inferno-core`), AVX2+FMA intrinsics, inkwell/LLVM 18 codegen, proptest rig, insta snapshots, `cargo nextest` via `mise run test`, quiet-hw bench scripts (`scripts/quiet-hw/`).

## Global Constraints

- **`inferno-kernels` scalar and SIMD variants must stay bit-identical** — the rig asserts exact `to_bits()` equality per ISA. The query-blocked kernel joins this invariant.
- **`gemm(m=1)` / block-length-1 bit-equals the per-token path** — `qblock(m_block=1)` must equal `inferno_attention_f32_{isa}` bit-for-bit; decode (`inferno_par_attention_heads`) and the `m == 1` prefill special case depend on it.
- **`attn_rel_tol()` (`inferno-graph/src/tolerance.rs`) is NOT touched for Lever 1** — the block kernel is bit-identical to the kernel the tolerance was armed against; the compiled-vs-interpreter differential must stay green with zero tolerance change. Never loosen a tolerance to make a red test green (standing `LOGIT_TIE_EPSILON`/`gemv_rel_tol` rule).
- **`inferno-kernels` and `inferno-core`/`inferno-pool` are the only crates allowed `unsafe`** — kernels are `extern "C"`, alloc-free in the hot path (scratch is caller-provided).
- **Tensor shapes are row-major, outermost first.** `head_dim` is a multiple of 8 (the 8-lane `dot8`/`reduce8` partition depends on it).
- **Workflows are mise tasks:** `mise run test` / `lint` / `differential`. Don't hand-roll cargo invocations in CI/docs. Run `mise run lint` (clippy `-D warnings`) before pushing — `mise run test` skips clippy.
- **`inferno bench` and all quiet-hw gates are a manual protocol, never a CI gate:** quiet hardware, devenv shell, release build; record each data point in a spec §Amendments section verbatim, never edit a recorded point.
- **Phase scope: prefill attention (`m > 1`) only.** Decode attention and every GEMM/GEMV path are untouched. No tg claim is made. KV stays f32 (M3 invariant) — no KV dtype change.
- **Criterion model:** `qwen2.5-0.5b-instruct-q8_0.gguf` (qwen2, 14 heads / 2 KV heads / head_dim 64, 24 layers), at `/home/dev/.cache/inferno-tests/qwen2.5-0.5b-instruct-q8_0.gguf`.
- **Machines:** 16c `d2.c1.medium` (Ice Lake SP 6336Y) + 8c `s2.c2.medium` (Rocket Lake E-2388G). No parallel PNAP provisions (403); run `metal-gc` to zero servers after every session.

**Spec:** `docs/superpowers/specs/2026-07-17-m4b14-prefill-attention-query-blocking-design.md`

---

## File Structure

- **`crates/inferno-kernels/src/attention.rs`** — add the query-blocked kernels (`inferno_attention_f32_scalar_qblock`, `inferno_attention_f32_avx2_qblock`) and their shared cores, alongside the existing per-token and `_hspan` kernels. The per-token kernels stay (rig oracle + the reference the block kernel is proven equal to).
- **`crates/inferno-kernels/tests/rig.rs`** — add block-kernel drivers + bit-identity property tests in the `Attention` section.
- **`crates/inferno-pool/src/pool.rs`** — define the `AttnBlockFn` type alias next to `AttnFn`/`AttnHspanFn`; change `AttnJob.kernel` to it; rewrite `run_attn_span` to one block-kernel call per shard; update the `stamp_attn` fake kernel + `attn_dispatch` in the pool tests. (The pool owns these fn-pointer type aliases — `inferno-kernels`' registry `AttnFn`/`attention_kernel` have no consumers and are left untouched.)
- **`crates/inferno-pool/src/lib.rs`** — re-export `AttnBlockFn`; change `inferno_par_attention`'s `kernel` parameter type and (unchanged) the `m == 1` path; the pool's `AttnFn`/`AttnHspanFn` public types stay.
- **`crates/inferno-codegen/src/loopir.rs`** — `attention_symbol` returns the `_qblock` symbol.
- **`crates/inferno-codegen/src/llvm/mod.rs`** — declare the `_qblock` kernel extern (14-param signature).
- **`crates/inferno-codegen/src/lib.rs`** — bump `HOST_ABI_VERSION` "7" → "8".
- **`crates/inferno-core/src/artifact.rs`** — register the two `_qblock` symbols in the dlopen symbol table.
- **`crates/inferno-codegen/src/llvm/snapshots/`** — insta snapshots that reference the attention symbol get regenerated (reviewed, never blind-accepted).
- **`scripts/quiet-hw/gate-prefill-attn-split.sh`** (new) + a feature-gated `attn:scores`/`attn:softmax`/`attn:output` sub-bracket instrument for the mid-milestone gate.

---

## Task 1: Scalar query-blocked attention kernel

**Files:**
- Modify: `crates/inferno-kernels/src/attention.rs` (add `attn_core_scalar_qblock` + `inferno_attention_f32_scalar_qblock` after `attn_core_scalar`)
- Test: `crates/inferno-kernels/tests/rig.rs` (add `attn_kernel_scalar_qblock` driver + `attention_qblock_scalar_matches_per_token` proptest in the `Attention` section)

**Interfaces:**
- Consumes: existing `dot8`, `reduce8`, `expf::expf_scalar` in `attention.rs`; the existing `inferno_attention_f32_scalar` (as the per-token reference in the test).
- Produces: `pub unsafe extern "C" fn inferno_attention_f32_scalar_qblock(out: *mut f32, q: *const f32, kv: *mut f32, scores: *mut f32, kv_base: usize, v_off: usize, pos0: usize, m_block: usize, kv_dim: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize, q_stride: usize, out_stride: usize)`. Contract: computes attention for query rows `[0, m_block)` where row `r` is at query position `pos0 + r`, `out`/`q` rows are `out_stride`/`q_stride` apart, `scores` is scratch of at least `m_block * (pos0 + m_block)` f32, `kv` holds every position `< pos0 + m_block` already appended. Bitwise-identical, per row, to `inferno_attention_f32_scalar` called once per token.

- [ ] **Step 1: Write the failing test**

Add to `crates/inferno-kernels/tests/rig.rs`, in the `// ---------- Attention ----------` section (after `attn_kernel_avx2`):

```rust
/// Drive the query-blocked scalar kernel over a block of `m_block` query
/// tokens at positions `pos0..pos0+m_block`. The KV cache must already hold
/// K/V for every position `< pos0 + m_block`. Returns the [m_block *
/// n_heads*head_dim] output block (row-major, out_stride = n_heads*head_dim).
#[allow(clippy::too_many_arguments)]
fn attn_kernel_scalar_qblock(
    q_block: &[f32], // m_block rows of n_heads*head_dim, contiguous
    kv: &mut [f32],
    seq_len: usize,
    pos0: usize,
    m_block: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let stride = n_heads * head_dim;
    let s = pos0 + m_block;
    let mut out = vec![f32::NAN; m_block * stride];
    let mut scores = vec![0f32; m_block * s];
    // SAFETY: buffers sized to the documented contract; every row's position
    // pos0+r < seq_len, and kv holds those positions.
    unsafe {
        inferno_kernels::inferno_attention_f32_scalar_qblock(
            out.as_mut_ptr(),
            q_block.as_ptr(),
            kv.as_mut_ptr(),
            scores.as_mut_ptr(),
            0,
            seq_len * kv_dim,
            pos0,
            m_block,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            stride,
            stride,
        );
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]
    #[test]
    fn attention_qblock_scalar_matches_per_token(
        seed in any::<u64>(),
        pos0 in 0usize..6,
        m_block in prop::sample::select(vec![1usize, 2, 3, 7, 8, 9]),
        hd in prop::sample::select(vec![8usize, 16, 64]),
    ) {
        let (n_heads, n_kv_heads, head_dim) = (4usize, 2usize, hd);
        let kv_dim = n_kv_heads * head_dim;
        let seq_len = 32usize;
        prop_assume!(pos0 + m_block <= seq_len);
        let stride = n_heads * head_dim;
        // A distinct q row per block token.
        let mut q_block = vec![0f32; m_block * stride];
        for r in 0..m_block {
            let row = pseudo(seed ^ (0x100 + r as u64), stride);
            q_block[r * stride..(r + 1) * stride].copy_from_slice(&row);
        }
        // KV cache holding every position < pos0+m_block. K region then V.
        let base_kv = pseudo(seed, 2 * seq_len * kv_dim);

        // Reference: per-token kernel, row by row, each on its own KV copy
        // pre-appended for its own position (the k/v for position pos0+r).
        let mut want = vec![0f32; m_block * stride];
        for r in 0..m_block {
            let pos = pos0 + r;
            let k = pseudo(seed ^ (0x200 + r as u64), kv_dim);
            let v = pseudo(seed ^ (0x300 + r as u64), kv_dim);
            let mut kv_pt = base_kv.clone();
            let row = attn_kernel_scalar(
                &q_block[r * stride..(r + 1) * stride], &k, &v, &mut kv_pt,
                seq_len, pos, kv_dim, n_heads, n_kv_heads, head_dim,
            );
            want[r * stride..(r + 1) * stride].copy_from_slice(&row);
        }
        // Block: append all rows' k/v into one cache, then one block call.
        let mut kv_blk = base_kv;
        let vreg = seq_len * kv_dim;
        for r in 0..m_block {
            let pos = pos0 + r;
            let k = pseudo(seed ^ (0x200 + r as u64), kv_dim);
            let v = pseudo(seed ^ (0x300 + r as u64), kv_dim);
            kv_blk[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
            kv_blk[vreg + pos * kv_dim..vreg + (pos + 1) * kv_dim].copy_from_slice(&v);
        }
        let got = attn_kernel_scalar_qblock(
            &q_block, &mut kv_blk, seq_len, pos0, m_block, kv_dim, n_heads, n_kv_heads, head_dim,
        );
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            prop_assert_eq!(g.to_bits(), w.to_bits(), "elem {}: block {} per-token {}", i, g, w);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p inferno-kernels --test rig attention_qblock_scalar_matches_per_token 2>&1 | tail -20`
Expected: FAIL to compile — `inferno_attention_f32_scalar_qblock` is not defined.

- [ ] **Step 3: Write the kernel**

Add to `crates/inferno-kernels/src/attention.rs` immediately after `attn_core_scalar` (before the `dot8` definition):

```rust
/// Query-blocked scalar attention (M4b.14). Computes query rows `[0,
/// m_block)` — row `r` at position `pos0 + r` — reusing each visible K and
/// V vector across the block's rows (streamed once per head per block
/// instead of once per token). Blocking only reorders the query axis, so
/// each row's arithmetic is bit-for-bit the per-token kernel's: same
/// `dot8`/`reduce8` order, same block-of-8 `expf_scalar` softmax + scalar
/// tail, same ascending-`t` `mul_add` V-accumulation.
///
/// # Safety
/// - `out`/`q` valid for `(m_block-1)*{out,q}_stride + n_heads*head_dim` f32.
/// - `scores` valid for `m_block * (pos0 + m_block)` f32 (scratch).
/// - `kv` valid for the K/V regions, holding every position `< pos0+m_block`.
/// - `pos0 + m_block <= seq_len`; `m_block >= 1`; `head_dim` a multiple of 8.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_attention_f32_scalar_qblock(
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    scores: *mut f32,
    kv_base: usize,
    v_off: usize,
    pos0: usize,
    m_block: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_stride: usize,
    out_stride: usize,
) {
    // SAFETY: contract above; delegate to a safe-slice core.
    unsafe {
        let s = pos0 + m_block;
        let q_extent = (m_block - 1) * q_stride + n_heads * head_dim;
        let out_extent = (m_block - 1) * out_stride + n_heads * head_dim;
        let q = std::slice::from_raw_parts(q, q_extent);
        let out = std::slice::from_raw_parts_mut(out, out_extent);
        let scores = std::slice::from_raw_parts_mut(scores, m_block * s);
        let kv = std::slice::from_raw_parts(kv, kv_base + 2 * v_off);
        attn_core_scalar_qblock(
            out, q, kv, scores, kv_base, v_off, pos0, m_block, kv_dim, n_heads,
            n_kv_heads, head_dim, q_stride, out_stride,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn attn_core_scalar_qblock(
    out: &mut [f32],
    q: &[f32],
    kv: &[f32],
    scores: &mut [f32],
    kv_base: usize,
    v_off: usize,
    pos0: usize,
    m_block: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_stride: usize,
    out_stride: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let group = n_heads / n_kv_heads;
    let kreg = kv_base;
    let vreg = kv_base + v_off;
    let s = pos0 + m_block; // per-row scores stride = max visible over the block
    for h in 0..n_heads {
        let g = h / group;
        // scores pass: each visible K vector loaded once, reused across rows.
        for t in 0..s {
            let kb = kreg + t * kv_dim + g * head_dim;
            let kt = &kv[kb..kb + head_dim];
            for r in 0..m_block {
                if t <= pos0 + r {
                    let qh = &q[r * q_stride + h * head_dim..][..head_dim];
                    // Same 8-lane partition order as the per-token kernel.
                    scores[r * s + t] = dot8(qh, kt) * scale;
                }
            }
        }
        // softmax + in-place normalize per row (denom is a scalar). Mirrors
        // the per-token loop EXACTLY: block-of-8 reduce8 denom + scalar tail.
        for r in 0..m_block {
            let visible = pos0 + r + 1;
            let row = &mut scores[r * s..r * s + s];
            let max = row[..visible]
                .iter()
                .fold(f32::NEG_INFINITY, |m, v| m.max(*v));
            let mut denom = 0f32;
            let mut t = 0;
            while t + 8 <= visible {
                let mut lanes = [0f32; 8];
                for (l, lane) in lanes.iter_mut().enumerate() {
                    let e = expf_scalar(row[t + l] - max);
                    row[t + l] = e;
                    *lane = e;
                }
                denom += reduce8(lanes);
                t += 8;
            }
            while t < visible {
                let e = expf_scalar(row[t] - max);
                row[t] = e;
                denom += e;
                t += 1;
            }
            // Normalize now; w/denom is the same value the per-token kernel
            // computes lazily in its output loop.
            for w in row[..visible].iter_mut() {
                *w /= denom;
            }
        }
        // output pass: each visible V vector loaded once, reused across rows.
        // Zero every row's head-span first, then accumulate in ascending t.
        for r in 0..m_block {
            let ob = r * out_stride + h * head_dim;
            out[ob..ob + head_dim].fill(0.0);
        }
        for t in 0..s {
            let vb = vreg + t * kv_dim + g * head_dim;
            for r in 0..m_block {
                if t <= pos0 + r {
                    let wn = scores[r * s + t];
                    let ob = r * out_stride + h * head_dim;
                    for d in 0..head_dim {
                        out[ob + d] = wn.mul_add(kv[vb + d], out[ob + d]);
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p inferno-kernels --test rig attention_qblock_scalar_matches_per_token 2>&1 | tail -20`
Expected: PASS (96 proptest cases).

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-kernels/src/attention.rs crates/inferno-kernels/tests/rig.rs
git commit -m "M4b.14: scalar query-blocked attention kernel (bit-identical to per-token)"
```

---

## Task 2: AVX2 query-blocked attention kernel

**Files:**
- Modify: `crates/inferno-kernels/src/attention.rs` (add `attn_core_avx2_qblock` + `inferno_attention_f32_avx2_qblock` after `attn_core_avx2`)
- Test: `crates/inferno-kernels/tests/rig.rs` (add `attn_kernel_avx2_qblock` driver + `attention_qblock_isa_bitwise_equal` proptest)

**Interfaces:**
- Consumes: existing `hsum8`, `expf::expf_avx2`, `expf::expf_scalar`, and the Task 1 `inferno_attention_f32_scalar_qblock` (as the bit-equality reference).
- Produces: `pub unsafe extern "C" fn inferno_attention_f32_avx2_qblock(...)` — same 14-argument signature as the scalar block kernel. Bitwise-identical to the scalar block kernel.

- [ ] **Step 1: Write the failing test**

Add to `crates/inferno-kernels/tests/rig.rs` after `attn_kernel_scalar_qblock`:

```rust
/// Drive the query-blocked AVX2 kernel; same contract as
/// `attn_kernel_scalar_qblock`.
#[allow(clippy::too_many_arguments)]
fn attn_kernel_avx2_qblock(
    q_block: &[f32],
    kv: &mut [f32],
    seq_len: usize,
    pos0: usize,
    m_block: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let stride = n_heads * head_dim;
    let s = pos0 + m_block;
    let mut out = vec![f32::NAN; m_block * stride];
    let mut scores = vec![0f32; m_block * s];
    // SAFETY: same contract as the scalar block driver; avx2 checked by caller.
    unsafe {
        inferno_kernels::inferno_attention_f32_avx2_qblock(
            out.as_mut_ptr(),
            q_block.as_ptr(),
            kv.as_mut_ptr(),
            scores.as_mut_ptr(),
            0,
            seq_len * kv_dim,
            pos0,
            m_block,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            stride,
            stride,
        );
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn attention_qblock_isa_bitwise_equal(
        seed in any::<u64>(),
        pos0 in 0usize..6,
        m_block in prop::sample::select(vec![1usize, 2, 3, 7, 8, 9]),
        hd in prop::sample::select(vec![8usize, 16, 64]),
    ) {
        if !std::is_x86_feature_detected!("avx2") { return Ok(()); }
        let (n_heads, n_kv_heads, head_dim) = (4usize, 2usize, hd);
        let kv_dim = n_kv_heads * head_dim;
        let seq_len = 32usize;
        prop_assume!(pos0 + m_block <= seq_len);
        let stride = n_heads * head_dim;
        let mut q_block = vec![0f32; m_block * stride];
        for r in 0..m_block {
            let row = pseudo(seed ^ (0x100 + r as u64), stride);
            q_block[r * stride..(r + 1) * stride].copy_from_slice(&row);
        }
        let base_kv = pseudo(seed, 2 * seq_len * kv_dim);
        let vreg = seq_len * kv_dim;
        let mut kv = base_kv;
        for r in 0..m_block {
            let pos = pos0 + r;
            let k = pseudo(seed ^ (0x200 + r as u64), kv_dim);
            let v = pseudo(seed ^ (0x300 + r as u64), kv_dim);
            kv[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&k);
            kv[vreg + pos * kv_dim..vreg + (pos + 1) * kv_dim].copy_from_slice(&v);
        }
        let mut kv_s = kv.clone();
        let a = attn_kernel_scalar_qblock(&q_block, &mut kv_s, seq_len, pos0, m_block, kv_dim, n_heads, n_kv_heads, head_dim);
        let b = attn_kernel_avx2_qblock(&q_block, &mut kv, seq_len, pos0, m_block, kv_dim, n_heads, n_kv_heads, head_dim);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            prop_assert_eq!(x.to_bits(), y.to_bits(), "elem {}: scalar {} avx2 {}", i, x, y);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p inferno-kernels --test rig attention_qblock_isa_bitwise_equal 2>&1 | tail -20`
Expected: FAIL to compile — `inferno_attention_f32_avx2_qblock` not defined.

- [ ] **Step 3: Write the kernel**

Add to `crates/inferno-kernels/src/attention.rs` after `attn_core_avx2` (before `hsum8`):

```rust
/// Query-blocked AVX2 attention (M4b.14). The block structure of
/// [`inferno_attention_f32_scalar_qblock`] with the AVX2 inner loops of
/// [`attn_core_avx2`] (fmadd dot + `hsum8`, `expf_avx2` softmax blocks +
/// `expf_scalar` tail, broadcast-`wn` fmadd V-accumulation). Bit-identical
/// to the scalar block kernel and, by transitivity, to the per-token kernel.
///
/// # Safety
/// As [`inferno_attention_f32_scalar_qblock`], plus the running CPU has
/// AVX2+FMA.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferno_attention_f32_avx2_qblock(
    out: *mut f32,
    q: *const f32,
    kv: *mut f32,
    scores: *mut f32,
    kv_base: usize,
    v_off: usize,
    pos0: usize,
    m_block: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_stride: usize,
    out_stride: usize,
) {
    // SAFETY: contract as the scalar block kernel; head_dim a multiple of 8.
    unsafe {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group = n_heads / n_kv_heads;
        let kreg = kv_base;
        let vreg = kv_base + v_off;
        let s = pos0 + m_block;
        for h in 0..n_heads {
            let g = h / group;
            // scores pass: load each K vector once, reuse across rows.
            for t in 0..s {
                let kb = kv.add(kreg + t * kv_dim + g * head_dim);
                for r in 0..m_block {
                    if t <= pos0 + r {
                        let qh = q.add(r * q_stride + h * head_dim);
                        let mut acc = _mm256_setzero_ps();
                        let mut d = 0;
                        while d < head_dim {
                            let qv = _mm256_loadu_ps(qh.add(d));
                            let kvv = _mm256_loadu_ps(kb.add(d));
                            acc = _mm256_fmadd_ps(qv, kvv, acc);
                            d += 8;
                        }
                        *scores.add(r * s + t) = hsum8(acc) * scale;
                    }
                }
            }
            // softmax + in-place normalize per row (scalar denom).
            for r in 0..m_block {
                let visible = pos0 + r + 1;
                let base = scores.add(r * s);
                let mut max = f32::NEG_INFINITY;
                for t in 0..visible {
                    max = max.max(*base.add(t));
                }
                let maxv = _mm256_set1_ps(max);
                let mut denom = 0f32;
                let mut t = 0;
                while t + 8 <= visible {
                    let sc = _mm256_loadu_ps(base.add(t));
                    let e = crate::expf::expf_avx2(_mm256_sub_ps(sc, maxv));
                    _mm256_storeu_ps(base.add(t), e);
                    denom += hsum8(e);
                    t += 8;
                }
                while t < visible {
                    let e = crate::expf::expf_scalar(*base.add(t) - max);
                    *base.add(t) = e;
                    denom += e;
                    t += 1;
                }
                for t in 0..visible {
                    *base.add(t) /= denom;
                }
            }
            // output pass: load each V vector once, reuse across rows.
            for r in 0..m_block {
                let oh = out.add(r * out_stride + h * head_dim);
                for d in (0..head_dim).step_by(8) {
                    _mm256_storeu_ps(oh.add(d), _mm256_setzero_ps());
                }
            }
            for t in 0..s {
                let vb = kv.add(vreg + t * kv_dim + g * head_dim);
                for r in 0..m_block {
                    if t <= pos0 + r {
                        let wn = _mm256_set1_ps(*scores.add(r * s + t));
                        let oh = out.add(r * out_stride + h * head_dim);
                        for d in (0..head_dim).step_by(8) {
                            let cur = _mm256_loadu_ps(oh.add(d));
                            let vv = _mm256_loadu_ps(vb.add(d));
                            _mm256_storeu_ps(oh.add(d), _mm256_fmadd_ps(wn, vv, cur));
                        }
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p inferno-kernels --test rig attention_qblock 2>&1 | tail -20`
Expected: PASS — both `attention_qblock_scalar_matches_per_token` and `attention_qblock_isa_bitwise_equal`.

- [ ] **Step 5: Run the whole kernel rig + clippy to confirm no regression**

Run: `cargo test -p inferno-kernels 2>&1 | tail -15 && cargo clippy -p inferno-kernels --all-targets 2>&1 | tail -5`
Expected: all tests PASS; clippy clean (no warnings).

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-kernels/src/attention.rs crates/inferno-kernels/tests/rig.rs
git commit -m "M4b.14: AVX2 query-blocked attention kernel (bit-identical to scalar block)"
```

---

## Task 3: Pool block dispatch

**Files:**
- Modify: `crates/inferno-pool/src/pool.rs` (define `AttnBlockFn`; `AttnJob.kernel` type; rewrite `run_attn_span`; update `stamp_attn` fake kernel + `attn_dispatch` in tests)
- Modify: `crates/inferno-pool/src/lib.rs` (re-export `AttnBlockFn`; `inferno_par_attention` kernel param type)

**Interfaces:**
- Consumes: Task 1/2 `inferno_attention_f32_{scalar,avx2}_qblock` (by pointer, passed from generated code — the pool never selects the kernel itself).
- Produces:
  - `pub type AttnBlockFn = unsafe extern "C" fn(*mut f32, *const f32, *mut f32, *mut f32, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize)` in `crates/inferno-pool/src/pool.rs` (14 args, matching the block kernel), re-exported from `lib.rs`.
  - `AttnJob.kernel: AttnBlockFn` (changed from `AttnFn`).
  - `run_attn_span(j: &AttnJob, start: usize, end: usize)` — one block-kernel call for the whole shard `[start, end)`.
  - `inferno_par_attention`'s first parameter becomes `kernel: AttnBlockFn`.

- [ ] **Step 1: Define `AttnBlockFn` in the pool and re-export it**

In `crates/inferno-pool/src/pool.rs`, after the `AttnHspanFn` type alias (ends near line 64, just before `pub struct AttnJob`), add:

```rust
/// The M4b.14 query-blocked attention kernel ABI: the [`AttnFn`] region args
/// plus `pos0`/`m_block` (a shard of query tokens) and the `q`/`out` row
/// strides. Matches `inferno-kernels`' `inferno_attention_f32_*_qblock`
/// symbols exactly; passed by pointer (from generated code) to
/// `inferno_par_attention`, which calls it once per lane shard.
pub type AttnBlockFn = unsafe extern "C" fn(
    *mut f32,   // out (shard's first row)
    *const f32, // q (shard's first row)
    *mut f32,   // kv
    *mut f32,   // scores scratch (>= m_block * (pos0 + m_block))
    usize,      // kv_base
    usize,      // v_off
    usize,      // pos0 (position of the shard's first row)
    usize,      // m_block (rows in this shard)
    usize,      // kv_dim
    usize,      // n_heads
    usize,      // n_kv_heads
    usize,      // head_dim
    usize,      // q_stride
    usize,      // out_stride
);
```

In `crates/inferno-pool/src/lib.rs`, add `AttnBlockFn` to the existing `pub use pool::{...}` (line 16):
```rust
pub use pool::{AttnBlockFn, AttnFn, AttnHeadsJob, AttnHspanFn, AttnJob, GemmFn, GemvFn, Pool, TokenBodyFn};
```

- [ ] **Step 2: Change the pool `AttnJob` + `run_attn_span` (this breaks the pool tests — expected)**

In `crates/inferno-pool/src/pool.rs`:

Change the `AttnJob.kernel` field type. Find (near line 70):
```rust
pub struct AttnJob {
    pub kernel: AttnFn,
```
Change to:
```rust
pub struct AttnJob {
    pub kernel: AttnBlockFn,
```
`AttnBlockFn` is defined in the same module (Step 1), so no import change is needed.

Replace `run_attn_span` (lines 178-199) with a single block call per shard:
```rust
pub(crate) unsafe fn run_attn_span(j: &AttnJob, start: usize, end: usize) {
    let m_block = end - start;
    if m_block == 0 {
        return;
    }
    // Scratch: m_block rows, each row's stride is the block's max visible
    // (pos0 + start) + m_block = j.pos0 + end.
    let s = j.pos0 + end;
    let mut scores = vec![0f32; m_block * s];
    // SAFETY: forwarding the caller's contract for the shard's tokens. `out`
    // and `q` are advanced to the shard's first row; the kernel walks rows by
    // out_stride/q_stride. scores is sized m_block * s per the block ABI.
    unsafe {
        (j.kernel)(
            j.out.add(start * j.out_stride),
            j.q.add(start * j.q_stride),
            j.kv,
            scores.as_mut_ptr(),
            j.kv_base,
            j.v_off,
            j.pos0 + start,
            m_block,
            j.kv_dim,
            j.n_heads,
            j.n_kv_heads,
            j.head_dim,
            j.q_stride,
            j.out_stride,
        );
    }
}
```

- [ ] **Step 3: Change `inferno_par_attention`'s kernel parameter type**

In `crates/inferno-pool/src/lib.rs`, change the `inferno_par_attention` signature (line 237):
```rust
pub unsafe extern "C" fn inferno_par_attention(
    kernel: AttnFn,
```
to:
```rust
pub unsafe extern "C" fn inferno_par_attention(
    kernel: AttnBlockFn,
```
`inferno_par_attention` refers to `AttnFn` via the module path used at line 237 — change it to `AttnBlockFn`; confirm the symbol resolves (both live in `pool`, imported the same way — `grep -n "AttnFn\|AttnBlockFn\|use crate::pool" crates/inferno-pool/src/lib.rs`). The `m == 1` path (lines 270-274) calls `run_attn_span(&job, 0, 1)` and needs no change — the block kernel with `m_block=1` is the per-token equivalent. The rest of the body (the `AttnJob { kernel, ... }` construction and the dispatch) is unchanged.

- [ ] **Step 4: Update the pool test fake kernel to the block ABI**

In `crates/inferno-pool/src/pool.rs` tests, replace `stamp_attn` (lines 1079-1098) with a block-ABI fake that stamps each row exactly as the per-token version did, so `attn_expected` (unchanged) still holds:

```rust
/// Fake query-blocked attention kernel with the real block ABI:
/// deterministic function of (q row, pos), writes each row's whole out
/// span, and touches `scores[r*s + pos]` to prove the scratch covers the
/// block's per-row visible range.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn stamp_attn(
    out: *mut f32,
    q: *const f32,
    _kv: *mut f32,
    scores: *mut f32,
    _kv_base: usize,
    _v_off: usize,
    pos0: usize,
    m_block: usize,
    _kv_dim: usize,
    n_heads: usize,
    _n_kv_heads: usize,
    head_dim: usize,
    q_stride: usize,
    out_stride: usize,
) {
    let hd = n_heads * head_dim;
    let s = pos0 + m_block;
    for r in 0..m_block {
        let pos = pos0 + r;
        // SAFETY: scratch is m_block * s; index r*s + pos with pos < s.
        unsafe { *scores.add(r * s + pos) = pos as f32 };
        for i in 0..hd {
            // SAFETY: out/q rows are out_stride/q_stride apart, hd wide.
            unsafe {
                *out.add(r * out_stride + i) = *q.add(r * q_stride + i) + (pos * 31 + i) as f32;
            }
        }
    }
}
```

The `attn_dispatch` helper (line 1104) builds `AttnJob { kernel: stamp_attn, ... }`; with `stamp_attn` now the block ABI and `AttnJob.kernel: AttnBlockFn`, it compiles unchanged. `attn_expected` (line 1128) is unchanged — the per-row stamp formula is identical.

- [ ] **Step 5: Run the pool tests to verify they pass**

Run: `cargo test -p inferno-pool 2>&1 | tail -20`
Expected: PASS — `attention_parallel_matches_serial_expectation`, `attention_threads_exceeding_tokens_collapses`, `attention_capacity_one_runs_inline`, `attention_ignores_decode_cap`, `attention_zero_tokens_is_a_noop`, and the `par_attention_fallback` / `par_rig` suites.

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-pool/src/pool.rs crates/inferno-pool/src/lib.rs
git commit -m "M4b.14: pool dispatches prefill attention as query blocks (one call per shard)"
```

---

## Task 4: Codegen emits the block kernel + host-ABI bump

**Files:**
- Modify: `crates/inferno-codegen/src/loopir.rs` (`attention_symbol` → `_qblock`)
- Modify: `crates/inferno-codegen/src/llvm/mod.rs` (declare the `_qblock` extern; 14-param type)
- Modify: `crates/inferno-codegen/src/lib.rs` (`HOST_ABI_VERSION` "7" → "8")
- Modify: `crates/inferno-core/src/artifact.rs` (register the two `_qblock` symbols)
- Modify: `crates/inferno-codegen/src/llvm/snapshots/` (regenerated insta snapshots — reviewed)

**Interfaces:**
- Consumes: Task 1/2 kernel symbols; Task 3 `inferno_par_attention` (block ABI). `inferno_par_attention`'s *argument list* from generated code is unchanged (it already passes `q_stride`/`out_stride`); only the kernel symbol string it loads changes.

- [ ] **Step 1: Point `attention_symbol` at the block kernel**

In `crates/inferno-codegen/src/loopir.rs`, change `attention_symbol` (lines 117-125):
```rust
/// `inferno_attention_f32_{isa}_qblock`: the query-blocked f32 attention
/// kernel (M4b.14). Passed by pointer to `inferno_par_attention`, which now
/// calls it once per lane shard. Selected by the same `KernelIsa` codegen
/// uses for gemv/gemm.
pub fn attention_symbol(isa: inferno_kernels::KernelIsa) -> String {
    let isa = match isa {
        inferno_kernels::KernelIsa::Scalar => "scalar",
        inferno_kernels::KernelIsa::Avx2 => "avx2",
    };
    format!("inferno_attention_f32_{isa}_qblock")
}
```
`attention_hspan_symbol` (line 130) is unchanged — it calls a helper string; verify it still builds `..._hspan` off the base name, NOT off `attention_symbol` (it currently does `format!("{}_hspan", attention_symbol(isa))`, which would now wrongly produce `..._qblock_hspan`). **Fix `attention_hspan_symbol` to not depend on `attention_symbol`:**
```rust
pub fn attention_hspan_symbol(isa: inferno_kernels::KernelIsa) -> String {
    let isa = match isa {
        inferno_kernels::KernelIsa::Scalar => "scalar",
        inferno_kernels::KernelIsa::Avx2 => "avx2",
    };
    format!("inferno_attention_f32_{isa}_hspan")
}
```

- [ ] **Step 2: Declare the block-kernel extern in the module**

In `crates/inferno-codegen/src/llvm/mod.rs`, after the per-token attention declaration loop (lines 120-126), add a declaration for the 14-param block kernel (the per-token `attn_ty` has 11 params; the block kernel adds `m_block` and the two strides → 14):

```rust
        // void inferno_attention_f32_<isa>_qblock(ptr out, ptr q, ptr kv,
        //   ptr scores, i64 kv_base, i64 v_off, i64 pos0, i64 m_block,
        //   i64 kv_dim, i64 n_heads, i64 n_kv_heads, i64 head_dim,
        //   i64 q_stride, i64 out_stride)
        // — the M4b.14 query-blocked prefill kernel; passed as a function
        // pointer to inferno_par_attention, never called directly.
        let attn_qblock_ty = void.fn_type(
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
        for isa in ["scalar", "avx2"] {
            self.module.add_function(
                &format!("inferno_attention_f32_{isa}_qblock"),
                attn_qblock_ty,
                Some(Linkage::External),
            );
        }
```

The per-token `inferno_attention_f32_{isa}` declarations (lines 120-126) may now be unused by generated code. Keep them declared — the extern declaration is harmless and the symbols still exist (rig + the block kernel's reference); removing them is out of scope. `inferno_par_attention`'s own declared type (`par_attn_ty`, lines 203-226) is **unchanged**.

- [ ] **Step 3: Bump the host-ABI version**

In `crates/inferno-codegen/src/lib.rs`, change `HOST_ABI_VERSION` (line 21) and add a history line (lines 12-21 comment block):
```rust
/// "8" = M4b.14's query-blocked prefill attention kernel
/// (`inferno_attention_f32_*_qblock`; `inferno_par_attention` now invokes
/// its kernel argument with the block ABI — a stale artifact would pass the
/// old per-token pointer to the block-calling dispatcher, so the bump is
/// mandatory to force recompile);
/// "7" = M4b.11's head-sharded decode attention
```
```rust
pub const HOST_ABI_VERSION: &str = "8";
```

- [ ] **Step 4: Register the block symbols for dlopen resolution**

In `crates/inferno-core/src/artifact.rs`, after the per-token attention registrations (lines 547-550), add:
```rust
    p(inferno_kernels::inferno_attention_f32_scalar_qblock as *const ());
    p(inferno_kernels::inferno_attention_f32_avx2_qblock as *const ());
```

- [ ] **Step 5: Build and update the codegen snapshots**

Run: `cargo build -p inferno-codegen -p inferno-core 2>&1 | tail -5`
Expected: builds clean.

Run: `cargo test -p inferno-codegen 2>&1 | tail -20`
Expected: snapshot tests that assert on emitted IR (e.g. the `mod.rs` test at line 440 asserting `ir.contains("inferno_par_attention")` still passes; any snapshot capturing the attention symbol string now shows `_qblock` and FAILS as a pending snapshot).

Run: `cargo insta review`
Review each changed snapshot: the only diffs should be `inferno_attention_f32_{scalar,avx2}` → `..._qblock` in the attention lowering IR, plus the added extern declaration. **Do not blind-accept** — confirm every diff is exactly that symbol rename / declaration, nothing else. Accept the reviewed snapshots.

If `mod.rs`'s inline assertion at line 440-441 needs updating (it checks `inferno_par_attention` / `inferno_par_attention_heads`, both unchanged names — so it should still pass), leave it. If a test asserts the exact per-token symbol name, update it to `_qblock`.

- [ ] **Step 6: Commit**

```bash
git add crates/inferno-codegen/src/loopir.rs crates/inferno-codegen/src/llvm/mod.rs crates/inferno-codegen/src/lib.rs crates/inferno-core/src/artifact.rs crates/inferno-codegen/src/llvm/snapshots/
git commit -m "M4b.14: codegen emits query-blocked attention kernel; bump HOST_ABI_VERSION to 8"
```

---

## Task 5: End-to-end differential verification (no tolerance change)

**Files:**
- No source changes expected. If a differential fails, that is a Lever-1 bug (the kernel is not actually bit-identical) — fix the kernel, never the tolerance.

**Interfaces:**
- Consumes: everything from Tasks 1-4.

- [ ] **Step 1: Run the codegen compiled-vs-interpreter differential**

Run: `cargo test -p inferno-codegen --test differential 2>&1 | tail -20`
Expected: PASS. This is the compiled-vs-interpreter correctness gate; it exercises the block kernel through the compiled path. It must pass with `attn_rel_tol` unchanged.

- [ ] **Step 2: Run the artifact-level differential**

Run: `cargo test -p inferno-core --test artifact 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 3: Confirm `attn_rel_tol` is untouched**

Run: `git diff --stat crates/inferno-graph/src/tolerance.rs`
Expected: empty output (no changes to `tolerance.rs`). If this file was modified, revert it and re-diagnose the kernel — a bit-identical kernel needs no tolerance change.

- [ ] **Step 4: Run the full test + differential + lint suite**

Run: `mise run test 2>&1 | tail -25`
Expected: all green.

Run: `mise run differential 2>&1 | tail -15`
Expected: green (the gap distributions are unchanged — the kernel is bit-identical).

Run: `mise run lint 2>&1 | tail -10`
Expected: clippy clean (`-D warnings`), fmt clean.

- [ ] **Step 5: Commit (only if any test-file touch-ups were needed; otherwise skip)**

```bash
git add -A
git commit -m "M4b.14: differential + full suite green with query blocking, attn_rel_tol unchanged"
```

If no files changed in this task, there is nothing to commit — the verification stands on Task 4's commit.

---

## Task 6: Sub-bracket attention instrument + gate script

**Files:**
- Modify: `crates/inferno-codegen/src/llvm/ops.rs` (`lower_tile_attention`) and `crates/inferno-codegen/src/profile.rs` — emit three profiler brackets (`attn:scores`, `attn:softmax`, `attn:output`) around the attention kernel **only under a cargo feature**, off in shipping/bench builds.
- Create: `scripts/quiet-hw/gate-prefill-attn-split.sh`

**Interfaces:**
- Consumes: the existing profiler slot mechanism (`profile.rs::step_label`, the `--profile` op table). The instrument is the prefill-side analogue of M4b.12's `pool-profile` feature.

**Note on approach:** the current attention kernel is one opaque host call, so the *host-side* profiler can only time the whole call. To split `scores`/`softmax`/`output`, the instrument adds three `rdtsc`-bracketed regions **inside the block kernel** (compiled in only under the feature), reported through a thread-local accumulator the gate script reads — exactly M4b.12's `pool-profile` pattern. Keep it entirely out of the shipping and bench builds.

- [ ] **Step 1: Add the cargo feature**

In `crates/inferno-kernels/Cargo.toml`, add under `[features]`:
```toml
# M4b.14: split the query-blocked attention kernel into scores/softmax/output
# rdtsc sub-brackets for the mid-milestone gate. OFF in shipping and bench
# builds — only scripts/quiet-hw/gate-prefill-attn-split.sh enables it.
attn-subprofile = []
```

- [ ] **Step 2: Add the thread-local sub-bracket accumulator (feature-gated)**

In `crates/inferno-kernels/src/attention.rs`, add near the top (feature-gated so it vanishes in normal builds):
```rust
#[cfg(feature = "attn-subprofile")]
pub mod subprofile {
    use std::cell::Cell;
    thread_local! {
        pub static SCORES: Cell<u64> = const { Cell::new(0) };
        pub static SOFTMAX: Cell<u64> = const { Cell::new(0) };
        pub static OUTPUT: Cell<u64> = const { Cell::new(0) };
    }
    #[inline]
    pub fn rdtsc() -> u64 {
        // SAFETY: _rdtsc is always available on x86_64.
        #[cfg(target_arch = "x86_64")]
        unsafe { std::arch::x86_64::_rdtsc() }
        #[cfg(not(target_arch = "x86_64"))]
        0
    }
    /// Print and reset the accumulated cycle counts (called by the CLI after
    /// a profiled prefill run when the feature is on).
    pub fn drain() -> (u64, u64, u64) {
        (
            SCORES.with(|c| c.replace(0)),
            SOFTMAX.with(|c| c.replace(0)),
            OUTPUT.with(|c| c.replace(0)),
        )
    }
}
```

Bracket the three passes in **both** `attn_core_scalar_qblock` and `inferno_attention_f32_avx2_qblock` with `#[cfg(feature = "attn-subprofile")]` timing (start `rdtsc` before the scores pass, accumulate into `SCORES` after it; same for softmax and output). Example wrapper around the scores pass (repeat pattern for softmax/output):
```rust
#[cfg(feature = "attn-subprofile")]
let _t0 = crate::attention::subprofile::rdtsc();
// ... scores pass ...
#[cfg(feature = "attn-subprofile")]
crate::attention::subprofile::SCORES.with(|c| c.set(c.get() + crate::attention::subprofile::rdtsc() - _t0));
```
(Inside `attention.rs` the path is `subprofile::` not `crate::attention::subprofile::`.) The timing lines compile to nothing without the feature, so the shipping/bench kernel is byte-identical to Task 1/2.

- [ ] **Step 3: Wire the drain into the CLI `--profile` output (feature-gated)**

Find how `--profile` prints the op table:

Run: `grep -rn "profile\b\|--profile\|profile \[" crates/cli/src/*.rs crates/inferno-runtime/src/*.rs 2>/dev/null | head`

Under `#[cfg(feature = "attn-subprofile")]`, after the existing profile table prints, call `inferno_kernels::attention::subprofile::drain()` and print three lines:
```
attn:scores   <cycles>
attn:softmax  <cycles>
attn:output   <cycles>
```
formatted to match the existing op-table rows so the gate script can `grep` them. Propagate the `attn-subprofile` feature from the CLI crate down to `inferno-kernels` (add a matching passthrough feature in the CLI's `Cargo.toml`).

- [ ] **Step 4: Verify the feature is off by default and on when requested**

Run: `cargo build -p inferno-kernels 2>&1 | tail -3`
Expected: builds clean; `subprofile` module absent.

Run: `cargo build -p inferno-kernels --features attn-subprofile 2>&1 | tail -3`
Expected: builds clean with the module present.

Run: `cargo test -p inferno-kernels --test rig attention_qblock 2>&1 | tail -5`
Expected: PASS — the bit-identity tests still pass (default build, instrument absent).

- [ ] **Step 5: Write the gate script**

Create `scripts/quiet-hw/gate-prefill-attn-split.sh`, mirroring `gate-prefill-attr.sh` (read that file first for the exact `lib.sh` helpers, `smoke_header`, `machine_block`, `QHW_OUT`/`QHW_SMOKE` env, and the `sed -n '/^profile \[/,$p'` extraction). Differences: build/run with `--features attn-subprofile` and extract the three `attn:*` rows in addition to the op table:

```bash
#!/usr/bin/env bash
# M4b.14 mid-milestone attention sub-bracket gate — the fresh split-bracket
# t=1 prefill profile the pre-registered ladder rule consumes (spec §Mid-
# Milestone Gate). Prints the t=1 prefill op table PLUS the attention kernel's
# scores/softmax/output sub-brackets (via the attn-subprofile feature, OFF in
# every shipping/bench build). The pp ratios come from gate-bench-protocol.sh
# in the same session. VERDICTS ARE HUMAN: paste into the M4b.14 spec
# §Amendments and compute there attn_share = attn_total / prefill_total and
# the ceiling check pp_ratio / (1 - f*c) >= 1.0 per the spec's pre-registered
# rule (docs/runbooks/quiet-hw-verification.md).
# Usage: gate-prefill-attn-split.sh <model.gguf>   (env: QHW_OUT QHW_SMOKE)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
command -v cargo >/dev/null || { echo "missing cargo (devenv shell)" >&2; exit 2; }

MODEL="${1:?usage: gate-prefill-attn-split.sh <model.gguf>}"
OUT="${QHW_OUT:-$(mktemp -d)}"
if [ "${QHW_SMOKE:-0}" = 1 ]; then PROMPT_BYTES=256; MAXTOK=8; else PROMPT_BYTES=2048; MAXTOK=32; fi

smoke_header "gate-prefill-attn-split (M4b.14: attn scores/softmax/output sub-brackets)"
machine_block
echo

PROMPT="$(head -c "$PROMPT_BYTES" /dev/urandom | base64 | tr -d '\n')"

echo "--- t=1 prefill profile + attn sub-brackets ---"
cargo run --release -q --features attn-subprofile -p inferno -- run "$MODEL" \
  --prompt "$PROMPT" --max-tokens "$MAXTOK" --threads 1 --profile \
  > "$OUT/prefill-attn-split-t1.txt" 2>&1
sed -n '/^profile \[/,$p' "$OUT/prefill-attn-split-t1.txt"
echo
echo "--- attn sub-brackets (grep) ---"
grep -E '^attn:(scores|softmax|output)' "$OUT/prefill-attn-split-t1.txt" || true
```

Confirm the `--features attn-subprofile` flag reaches the `inferno` binary crate (the CLI Cargo passthrough from Step 3). Make it executable:
```bash
chmod +x scripts/quiet-hw/gate-prefill-attn-split.sh
```

- [ ] **Step 6: Smoke the gate script locally (non-quiet — just proves it runs and emits the rows)**

Run (devenv shell, if the model is present):
`QHW_SMOKE=1 scripts/quiet-hw/gate-prefill-attn-split.sh /home/dev/.cache/inferno-tests/qwen2.5-0.5b-instruct-q8_0.gguf 2>&1 | tail -25`
Expected: prints the op table and three `attn:scores/softmax/output` rows with non-zero cycle counts. (If the binary can't load outside devenv — `libffi.so.8` — run inside the devenv shell.)

- [ ] **Step 7: Commit**

```bash
git add crates/inferno-kernels/Cargo.toml crates/inferno-kernels/src/attention.rs crates/cli scripts/quiet-hw/gate-prefill-attn-split.sh
git commit -m "M4b.14: feature-gated attn scores/softmax/output sub-bracket instrument + gate script"
```

---

## Task 7: Local dev gate — evidence to unblock metal spend

**Files:**
- No source changes. This is a measurement task (M4b.13 Task-3 precedent): a non-quiet local micro-benchmark + t=1 pp delta, honestly labeled non-quiet, run before any PhoenixNAP provision.

**Interfaces:**
- Consumes: the built release binary with query blocking (Tasks 1-5).

- [ ] **Step 1: Micro-benchmark the block kernel vs per-token on the blamed shapes**

Add (or extend) a criterion bench in `crates/inferno-kernels/benches/` comparing `inferno_attention_f32_avx2` looped per token vs `inferno_attention_f32_avx2_qblock` over a block, at the criterion model's shape (n_heads=14, n_kv_heads=2, head_dim=64) across prefill-representative visible lengths (e.g. pos0 ∈ {64, 256, 512}, m_block = PREFILL_TILE = 64).

Run: `mise run bench-kernels 2>&1 | tail -30` (devenv shell, quiet-ish local box)
Record the block-vs-per-token geomean speedup on the blamed shapes.

- [ ] **Step 2: Measure t=1 prefill pp delta locally (non-quiet)**

Run, before and after (compare against the pre-M4b.14 binary via `git stash` or a separate checkout), inside the devenv shell:
`cargo run --release -q -p inferno -- run <model> --prompt "$(head -c 2048 /dev/urandom | base64 | tr -d '\n')" --max-tokens 32 --threads 1 --profile 2>&1 | sed -n '/^profile \[/,$p'`
Record the t=1 pp tokens/s before vs after and the attention-bracket share before vs after.

- [ ] **Step 3: Record the local dev data point + decide**

Append to the spec's §Amendments a "Local dev data point (Task 7)" entry: µbench geomean speedup, t=1 pp delta, attention-bracket share drop, **honestly labeled non-quiet**. Local gate PASS (block kernel shows a real per-thread win on the blamed shapes) unblocks the metal spend for Task 8. If the µbench shows no win, STOP here and diagnose before provisioning — do not spend metal on hope.

- [ ] **Step 4: Commit the bench + amendment**

```bash
git add crates/inferno-kernels/benches docs/superpowers/specs/2026-07-17-m4b14-prefill-attention-query-blocking-design.md
git commit -m "M4b.14: local dev gate — block-kernel µbench + t=1 pp delta (non-quiet), gate PASS"
```

---

## Task 8: Mid-milestone quiet-hw gate (both boxes)

**Files:**
- Modify: `docs/superpowers/specs/2026-07-17-m4b14-prefill-attention-query-blocking-design.md` (§Amendments — record verbatim)

**Interfaces:**
- Consumes: Task 6's `gate-prefill-attn-split.sh`, the existing `gate-bench-protocol.sh`.

**Protocol** (quiet-hw runbook, devenv shell, release build; no parallel provisions; `metal-gc` to zero after):

- [ ] **Step 1: Provision box A (16c d2.c1.medium, 6336Y), run the split profile + protocol**

In the box's devenv shell, quiet-hw:
```bash
scripts/quiet-hw/gate-prefill-attn-split.sh <model>     # split-bracket t=1 profile
scripts/quiet-hw/gate-bench-protocol.sh <model>         # pp512 vs llama best-of
```
Capture both outputs verbatim.

- [ ] **Step 2: Provision box B (8c s2.c2.medium, E-2388G), same two scripts**

Same as Step 1 on box B. Run boxes sequentially (no parallel PNAP provisions — 403). After each box, `metal-gc` and confirm zero servers.

- [ ] **Step 3: Compute the gate arithmetic per box (pre-registered rule)**

For each box, from the split-bracket table:
1. `attn_share = attn_total / prefill_total` (attn_total = scores + softmax + output).
2. `pp_ratio` = post-Lever-1 pp512 vs llama best-of.
3. For each candidate Lever-2 menu entry targeting a sub-bracket of prefill-fraction `f` with pre-registered optimistic ceiling `c`: authorized on this box iff `pp_ratio / (1 - f*c) >= 1.0`.
4. Blame gate: the chosen entry must target the sub-bracket the fresh table actually blames (largest admissible `f`).

- [ ] **Step 4: Record the verdict verbatim in §Amendments**

Paste both boxes' split-bracket tables and pp ratios verbatim, show the `attn_share`, ceiling, and blame arithmetic, and state the verdict: **authorize exactly one Lever-2 menu entry** (only if its arithmetic clears 1.0x on a box AND the profile blames its sub-bracket), **or STOP** (an all-STOP with the finding closes the milestone as a diagnostic — Task 9 is skipped, go to Task 10). Also record the instrument admissibility check (sum of `attn:*` sub-brackets ≈ the whole-attention bracket; the instrument didn't perturb the measurement it reports).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-17-m4b14-prefill-attention-query-blocking-design.md
git commit -m "M4b.14: mid-milestone gate sessions (both boxes), ladder verdict"
```

---

## Task 9: Lever 2 (CONDITIONAL — only if Task 8 authorized it)

**Files:** depend on the authorized menu entry. **If Task 8 STOP'd, skip this task entirely** and proceed to Task 10 — do not build anything.

**Interfaces:**
- Consumes: the Task 8 verdict naming exactly one menu entry.

**The three pre-registered menu entries** (spec §Mid-Milestone Gate), each with its numerics discipline:

- [ ] **Step 1: If authorized entry = "wider query blocks / KV-panel prefetch tuning"** — a tuning change within the existing bit-identical block kernel (larger `m_block` panels, software prefetch of the next K/V panel). Implement, keep the Task 1/2 bit-identity property tests green (this entry does not touch the reduction order → `attn_rel_tol` unchanged). Re-run Task 5's differential suite.

- [ ] **Step 2: If authorized entry = "vectorized/batched softmax"** — share the max/denom reductions across the block's score rows using the `expf` poly and `reduce8` tree. **Bit-identity holds only if the per-row reduction order is preserved.** Extend the Task 1 property test to cover the batched softmax path. Re-derive `attn_rel_tol` from an `observed_error` sweep **only if** the block-shared reduction changes the order (flagged, data-armed — never loosened).

- [ ] **Step 3: If authorized entry = "flash-attention online-softmax fused pass"** — single KV sweep with running-max renormalize. **This breaks bit-identity** (running renormalize changes the reduction order and the exp count). Required discipline: (a) implement the fused scalar + AVX2 kernels with a new `observed_error_attention_flash` rig sweep; (b) re-derive `attn_rel_tol()` from that measured error distribution (data-armed, never loosened to make a red test green); (c) re-green `cargo test -p inferno-codegen --test differential` and `cargo test -p inferno-core --test artifact` with the new tolerance; (d) the scalar↔AVX2 fused kernels stay bit-identical to each other (the rig's isa-equality test extends to the fused kernel).

- [ ] **Step 4: Re-measure on both quiet-hw boxes** (same protocol as Task 8) and record the Lever-2 data point + any tolerance re-derivation in §Amendments.

- [ ] **Step 5: Commit** the authorized entry with its tests and amendment.

---

## Task 10: Closing quiet-hw session + exit-criteria walk

**Files:**
- Modify: `docs/superpowers/specs/2026-07-17-m4b14-prefill-attention-query-blocking-design.md` (§Amendments — closing verdict)
- Modify: `docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md` (§Amendments — closing bench data point, per the standing manual-protocol rule)

**Interfaces:**
- Consumes: all landed levers.

- [ ] **Step 1: Run the closing protocol on both boxes**

`gate-bench-protocol.sh <model>` on box A then box B (sequential; `metal-gc` to zero after each). Capture pp512 / tg128 vs llama best-of verbatim.

- [ ] **Step 2: Walk the four exit criteria and write the closing verdict**

In the spec §Amendments, a "closing verdict: exit-criteria walk" section:
1. Local dev data point recorded (Task 7)?
2. Fresh split-bracket profiles + gate verdict with arithmetic recorded (Task 8)?
3. Every gate outcome recorded; no lever shipped without its gate (Lever 1 after Task 7 PASS; Lever 2 only if Task 8 authorized)?
4. **Closing pp512 vs llama best-of on both boxes** — is `pp ≥ 1.0x` on both? If yes: **v1 pp criterion MET (for this model)**. If no: record the STOP finding (which sub-bracket still dominates, the residual-gap shape) and close as a **diagnostic** (M4b.12/M4b.13 precedent). tg is context-only, never the gate.

- [ ] **Step 3: Record the closing bench data point in the M4a spec**

Append the verbatim closing pp/tg numbers to `2026-07-06-m4a-bench-sampling-design.md` §Amendments (dated), per the standing rule that every `inferno bench` report is recorded there and never edited.

- [ ] **Step 4: Final full-suite green + lint before closing**

Run: `mise run test 2>&1 | tail -10 && mise run lint 2>&1 | tail -5 && mise run differential 2>&1 | tail -5`
Expected: all green.

- [ ] **Step 5: Commit the closing verdict**

```bash
git add docs/superpowers/specs/2026-07-17-m4b14-prefill-attention-query-blocking-design.md docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md
git commit -m "M4b.14: closing quiet-hw session + exit-criteria walk verdict"
```

- [ ] **Step 6: Update AGENTS.md if the shipping behavior changed**

If Lever 1 (or an authorized Lever 2) changed a durable, non-obvious constraint (e.g. "prefill attention is query-blocked below `inferno_par_attention`; the block kernel is bit-identical to the per-token kernel and the rig's qblock tests are the guard"), add a bullet to `AGENTS.md`'s decode/prefill threading note. Commit.

```bash
git add AGENTS.md
git commit -m "M4b.14: AGENTS.md — record prefill attention query-blocking invariant"
```

---

## Self-Review

**1. Spec coverage:**
- Lever 1 query-blocked kernel (two-pass, bit-identical, K/V streamed once per block) → Tasks 1-2. ✓
- Pool plumbing (`run_attn_span` one call per shard, `AttnJob` scratch resize, sharded query axis) → Task 3. ✓
- Codegen unchanged except symbol + mandatory `HOST_ABI_VERSION` bump → Task 4. ✓
- Bit-identity invariants (qblock==per-token, scalar==AVX2, mb=1==per-token, cross-thread, cross-tile) → Tasks 1, 2, 3 (pool rig), 5. ✓
- `attn_rel_tol` untouched for Lever 1; differential green → Task 5. ✓
- Sub-bracket instrument (feature-gated, off in shipping/bench) + `gate-prefill-attn-split.sh` → Task 6. ✓
- Local dev gate → Task 7. Mid-milestone gate with ceiling/blame arithmetic → Task 8. Pre-registered Lever-2 menu with per-entry numerics discipline → Task 9. Closing exit-criteria walk + M4a amendment → Task 10. ✓
- Explicit out-of-scope (decode attention, F16 KV, flash as un-gated Lever 1, attention-as-GEMM) → respected; flash only appears gated in Task 9. ✓
- Standing invariant "`m == 1` bit-equals decode/GEMV attention path" → Task 3 (m==1 path unchanged, block kernel mb=1 reference) + Task 1 test covers mb=1. ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Kernel bodies, pool rewrite, codegen edits, and the gate script are shown in full. Task 9's branches are conditional-by-design (one of three pre-registered entries), each with concrete numerics steps — not placeholders. ✓

**3. Type consistency:** `AttnBlockFn` (14 args) is defined once in the pool (Task 3) and consumed by `AttnJob.kernel` and `inferno_par_attention`. The block-kernel C signature (`out, q, kv, scores, kv_base, v_off, pos0, m_block, kv_dim, n_heads, n_kv_heads, head_dim, q_stride, out_stride`) matches across the kernel (Tasks 1-2), the pool type + call (Task 3), and the codegen extern declaration (Task 4, 14 params). `attention_symbol` → `_qblock` (Task 4) matches the kernel symbol names (Tasks 1-2). `HOST_ABI_VERSION` "8" is set once (Task 4). ✓
