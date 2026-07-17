//! The compiled-artifact loader: mmap `weights.bin`, `dlopen` `model.so`,
//! resolve the `prefill`/`decode_step` entry points, and call through them.
//!
//! This is the second sanctioned `unsafe` crate (after `inferno-kernels`). It
//! carries the FFI boundary of the M3 compiler: memory-mapping the weight
//! image, loading a compiled shared object, and calling a raw C-ABI entry
//! point through a function pointer. Every `unsafe` block below documents its
//! invariant with a `// SAFETY:` comment.
//!
//! ## Trust boundary
//! A cached `model.so` is executable code. Before it is ever `dlopen`ed,
//! [`Artifact::load_or_compile`] re-verifies the cache's content hashes
//! (`weights.bin`, the model file) and the compiler version against the sidecar
//! `meta.json`. Any mismatch discards the cache entry and recompiles from the
//! model — a tampered or stale artifact is never loaded. This is the
//! trusted-local boundary control from the threat model.
//!
//! ## Alignment / sizing contracts (from the Task 12 differential gate)
//! - `weights.bin` is **mmapped** (page-aligned, 4096) rather than
//!   `fs::read` (~16-aligned): the rs8 AVX2 GEMV kernels do 32-byte *aligned*
//!   loads on the weight base, so a heap buffer would SIGSEGV.
//! - `meta.arena_f32` already includes the activation-quant scratch region, so
//!   the arena is a single `vec![0f32; meta.arena_f32]`.
//! - `kv` is `vec![0f32; meta.kv_total_bytes / 4]`, `logits_out` is
//!   `vec![0f32; meta.vocab]`.

use std::os::fd::AsFd;
use std::path::Path;
use std::ptr::NonNull;

use inferno_codegen::Meta;
use inferno_target::TargetDesc;

use crate::{CoreError, Result, cache_dir, cache_key, content_hash};

/// `prefill(tokens, n, pos_off, weights, kv, arena, logits_out)` — the C ABI of
/// the generated entry point (see `declare_entry_points` in inferno-codegen).
/// `n`/`pos_off` are `i64` params, surfaced as `usize` (same width/repr on the
/// LP64 targets this crate builds for).
type PrefillFn = unsafe extern "C" fn(
    *const u32, // tokens
    usize,      // n
    usize,      // pos_off
    *const u8,  // weights image base
    *mut f32,   // kv cache
    *mut f32,   // arena
    *mut f32,   // logits_out
);

/// `decode_step(token, pos, weights, kv, arena, logits_out)` — the single-token
/// entry point. `token` is an `i32`/`u32` (bit-identical), `pos` an `i64`.
type DecodeStepFn = unsafe extern "C" fn(
    u32,       // token
    usize,     // pos
    *const u8, // weights image base
    *mut f32,  // kv cache
    *mut f32,  // arena
    *mut f32,  // logits_out
);

/// A read-only, page-aligned memory map of a file, unmapped on drop.
///
/// Backed by `rustix::mm::mmap` (`PROT_READ`/`MAP_PRIVATE`). The base pointer
/// is page-aligned (4096), which satisfies the rs8 kernels' 32-byte
/// aligned-load requirement on the weight base.
struct Mmap {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the mapping is `PROT_READ` (immutable) and owns its region for its
// whole lifetime; sharing/moving the raw pointer across threads is sound
// because it is never written through and `munmap` runs exactly once on drop.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    /// Read-only mmap of the entire file at `path`.
    fn open(path: &Path) -> Result<Mmap> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            // mmap of length 0 is EINVAL; represent an empty file as a dangling
            // (but well-aligned) map so callers get a valid empty slice.
            return Ok(Mmap {
                ptr: NonNull::dangling(),
                len: 0,
            });
        }
        // SAFETY: `file` is a valid open fd held for the duration of the call;
        // `len` is its real size; a null hint lets the kernel choose the
        // (page-aligned) address. The map is read-only and private, so no other
        // process can observe or mutate it, and we never write through it.
        let addr = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                len,
                rustix::mm::ProtFlags::READ,
                rustix::mm::MapFlags::PRIVATE,
                file.as_fd(),
                0,
            )
        }
        .map_err(|e| CoreError::Mmap(std::io::Error::from_raw_os_error(e.raw_os_error())))?;
        // mmap never returns null on success; MAP_FAILED surfaces as Err above.
        let ptr = NonNull::new(addr as *mut u8)
            .ok_or_else(|| CoreError::Mmap(std::io::Error::other("mmap returned null")))?;
        Ok(Mmap { ptr, len })
    }

    /// The mapped bytes. Page-aligned base; valid for the `Mmap`'s lifetime.
    fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr`..`ptr+len` is a single live read-only mapping we own;
        // the borrow is tied to `&self`, so the region cannot be unmapped while
        // the slice is alive.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: `ptr`/`len` are exactly the base/length returned by `mmap`
        // and never handed out as an owning pointer, so this is the unique,
        // final unmap of a still-mapped region.
        unsafe {
            let _ = rustix::mm::munmap(self.ptr.as_ptr() as *mut _, self.len);
        }
    }
}

/// A loaded, verified compiled model ready to run.
///
/// Owns the `dlopen`ed [`libloading::Library`] and the [`Mmap`] of
/// `weights.bin` for its whole lifetime: dropping either invalidates the
/// resolved function pointers / weight base. The raw `prefill`/`decode_step`
/// pointers are extracted from the library (via `Symbol` -> raw fn) so the
/// struct is not self-referential; they stay valid as long as `_lib` lives.
pub struct Artifact {
    // Field drop order is top-to-bottom: fn ptrs (Copy, no drop), then the lib,
    // then the mmap. Keeping `_lib`/`_weights` last-referenced anchors their
    // lifetimes; they MUST outlive any call through the fn pointers.
    prefill: PrefillFn,
    decode: DecodeStepFn,
    weights: Mmap,
    meta: Meta,
    _lib: libloading::Library,
    /// Base of the profiled `model.so`'s `[N x i64]` counter array, resolved
    /// at load time when `meta.profile_slots` is non-empty; None otherwise.
    prof_counters: Option<NonNull<u64>>,
}

// SAFETY: `Artifact` is immutable after construction. `prefill`/`decode` are
// plain fn pointers; `Mmap` is `Send + Sync` (read-only); `libloading::Library`
// is `Send + Sync`. The compiled entry points read weights and write only into
// caller-provided buffers, so concurrent shared (`&self`) use is sound.
// `prof_counters` is a raw pointer into the profiled artifact's global
// counter array; it is only ever read/written by the single-threaded CLI
// `--profile` measurement path (never concurrently with `prefill`/
// `decode_step`, and never from more than one thread), so its presence does
// not weaken the `Send`/`Sync` argument above.
unsafe impl Send for Artifact {}
unsafe impl Sync for Artifact {}

impl Artifact {
    /// Load a compiled model for `model`/`target`/`max_seq_len` from the cache,
    /// or compile it if absent or if the cached artifact fails verification.
    ///
    /// On a cache hit whose hashes and compiler version verify, the cached
    /// `model.so`/`weights.bin` are loaded directly. Otherwise the model is
    /// (re)compiled and published into the cache directory (see
    /// [`compile_and_publish`](Self::compile_and_publish) for the atomicity
    /// contract), the real content hashes are written into `meta.json`, and
    /// the fresh artifact is loaded. A cached artifact whose `weights.bin`/
    /// model hash or `inferno_version` does not match is discarded and
    /// recompiled — never `dlopen`ed.
    pub fn load_or_compile(
        model: &Path,
        target: &TargetDesc,
        max_seq_len: usize,
        opts: &inferno_codegen::CompileOptions,
    ) -> Result<Artifact> {
        let key = cache_key(model, target, max_seq_len, opts)?;
        let dir = cache_dir(&key);

        // Try the cache. Any verification failure (missing files, hash/version
        // mismatch) falls through to a clean recompile.
        if dir.join("meta.json").exists() {
            match verify_cache(&dir, model) {
                Ok(meta) => return Self::load_from(&dir, meta),
                Err(CoreError::Verification(_)) => {
                    // Stale/tampered: discard the old dir before recompiling
                    // so the atomic publish below (`rename` in
                    // `compile_and_publish`) never lands on a stale
                    // non-empty directory it doesn't own. If a concurrent
                    // process re-publishes a fresh `dir` in the window
                    // between this remove and our later rename, that's the
                    // ordinary lost-race case `compile_and_publish` already
                    // handles.
                    match std::fs::remove_dir_all(&dir) {
                        Ok(()) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(e.into()),
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Self::compile_and_publish(model, target, max_seq_len, opts, &key, &dir)
    }

    /// Compile a fresh artifact into a unique staging directory beside `dir`
    /// (`<parent>/<key>.tmp-<pid>` — same parent, hence same filesystem, so
    /// the publish below is a single atomic `rename`), then publish it as
    /// `dir`.
    ///
    /// This is the race fix: two processes racing a cold cache each compile
    /// into their OWN staging directory (no shared partially-written
    /// `model.so`/`meta.json` for either to observe). Whichever calls
    /// `rename` first wins — its staging dir atomically becomes `dir`, and
    /// it loads straight from there. The loser's `rename` fails because
    /// `dir` now exists (a `rename` onto a non-empty directory fails on
    /// Linux); the loser discards its own staging dir and loads the
    /// winner's freshly published artifact instead of recompiling. If the
    /// winner's artifact somehow fails verification, that error is returned
    /// directly rather than looping.
    fn compile_and_publish(
        model: &Path,
        target: &TargetDesc,
        max_seq_len: usize,
        opts: &inferno_codegen::CompileOptions,
        key: &str,
        dir: &Path,
    ) -> Result<Artifact> {
        let parent = dir
            .parent()
            .expect("cache_dir(key) always nests under a parent directory");
        std::fs::create_dir_all(parent)?;
        // `<pid>` alone disambiguates the common case (separate racing
        // processes, e.g. two CI jobs / two `inferno` invocations), but this
        // engine can also be driven by multiple threads of one process, which
        // all share a pid; append the thread id too so concurrent compiles
        // within a single process never collide on the same staging dir.
        // `ThreadId`s are never reused for the life of the process, so the
        // pair is unique across both axes.
        let staging = parent.join(format!(
            "{key}.tmp-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        // Best-effort cleanup of a leftover staging dir (e.g. a prior crash
        // under a reused pid): each attempt should start from a clean dir so
        // `compile`'s `create_dir_all` doesn't merge with stale contents.
        let _ = std::fs::remove_dir_all(&staging);

        // Compile + finalize entirely into the staging dir; `dir` is never
        // touched until the rename below.
        let desc = inferno_formats::load_desc(model)?;
        let graph = inferno_graph::build_graph(&desc)?;
        inferno_codegen::compile(&desc, &graph, target, max_seq_len, opts, &staging)?;
        // codegen leaves the hash fields empty; fill them with the real content
        // hashes so subsequent loads can verify integrity.
        let meta = finalize_meta(&staging, model, target)?;

        match std::fs::rename(&staging, dir) {
            Ok(()) => Self::load_from(dir, meta),
            Err(_) if dir.join("meta.json").exists() => {
                // Lost the race: another process's rename beat ours. Discard
                // our staging dir and load the winner's published artifact
                // rather than recompiling again.
                let _ = std::fs::remove_dir_all(&staging);
                let winner_meta = verify_cache(dir, model)?;
                Self::load_from(dir, winner_meta)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// mmap `weights.bin`, `dlopen` `model.so`, and resolve the entry points.
    fn load_from(dir: &Path, meta: Meta) -> Result<Artifact> {
        // Keep the kernel symbols in the host binary and (with `-rdynamic`)
        // exported, so `model.so`'s undefined `inferno_gemv_*` /
        // `inferno_quantize_row_*` resolve against us at dlopen time.
        ensure_kernels_linked();

        let weights = Mmap::open(&dir.join("weights.bin"))?;

        // SAFETY: we load a shared object we just compiled (or verified by hash)
        // from the trusted cache directory. `Library::new` runs the object's
        // initializers; our objects have none with side effects.
        let lib = unsafe { libloading::Library::new(dir.join("model.so")) }?;

        // SAFETY: `prefill`/`decode_step` are declared in `model.so` with
        // exactly the C ABI of `PrefillFn`/`DecodeStepFn` (see codegen
        // `declare_entry_points`). The returned `Symbol` borrows `lib`; we copy
        // out the raw fn pointer immediately and keep `lib` alive in the
        // returned `Artifact`, so the pointer stays valid for `Artifact`'s life.
        let (prefill, decode) = unsafe {
            let p: libloading::Symbol<PrefillFn> = lib.get(b"prefill\0")?;
            let d: libloading::Symbol<DecodeStepFn> = lib.get(b"decode_step\0")?;
            (*p, *d)
        };

        // SAFETY: a profiled artifact exports `inferno_prof_counters` as
        // `[N x i64]` with N == meta.profile_slots.len(); we copy out the raw
        // base pointer and keep `lib` alive in the returned Artifact.
        let prof_counters = if meta.profile_slots.is_empty() {
            None
        } else {
            let sym: libloading::Symbol<*mut u64> = unsafe { lib.get(b"inferno_prof_counters\0") }?;
            // `*sym` just derefs the `Symbol` wrapper (safe `Deref`) to copy
            // out the raw `*mut u64` it wraps; no unsafe needed here.
            NonNull::new(*sym)
        };

        Ok(Artifact {
            prefill,
            decode,
            weights,
            meta,
            _lib: lib,
            prof_counters,
        })
    }

    /// Run a batched prefill over `tokens` starting at position `pos_off`,
    /// writing the final-token logits into `logits_out`.
    ///
    /// `kv`/`arena`/`logits_out` must be sized per [`meta`](Self::meta):
    /// `kv.len() >= kv_total_bytes/4`, `arena.len() >= arena_f32`,
    /// `logits_out.len() >= vocab`. Panics otherwise (a sizing bug would
    /// otherwise let the kernels write out of bounds).
    pub fn prefill(
        &self,
        tokens: &[u32],
        pos_off: usize,
        kv: &mut [f32],
        arena: &mut [f32],
        logits_out: &mut [f32],
    ) {
        self.assert_buffers(kv, arena, logits_out);
        // The `kv` buffer is sized for `max_seq_len`; the compiled kernels write
        // KV entries at positions `pos_off..pos_off+tokens.len()`. A caller that
        // exceeds `max_seq_len` would drive an OOB KV write through this SAFE
        // method, so bound the write position here (checked add avoids an
        // overflow wrapping past the guard). Hard `assert!` (release too): this
        // converts UB into a clean panic; position management is the caller's job.
        let end = pos_off
            .checked_add(tokens.len())
            .expect("prefill: pos_off + tokens overflows usize");
        assert!(
            end <= self.meta.max_seq_len,
            "prefill: pos_off ({pos_off}) + tokens ({}) exceeds max_seq_len ({})",
            tokens.len(),
            self.meta.max_seq_len
        );
        // SAFETY: `tokens` is a valid `&[u32]` (ptr+len passed together); the
        // weight base is the page-aligned mmap owned by `self`; `kv`/`arena`/
        // `logits_out` are exclusive `&mut` slices asserted large enough above,
        // and the write position `pos_off+tokens.len() <= max_seq_len` is
        // asserted, so every KV write lands inside `kv`. The fn pointer's ABI
        // matches `PrefillFn` and `self._lib`/`self.weights` outlive this call.
        // The compiled code only writes within these buffers.
        unsafe {
            (self.prefill)(
                tokens.as_ptr(),
                tokens.len(),
                pos_off,
                self.weights.as_slice().as_ptr(),
                kv.as_mut_ptr(),
                arena.as_mut_ptr(),
                logits_out.as_mut_ptr(),
            );
        }
    }

    /// Decode a single `token` at position `pos`, writing logits into
    /// `logits_out`. Buffer sizing / panic contract as in [`prefill`](Self::prefill).
    pub fn decode_step(
        &self,
        token: u32,
        pos: usize,
        kv: &mut [f32],
        arena: &mut [f32],
        logits_out: &mut [f32],
    ) {
        self.assert_buffers(kv, arena, logits_out);
        // Bound the KV write position: the kernel writes the KV entry at `pos`,
        // and `kv` is sized for `max_seq_len`. A loop calling `decode_step` past
        // `max_seq_len` would write OOB through this SAFE method. Hard `assert!`
        // (release too) turns that UB into a clean panic.
        assert!(
            pos < self.meta.max_seq_len,
            "decode_step: pos ({pos}) >= max_seq_len ({})",
            self.meta.max_seq_len
        );
        // SAFETY: identical invariants to `prefill`; `token`/`pos` are scalars,
        // the weight base is the owned page-aligned mmap, the `&mut` buffers are
        // exclusive and asserted large enough, and `pos < max_seq_len` is
        // asserted so the KV write lands inside `kv`.
        unsafe {
            (self.decode)(
                token,
                pos,
                self.weights.as_slice().as_ptr(),
                kv.as_mut_ptr(),
                arena.as_mut_ptr(),
                logits_out.as_mut_ptr(),
            );
        }
    }

    /// The sidecar metadata (buffer sizes, entry-point names, hashes).
    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Profiler slot labels (empty unless compiled with `profile`).
    pub fn profile_slots(&self) -> &[String] {
        &self.meta.profile_slots
    }

    /// Current per-slot cycle counters, or None if unprofiled. Reads the raw
    /// `[N x i64]` global the compiled code accumulates into.
    pub fn profile_snapshot(&self) -> Option<Vec<u64>> {
        let base = self.prof_counters?;
        let n = self.meta.profile_slots.len();
        // SAFETY: `base` points at the artifact's live `[n x i64]` global for
        // as long as `self._lib` is alive; we only read it.
        Some(unsafe { std::slice::from_raw_parts(base.as_ptr(), n).to_vec() })
    }

    /// Zero the counters (separates prefill vs decode measurement).
    pub fn profile_reset(&self) {
        if let Some(base) = self.prof_counters {
            let n = self.meta.profile_slots.len();
            // SAFETY: exclusive logical access — the CLI resets between phases
            // while no forward pass is running.
            unsafe { std::ptr::write_bytes(base.as_ptr(), 0, n) };
        }
    }

    /// Validate caller buffers against `meta` before any raw call. A too-small
    /// buffer would let the compiled kernels write out of bounds, so this is a
    /// hard precondition, not a hint.
    fn assert_buffers(&self, kv: &[f32], arena: &[f32], logits_out: &[f32]) {
        assert!(
            kv.len() >= self.meta.kv_total_bytes / 4,
            "kv buffer too small: {} < {}",
            kv.len(),
            self.meta.kv_total_bytes / 4
        );
        assert!(
            arena.len() >= self.meta.arena_f32,
            "arena too small: {} < {}",
            arena.len(),
            self.meta.arena_f32
        );
        assert!(
            logits_out.len() >= self.meta.vocab,
            "logits_out too small: {} < {}",
            logits_out.len(),
            self.meta.vocab
        );
    }
}

/// Re-verify a cached artifact before trusting its `model.so`. Recomputes the
/// `weights.bin` and model-file content hashes and compares them (plus the
/// compiler version) against `meta.json`. Returns the parsed [`Meta`] on
/// success, or [`CoreError::Verification`] on any mismatch / missing file.
///
/// This is the trusted-local boundary control: `load_or_compile` calls it
/// before ever `dlopen`ing a cached `model.so`, and it is exposed so callers /
/// tests can audit a cache entry's integrity independently.
pub fn verify_cache(dir: &Path, model: &Path) -> Result<Meta> {
    let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;

    let fail = |what: &str| Err(CoreError::Verification(what.to_string()));

    if meta.inferno_version != env!("CARGO_PKG_VERSION") {
        return fail("inferno_version mismatch");
    }
    if !dir.join("model.so").exists() {
        return fail("model.so missing");
    }

    let weights = std::fs::read(dir.join("weights.bin"))?;
    if content_hash(&weights) != meta.weights_hash {
        return fail("weights.bin hash mismatch");
    }
    // `model` may be a single file (GGUF/safetensors) or a directory (MLX);
    // `read_model_bytes` handles both the same way `cache_key` does.
    let model_bytes = crate::cache::read_model_bytes(model)?;
    if content_hash(&model_bytes) != meta.model_hash {
        return fail("model hash mismatch");
    }
    Ok(meta)
}

/// After a fresh compile, recompute and persist the real content hashes into
/// `meta.json` (codegen leaves them empty). Returns the finalized [`Meta`].
fn finalize_meta(dir: &Path, model: &Path, target: &TargetDesc) -> Result<Meta> {
    let mut meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    meta.weights_hash = content_hash(&std::fs::read(dir.join("weights.bin"))?);
    meta.model_hash = content_hash(&crate::cache::read_model_bytes(model)?);
    meta.target_hash = content_hash(format!("{target:?}").as_bytes());
    std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
    Ok(meta)
}

/// Force the linker to retain (and, in a `-rdynamic` binary, export) every
/// kernel symbol *and the `inferno_par_gemv` dispatcher* a compiled
/// `model.so` resolves against the host binary.
///
/// Without at least one live reference the linker may drop `inferno-kernels`
/// (or `inferno-pool`) entirely, leaving nothing to export and `dlopen`
/// failing on the first undefined `inferno_gemv_*` / `inferno_quantize_row_*`
/// / `inferno_par_gemv` symbol. This is the reusable retention mechanism
/// (Task 16's CLI calls it too); the CLI must additionally pass `-rdynamic`
/// at link time (see build.rs note).
pub fn ensure_kernels_linked() {
    use std::hint::black_box;
    let p = |f: *const ()| black_box(f as usize);
    p(inferno_kernels::inferno_gemv_f32_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemv_f32_rs8_avx2 as *const ());
    p(inferno_kernels::inferno_gemv_q8_0_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemv_q8_0_rs8_avx2 as *const ());
    p(inferno_kernels::inferno_gemv_q4_k_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemv_q4_k_rs8_avx2 as *const ());
    p(inferno_kernels::act::inferno_quantize_row_q8a_scalar as *const ());
    p(inferno_kernels::act::inferno_quantize_row_q8a_avx2 as *const ());
    p(inferno_kernels::act::inferno_quantize_row_q8k_scalar as *const ());
    p(inferno_kernels::act::inferno_quantize_row_q8k_avx2 as *const ());
    p(inferno_kernels::inferno_gemm_f32_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemm_f32_rs8_avx2 as *const ());
    p(inferno_kernels::inferno_gemm_q8_0_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemm_q8_0_rs8_avx2 as *const ());
    p(inferno_kernels::inferno_gemm_q4_k_rs8_scalar as *const ());
    p(inferno_kernels::inferno_gemm_q4_k_rs8_avx2 as *const ());
    p(inferno_kernels::inferno_attention_f32_scalar as *const ());
    p(inferno_kernels::inferno_attention_f32_avx2 as *const ());
    p(inferno_kernels::inferno_attention_f32_scalar_hspan as *const ());
    p(inferno_kernels::inferno_attention_f32_avx2_hspan as *const ());
    p(inferno_kernels::inferno_attention_f32_scalar_qblock as *const ());
    p(inferno_kernels::inferno_attention_f32_avx2_qblock as *const ());
    p(inferno_pool::inferno_par_gemv as *const ());
    p(inferno_pool::inferno_par_gemm as *const ());
    p(inferno_pool::inferno_par_attention as *const ());
    p(inferno_pool::inferno_par_token_loop as *const ());
    p(inferno_pool::inferno_par_attention_heads as *const ());
}
