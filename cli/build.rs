//! The compiled path's `model.so` has UNDEFINED kernel symbols
//! (`inferno_gemv_*`, `inferno_quantize_row_*`) resolved at `dlopen` time
//! against the running process's global symbol scope. Rust does not put
//! `#[no_mangle]` symbols in a binary's dynamic symbol table without
//! `-rdynamic` (`--export-dynamic`), so without this the real `inferno`
//! binary's `run`/`compile` commands fail at runtime with `undefined symbol:
//! inferno_gemv_...` the first time they `dlopen` a compiled artifact.
//!
//! Mirrors `inferno-core/build.rs`'s `rustc-link-arg-tests`, but scoped to
//! `bins` (the `inferno` binary itself is not a test): the retention half of
//! this contract — keeping `inferno-kernels` in the link at all — is handled
//! at runtime by `inferno_core::ensure_kernels_linked`, called from
//! `CompiledBackend::new` on every compiled-path construction.
fn main() {
    println!("cargo:rustc-link-arg-bins=-rdynamic");
}
