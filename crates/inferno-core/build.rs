//! Export the test binaries' dynamic symbol table (mirrors
//! `inferno-codegen/build.rs`).
//!
//! The compiled `model.so` that `Artifact::load_or_compile` `dlopen`s has
//! UNDEFINED kernel symbols (`inferno_gemv_*`, `inferno_quantize_row_*`). They
//! resolve at `dlopen` time against the host binary's global symbol scope, but
//! Rust does not place `#[no_mangle]` symbols in the dynamic symbol table
//! without `-rdynamic` (`--export-dynamic`), so without it `dlopen` fails with
//! `undefined symbol: inferno_gemv_...`.
//!
//! `rustc-link-arg-tests` applies the flag only to this crate's test binaries
//! (where the `artifact` differential runs). Symbol *retention* — keeping the
//! `inferno-kernels` object in the link at all — is handled at runtime by
//! `artifact::ensure_kernels_linked`, which every binary using this crate runs.
//!
//! NOTE for Task 16 (the CLI): the CLI is a normal `bin`, not a test, so it
//! must add its OWN `-rdynamic` (e.g. `cargo:rustc-link-arg-bins=-rdynamic` in
//! the CLI crate's build.rs, or a `.cargo/config.toml` rustflag) for runtime
//! `dlopen` to resolve the kernels. This build.rs deliberately scopes to tests.
fn main() {
    println!("cargo:rustc-link-arg-tests=-rdynamic");
}
