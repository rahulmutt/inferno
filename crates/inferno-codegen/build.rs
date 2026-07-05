//! Export the integration-test binaries' dynamic symbol table.
//!
//! The compiled `model.so` produced by `compile` has UNDEFINED kernel symbols
//! (`inferno_gemv_*`, `inferno_quantize_row_*`). When the `differential` test
//! `dlopen`s it, those symbols must resolve against the host test binary's
//! global scope. Rust does not place `#[no_mangle]` symbols in the dynamic
//! symbol table by default, so without `-rdynamic` (`--export-dynamic`) the
//! `dlopen` fails with `undefined symbol: inferno_gemv_...`.
//!
//! `rustc-link-arg-tests` applies the flag only to integration-test binaries
//! (not the library or downstream crates), which is exactly where the
//! differential harness runs. Symbol retention (so the linker keeps
//! `inferno-kernels` at all) is handled inside the test via `black_box`
//! references to each kernel symbol.
fn main() {
    println!("cargo:rustc-link-arg-tests=-rdynamic");
}
