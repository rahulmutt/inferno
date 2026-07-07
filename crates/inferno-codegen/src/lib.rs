//! Plan -> Loop IR -> LLVM IR (inkwell) -> object -> model.so. The only crate
//! that links LLVM (18, matching devenv). See the M3 spec.

pub mod emit;
pub mod error;
pub mod llvm;
pub mod loopir;
pub use emit::{Artifact, Meta, compile};
pub use error::{CodegenError, Result};

/// Version of the host-symbol surface generated code links against
/// (kernel symbols + `inferno_par_gemv`). Folded into `inferno-core`'s
/// artifact cache key: bump it whenever the emitted code's host-call shape
/// changes, so stale cached `model.so`s are recompiled instead of silently
/// running with the old call pattern. "2" = M4b.1's `inferno_par_gemv`
/// dispatch (v1 was M3's direct kernel calls).
pub const HOST_ABI_VERSION: &str = "2";

#[cfg(test)]
mod smoke {
    use inkwell::context::Context;

    #[test]
    fn builds_empty_module() {
        let ctx = Context::create();
        let module = ctx.create_module("smoke");
        assert!(module.verify().is_ok());
    }
}
