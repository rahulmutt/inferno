//! Plan -> Loop IR -> LLVM IR (inkwell) -> object -> model.so. The only crate
//! that links LLVM (18, matching devenv). See the M3 spec.

pub mod emit;
pub mod error;
pub mod llvm;
pub mod loopir;
pub mod profile;
pub use emit::{Artifact, CompileOptions, Meta, compile};
pub use error::{CodegenError, Result};

/// Version of the host-symbol surface generated code links against
/// (kernel symbols + `inferno_par_gemv` + `inferno_par_gemm` + the profiler
/// global). Folded into `inferno-core`'s artifact cache key. "3" = M4b.2's
/// GEMM dispatch + optional profiling (v2 was M4b.1's `inferno_par_gemv`).
pub const HOST_ABI_VERSION: &str = "3";

/// Default prefill tile length (tokens per batched forward pass). Part of
/// `CompileOptions` and the artifact cache key.
pub const PREFILL_TILE: usize = 64;

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
