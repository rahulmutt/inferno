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
/// (kernel symbols + `inferno_par_{gemv,gemm,attention,token_loop}` + the
/// profiler global). Folded into `inferno-core`'s artifact cache key.
/// "8" = M4b.14's query-blocked prefill attention kernel
/// (`inferno_attention_f32_*_qblock`; `inferno_par_attention` now invokes
/// its kernel argument with the block ABI — a stale artifact would pass the
/// old per-token pointer to the block-calling dispatcher, so the bump is
/// mandatory to force recompile);
/// "7" = M4b.11's head-sharded decode attention
/// (`inferno_par_attention_heads` dispatch + `inferno_attention_f32_*_hspan`
/// kernel symbols); "6" = M4b.9's `inferno_par_token_loop` dispatch; "5" =
/// M4b.8's `inferno_par_attention` dispatch; "4" was M4b.3's attention
/// kernel symbols (`inferno_attention_f32_{scalar,avx2}`); "3" was M4b.2's
/// GEMM dispatch + optional profiling (v2 was M4b.1's `inferno_par_gemv`).
pub const HOST_ABI_VERSION: &str = "8";

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
