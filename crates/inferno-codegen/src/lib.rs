//! Plan -> Loop IR -> LLVM IR (inkwell) -> object -> model.so. The only crate
//! that links LLVM (18, matching devenv). See the M3 spec.

pub mod error;
pub mod llvm;
pub mod loopir;
pub use error::{CodegenError, Result};

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
