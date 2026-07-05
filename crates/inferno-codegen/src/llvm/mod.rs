//! LLVM module scaffold: entry-point signatures + kernel declarations.
//!
//! Builds the empty-bodied LLVM IR skeleton for a compiled model: extern
//! declarations for every gemv/quantize kernel symbol (frozen M2 ABI) plus
//! the two generated entry points (`prefill`, `decode_step`). Tasks 9-10 fill
//! in the entry-point bodies with real op lowering.
//!
//! inkwell 0.6 / LLVM 18 use opaque pointers: there is a single `ptr` type
//! (no more typed `i8*`/`float*`), constructed via `Context::ptr_type`. Every
//! pointer-typed parameter below (`y`, `xq`, `w`, `tokens`, `weights`, `kv`,
//! `arena`, `logits_out`) uses that one opaque `ptr` type; only scalars stay
//! distinctly typed (`i64` for `size_t`, `i32` for the raw `token` id).

use crate::Result;
use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::values::FunctionValue;

/// Context-borrowing wrapper around an inkwell [`Module`] that knows how to
/// populate itself with the frozen kernel ABI and the (currently empty)
/// entry-point signatures.
pub struct LlvmModule<'c> {
    ctx: &'c Context,
    module: Module<'c>,
}

impl<'c> LlvmModule<'c> {
    pub fn new(ctx: &'c Context, name: &str) -> Self {
        Self {
            ctx,
            module: ctx.create_module(name),
        }
    }

    /// Declare extern decls for every gemv/quantize kernel symbol the
    /// compiled model may call. The ABI here is frozen (M2 kernels); Tasks
    /// 9-12 depend on these exact signatures.
    pub fn declare_kernels(&self) {
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let i64_t = self.ctx.i64_type();
        let void = self.ctx.void_type();

        // void inferno_gemv_<d>_rs8_<isa>(ptr y, ptr xq, ptr w, i64 k, i64 row_start, i64 row_end)
        let gemv_ty = void.fn_type(
            &[
                ptr.into(),
                ptr.into(),
                ptr.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
            ],
            false,
        );
        for d in ["f32", "q8_0", "q4_k"] {
            for isa in ["scalar", "avx2"] {
                self.module.add_function(
                    &format!("inferno_gemv_{d}_rs8_{isa}"),
                    gemv_ty,
                    Some(Linkage::External),
                );
            }
        }

        // void inferno_quantize_row_<q>_<isa>(ptr x, ptr y, i64 k)
        let quantize_ty = void.fn_type(&[ptr.into(), ptr.into(), i64_t.into()], false);
        for q in ["q8a", "q8k"] {
            for isa in ["scalar", "avx2"] {
                self.module.add_function(
                    &format!("inferno_quantize_row_{q}_{isa}"),
                    quantize_ty,
                    Some(Linkage::External),
                );
            }
        }
    }

    /// Define the two generated entry points with empty (`ret void`) bodies.
    /// Tasks 9-10 replace the bodies with real op lowering; the signatures
    /// here must not change since later tasks depend on them.
    pub fn define_entry_points(&self) -> (FunctionValue<'c>, FunctionValue<'c>) {
        let ptr = self.ctx.ptr_type(AddressSpace::default());
        let i64_t = self.ctx.i64_type();
        let i32_t = self.ctx.i32_type();
        let void = self.ctx.void_type();

        // void prefill(ptr tokens, i64 n, i64 pos_off, ptr weights, ptr kv, ptr arena, ptr logits_out)
        let prefill_ty = void.fn_type(
            &[
                ptr.into(),
                i64_t.into(),
                i64_t.into(),
                ptr.into(),
                ptr.into(),
                ptr.into(),
                ptr.into(),
            ],
            false,
        );
        let prefill = self.module.add_function("prefill", prefill_ty, None);

        // void decode_step(i32 token, i64 pos, ptr weights, ptr kv, ptr arena, ptr logits_out)
        let decode_step_ty = void.fn_type(
            &[
                i32_t.into(),
                i64_t.into(),
                ptr.into(),
                ptr.into(),
                ptr.into(),
                ptr.into(),
            ],
            false,
        );
        let decode_step = self
            .module
            .add_function("decode_step", decode_step_ty, None);

        for f in [prefill, decode_step] {
            let bb = self.ctx.append_basic_block(f, "entry");
            let builder = self.ctx.create_builder();
            builder.position_at_end(bb);
            builder.build_return(None).unwrap();
        }

        (prefill, decode_step)
    }

    /// Run LLVM's module verifier; `Err` carries the verifier's diagnostic.
    pub fn verify(&self) -> Result<()> {
        self.module
            .verify()
            .map_err(|e| crate::CodegenError::Llvm(e.to_string()))
    }

    pub fn print_to_string(&self) -> String {
        self.module.print_to_string().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;

    #[test]
    fn scaffold_verifies() {
        let ctx = Context::create();
        let m = LlvmModule::new(&ctx, "tiny");
        m.declare_kernels();
        let (_prefill, _decode) = m.define_entry_points();
        m.verify().unwrap();
        let ir = m.print_to_string();
        assert!(ir.contains("define"));
        assert!(ir.contains("declare") && ir.contains("inferno_gemv_"));
    }
}
