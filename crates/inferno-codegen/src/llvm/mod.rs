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

mod ops;
pub use ops::build_full_module;

use crate::Result;
use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::values::{FunctionValue, PointerValue};

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
        // void inferno_gemm_<d>_rs8_<isa>(ptr y, ptr xq, ptr w, i64 k, i64 m,
        //                                 i64 rows, i64 row_start, i64 row_end)
        // — the batched sibling (M4b.2): two extra leading dims (`m`, `rows`)
        // over the gemv ABI so one call fills an `m`-token output panel.
        let gemm_ty = void.fn_type(
            &[
                ptr.into(),
                ptr.into(),
                ptr.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
            ],
            false,
        );
        for d in ["f32", "q8_0", "q4_k"] {
            for isa in ["scalar", "avx2"] {
                for kind in ["gemv", "gemm"] {
                    let ty = if kind == "gemv" { gemv_ty } else { gemm_ty };
                    self.module.add_function(
                        &format!("inferno_{kind}_{d}_rs8_{isa}"),
                        ty,
                        Some(Linkage::External),
                    );
                }
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

        // void inferno_attention_f32_<isa>(ptr out, ptr q, ptr kv, ptr scores,
        //   i64 kv_base, i64 v_off, i64 pos, i64 kv_dim,
        //   i64 n_heads, i64 n_kv_heads, i64 head_dim)
        let attn_ty = void.fn_type(
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
            ],
            false,
        );
        for isa in ["scalar", "avx2"] {
            self.module.add_function(
                &format!("inferno_attention_f32_{isa}"),
                attn_ty,
                Some(Linkage::External),
            );
        }

        // void inferno_par_gemv(ptr kernel, ptr y, ptr xq, ptr w, i64 k, i64 rows)
        // — the M4b.1 host dispatcher; the kernel chosen by `gemv_symbol` is
        // passed as a function pointer, so the per-(dtype, isa) selection
        // logic is unchanged.
        let par_gemv_ty = void.fn_type(
            &[
                ptr.into(),
                ptr.into(),
                ptr.into(),
                ptr.into(),
                i64_t.into(),
                i64_t.into(),
            ],
            false,
        );
        self.module
            .add_function("inferno_par_gemv", par_gemv_ty, Some(Linkage::External));

        // void inferno_par_gemm(ptr kernel, ptr y, ptr xq, ptr w, i64 k, i64 m, i64 rows)
        // — the M4b.2 batched-prefill dispatcher; the gemm kernel chosen by
        // `gemm_symbol` is passed as a function pointer, so the per-(dtype,
        // isa) selection logic is unchanged.
        let par_gemm_ty = void.fn_type(
            &[
                ptr.into(),
                ptr.into(),
                ptr.into(),
                ptr.into(),
                i64_t.into(),
                i64_t.into(),
                i64_t.into(),
            ],
            false,
        );
        self.module
            .add_function("inferno_par_gemm", par_gemm_ty, Some(Linkage::External));
    }

    /// Emit the profiler counter global `inferno_prof_counters : [n x i64]`
    /// (zero-initialized, external linkage so the host resolves it after
    /// `dlopen`). No-op when `n == 0`. Returns the global's pointer.
    pub(crate) fn declare_prof_counters(&self, n: usize) -> Option<PointerValue<'c>> {
        if n == 0 {
            return None;
        }
        let i64_t = self.ctx.i64_type();
        let arr = i64_t.array_type(n as u32);
        let g = self
            .module
            .add_global(arr, Some(AddressSpace::default()), "inferno_prof_counters");
        g.set_linkage(Linkage::External);
        g.set_initializer(&arr.const_zero());
        Some(g.as_pointer_value())
    }

    /// Declare the two generated entry points (signatures only, *no* body).
    /// Task 9's `build_full_module` fills the bodies with real op lowering;
    /// [`define_entry_points`](Self::define_entry_points) is the empty-body
    /// variant kept for the scaffold test. Signatures must not change — later
    /// tasks depend on the exact parameter order/types.
    pub(crate) fn declare_entry_points(&self) -> (FunctionValue<'c>, FunctionValue<'c>) {
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

        (prefill, decode_step)
    }

    /// Define the two generated entry points with empty (`ret void`) bodies.
    /// Only used by the scaffold test; real lowering goes through
    /// [`build_full_module`](crate::llvm::build_full_module).
    pub fn define_entry_points(&self) -> (FunctionValue<'c>, FunctionValue<'c>) {
        let (prefill, decode_step) = self.declare_entry_points();
        for f in [prefill, decode_step] {
            let bb = self.ctx.append_basic_block(f, "entry");
            let builder = self.ctx.create_builder();
            builder.position_at_end(bb);
            builder.build_return(None).unwrap();
        }
        (prefill, decode_step)
    }

    /// Access to the wrapped module (for op lowering in [`ops`]).
    pub(crate) fn module(&self) -> &Module<'c> {
        &self.module
    }

    /// Public accessor to the raw inkwell [`Module`], for object emission
    /// (Task 11's `TargetMachine::write_to_file`).
    pub fn raw_module(&self) -> &Module<'c> {
        &self.module
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
    fn lowered_module_verifies_on_tiny() {
        use inferno_formats::load_desc;
        use inferno_graph::build_graph;
        use inferno_target::TargetDesc;
        use std::path::Path;

        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let graph = build_graph(&desc).unwrap();
        let target = TargetDesc::detect().unwrap();
        let plan = inferno_plan::plan(&desc, &graph, &target, 64, 64).unwrap();

        let ctx = Context::create();
        let module = super::build_full_module(
            &ctx,
            &plan,
            &graph,
            &desc,
            &crate::CompileOptions::default(),
            &crate::profile::ProfileSlots::default(),
        )
        .unwrap();
        // Correctness (numeric) is Task 12; this catches malformed IR early:
        // bad pointer arithmetic, type mismatches, missing terminators.
        assert!(
            module.verify().is_ok(),
            "module failed verification:\n{}",
            module.print_to_string()
        );
    }

    #[test]
    fn profiled_module_verifies_and_exports_counters() {
        use inferno_formats::load_desc;
        use inferno_graph::build_graph;
        use inferno_target::TargetDesc;
        use std::path::Path;
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let graph = build_graph(&desc).unwrap();
        let target = TargetDesc::detect().unwrap();
        let plan = inferno_plan::plan(&desc, &graph, &target, 64, 64).unwrap();
        let lir = crate::loopir::build_loopir(&plan, &graph, &desc);
        let slots = crate::profile::assign_slots(&lir, &plan, &desc);
        let opts = crate::CompileOptions {
            profile: true,
            prefill_tile: 64,
        };
        let ctx = Context::create();
        let m = super::build_full_module(&ctx, &plan, &graph, &desc, &opts, &slots).unwrap();
        assert!(m.verify().is_ok(), "{}", m.print_to_string());
        let ir = m.print_to_string();
        assert!(ir.contains("inferno_prof_counters"));
        assert!(ir.contains("readcyclecounter"));
    }

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
        assert!(ir.contains("inferno_gemm_"));
        assert!(ir.contains("inferno_par_gemv"));
        assert!(ir.contains("inferno_par_gemm"));
        assert!(ir.contains("inferno_attention_f32_scalar"));
        assert!(ir.contains("inferno_attention_f32_avx2"));
    }
}
