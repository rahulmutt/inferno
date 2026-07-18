//! M4b.16: geometry-specialized decode attention, emitted as LLVM IR.
//!
//! `emit_attn_hspan_fn` emits ONE function with the exact 13-arg AttnFn ABI
//! (`inferno_attention_f32_scalar_hspan`'s signature). The geometry
//! parameters (kv_dim, n_heads, n_kv_heads, head_dim) are ACCEPTED AND
//! IGNORED — the baked constants are used instead — so the pool dispatcher
//! calls it exactly like the runtime symbol.
//!
//! Bit-neutrality contract: every float op below copies
//! `attn_core_scalar` (inferno-kernels/src/attention.rs) in order —
//! dot8's 8-lane-partitioned FMA chain + reduce8's fixed tree, the
//! sequential f32::max fold, the block-of-8 expf + reduce8 denominator
//! with scalar tail, and the ascending-t mul_add AV accumulation. expf
//! constants come from `inferno_kernels::expf` (single source). No
//! fast-math flags; `llvm.fma` only (mul_add is a guaranteed fused op —
//! `llvm.fmuladd` may split and is forbidden). All vector memory ops are
//! align-4 (pointers are only f32-aligned).
//!
//! The clamp in expf is emitted as compare+select pairs whose operand
//! order matches `_mm256_max_ps(-88, x)` / `_mm256_min_ps(88, x)` exactly
//! (NaN → x), which also matches scalar `f32::clamp` for every input the
//! kernel contract admits (finite scores).

use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::values::{BasicValue, FloatValue, FunctionValue, IntValue, PointerValue, VectorValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};

use inferno_kernels::expf::{C, LN2_HI, LN2_LO, LOG2E};

pub(crate) struct AttnEmitCtx<'c> {
    ctx: &'c Context,
    b: inkwell::builder::Builder<'c>,
    f32_t: inkwell::types::FloatType<'c>,
    i64_t: inkwell::types::IntType<'c>,
    i32_t: inkwell::types::IntType<'c>,
    ptr_t: inkwell::types::PointerType<'c>,
}

impl<'c> AttnEmitCtx<'c> {
    fn new(ctx: &'c Context) -> Self {
        AttnEmitCtx {
            ctx,
            b: ctx.create_builder(),
            f32_t: ctx.f32_type(),
            i64_t: ctx.i64_type(),
            i32_t: ctx.i32_type(),
            ptr_t: ctx.ptr_type(AddressSpace::default()),
        }
    }
    fn v8_t(&self) -> inkwell::types::VectorType<'c> {
        self.f32_t.vec_type(8)
    }
    fn ci64(&self, v: u64) -> IntValue<'c> {
        self.i64_t.const_int(v, false)
    }
    fn cf32(&self, v: f32) -> FloatValue<'c> {
        self.f32_t.const_float(v as f64) // f32→f64 is exact
    }
    /// f32-element pointer offset via ptrtoint/add/inttoptr (the ops.rs
    /// byte_ptr pattern — codegen avoids GEP).
    fn fptr(&self, base: PointerValue<'c>, elem_off: IntValue<'c>) -> PointerValue<'c> {
        let bytes = self
            .b
            .build_int_mul(elem_off, self.ci64(4), "boff")
            .unwrap();
        let bi = self.b.build_ptr_to_int(base, self.i64_t, "p2i").unwrap();
        let sum = self.b.build_int_add(bi, bytes, "paddr").unwrap();
        self.b.build_int_to_ptr(sum, self.ptr_t, "i2p").unwrap()
    }
    fn load_f32(&self, p: PointerValue<'c>) -> FloatValue<'c> {
        self.b
            .build_load(self.f32_t, p, "ld")
            .unwrap()
            .into_float_value()
    }
    fn store_f32(&self, p: PointerValue<'c>, v: FloatValue<'c>) {
        self.b.build_store(p, v).unwrap();
    }
    /// align-4 <8 x float> load (unaligned-safe: vmovups, never vmovaps).
    fn load_v8(&self, p: PointerValue<'c>) -> VectorValue<'c> {
        let ld = self.b.build_load(self.v8_t(), p, "ldv").unwrap();
        ld.as_instruction_value().unwrap().set_alignment(4).unwrap();
        ld.into_vector_value()
    }
    fn store_v8(&self, p: PointerValue<'c>, v: VectorValue<'c>) {
        let st = self.b.build_store(p, v).unwrap();
        st.set_alignment(4).unwrap();
    }
    fn splat(&self, v: FloatValue<'c>) -> VectorValue<'c> {
        let undef = self.v8_t().get_undef();
        let ins = self
            .b
            .build_insert_element(undef, v, self.i32_t.const_zero(), "ins")
            .unwrap();
        let zeros = inkwell::types::VectorType::const_vector(&[self.i32_t.const_zero(); 8]);
        self.b
            .build_shuffle_vector(ins, undef, zeros, "splat")
            .unwrap()
    }
    fn splat_c(&self, v: f32) -> VectorValue<'c> {
        inkwell::types::VectorType::const_vector(&[self.cf32(v); 8])
    }
    fn fma_v8(
        &self,
        m: &Module<'c>,
        a: VectorValue<'c>,
        b: VectorValue<'c>,
        c: VectorValue<'c>,
    ) -> VectorValue<'c> {
        let fma = inkwell::intrinsics::Intrinsic::find("llvm.fma")
            .unwrap()
            .get_declaration(m, &[self.v8_t().into()])
            .unwrap();
        self.b
            .build_call(fma, &[a.into(), b.into(), c.into()], "fma")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_vector_value()
    }
    fn fma_f32(
        &self,
        m: &Module<'c>,
        a: FloatValue<'c>,
        b: FloatValue<'c>,
        c: FloatValue<'c>,
    ) -> FloatValue<'c> {
        let fma = inkwell::intrinsics::Intrinsic::find("llvm.fma")
            .unwrap()
            .get_declaration(m, &[self.f32_t.into()])
            .unwrap();
        self.b
            .build_call(fma, &[a.into(), b.into(), c.into()], "fma")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_float_value()
    }
    /// reduce8: (0+4)(1+5)(2+6)(3+7) then pairwise — kernels' reduce8 tree.
    fn reduce8(&self, v: VectorValue<'c>) -> FloatValue<'c> {
        let mask = |idx: &[u64]| {
            inkwell::types::VectorType::const_vector(
                &idx.iter()
                    .map(|&i| self.i32_t.const_int(i, false))
                    .collect::<Vec<_>>(),
            )
        };
        let undef = self.v8_t().get_undef();
        let lo = self
            .b
            .build_shuffle_vector(v, undef, mask(&[0, 1, 2, 3]), "lo")
            .unwrap();
        let hi = self
            .b
            .build_shuffle_vector(v, undef, mask(&[4, 5, 6, 7]), "hi")
            .unwrap();
        let a = self.b.build_float_add(lo, hi, "a").unwrap();
        let u4 = self.f32_t.vec_type(4).get_undef();
        let a01 = self
            .b
            .build_shuffle_vector(a, u4, mask(&[0, 1]), "a01")
            .unwrap();
        let a23 = self
            .b
            .build_shuffle_vector(a, u4, mask(&[2, 3]), "a23")
            .unwrap();
        let bb = self.b.build_float_add(a01, a23, "b").unwrap();
        let b0 = self
            .b
            .build_extract_element(bb, self.i32_t.const_zero(), "b0")
            .unwrap()
            .into_float_value();
        let b1 = self
            .b
            .build_extract_element(bb, self.i32_t.const_int(1, false), "b1")
            .unwrap()
            .into_float_value();
        self.b.build_float_add(b0, b1, "s").unwrap()
    }
    /// A while-style IR loop: body(i) for i in [start, end) step `step`.
    fn loop_range(
        &self,
        f: FunctionValue<'c>,
        start: IntValue<'c>,
        end: IntValue<'c>,
        step: u64,
        name: &str,
        body: impl FnOnce(&Self, IntValue<'c>),
    ) {
        let header = self.ctx.append_basic_block(f, &format!("{name}.h"));
        let bodyb = self.ctx.append_basic_block(f, &format!("{name}.b"));
        let exit = self.ctx.append_basic_block(f, &format!("{name}.x"));
        let iv = self
            .b
            .build_alloca(self.i64_t, &format!("{name}.i"))
            .unwrap();
        self.b.build_store(iv, start).unwrap();
        self.b.build_unconditional_branch(header).unwrap();
        self.b.position_at_end(header);
        let i = self
            .b
            .build_load(self.i64_t, iv, "i")
            .unwrap()
            .into_int_value();
        let cont = self
            .b
            .build_int_compare(IntPredicate::ULT, i, end, "lt")
            .unwrap();
        self.b.build_conditional_branch(cont, bodyb, exit).unwrap();
        self.b.position_at_end(bodyb);
        body(self, i);
        let i2 = self
            .b
            .build_load(self.i64_t, iv, "i")
            .unwrap()
            .into_int_value();
        let nx = self.b.build_int_add(i2, self.ci64(step), "nx").unwrap();
        self.b.build_store(iv, nx).unwrap();
        self.b.build_unconditional_branch(header).unwrap();
        self.b.position_at_end(exit);
    }
}

impl<'c> AttnEmitCtx<'c> {
    /// Emit expf on a <8 x float>: constants and FMA order verbatim from
    /// inferno_kernels::expf (expf_avx2, bit-identical to expf_scalar).
    fn expf_v8(&self, m: &Module<'c>, x: VectorValue<'c>) -> VectorValue<'c> {
        // clamp: select order matches _mm256_max_ps(-88, x) / _mm256_min_ps(88, x).
        let lo = self.splat_c(-88.0);
        let gt = self
            .b
            .build_float_compare(FloatPredicate::OGT, lo, x, "gt")
            .unwrap();
        let x = self
            .b
            .build_select(gt, lo, x, "cl")
            .unwrap()
            .into_vector_value();
        let hi = self.splat_c(88.0);
        let lt = self
            .b
            .build_float_compare(FloatPredicate::OLT, hi, x, "lt")
            .unwrap();
        let x = self
            .b
            .build_select(lt, hi, x, "ch")
            .unwrap()
            .into_vector_value();
        // n = roundeven(x * LOG2E)
        let xl = self
            .b
            .build_float_mul(x, self.splat_c(LOG2E), "xl")
            .unwrap();
        let re = inkwell::intrinsics::Intrinsic::find("llvm.roundeven")
            .unwrap()
            .get_declaration(m, &[self.v8_t().into()])
            .unwrap();
        let n = self
            .b
            .build_call(re, &[xl.into()], "n")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_vector_value();
        // r = fma(n, -LN2_LO, fma(n, -LN2_HI, x))
        let r = self.fma_v8(m, n, self.splat_c(-LN2_HI), x);
        let r = self.fma_v8(m, n, self.splat_c(-LN2_LO), r);
        // Horner C6..C0
        let mut p = self.splat_c(C[6]);
        for k in (0..6).rev() {
            p = self.fma_v8(m, p, r, self.splat_c(C[k]));
        }
        // pow2n = bitcast((fptosi(n) + 127) << 23)
        let i32v8 = self.i32_t.vec_type(8);
        let ni = self.b.build_float_to_signed_int(n, i32v8, "ni").unwrap();
        let c127 = inkwell::types::VectorType::const_vector(&[self.i32_t.const_int(127, false); 8]);
        let c23 = inkwell::types::VectorType::const_vector(&[self.i32_t.const_int(23, false); 8]);
        let add = self.b.build_int_add(ni, c127, "e").unwrap();
        let shl = self.b.build_left_shift(add, c23, "bits").unwrap();
        let pf = self
            .b
            .build_bit_cast(shl, self.v8_t(), "p2")
            .unwrap()
            .into_vector_value();
        self.b.build_float_mul(p, pf, "exp").unwrap()
    }

    /// Scalar expf, same constants/order (for the softmax tail).
    fn expf_f32(&self, m: &Module<'c>, x: FloatValue<'c>) -> FloatValue<'c> {
        let lo = self.cf32(-88.0);
        let gt = self
            .b
            .build_float_compare(FloatPredicate::OGT, lo, x, "gt")
            .unwrap();
        let x = self
            .b
            .build_select(gt, lo, x, "cl")
            .unwrap()
            .into_float_value();
        let hi = self.cf32(88.0);
        let lt = self
            .b
            .build_float_compare(FloatPredicate::OLT, hi, x, "lt")
            .unwrap();
        let x = self
            .b
            .build_select(lt, hi, x, "ch")
            .unwrap()
            .into_float_value();
        let xl = self.b.build_float_mul(x, self.cf32(LOG2E), "xl").unwrap();
        let re = inkwell::intrinsics::Intrinsic::find("llvm.roundeven")
            .unwrap()
            .get_declaration(m, &[self.f32_t.into()])
            .unwrap();
        let n = self
            .b
            .build_call(re, &[xl.into()], "n")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_float_value();
        let r = self.fma_f32(m, n, self.cf32(-LN2_HI), x);
        let r = self.fma_f32(m, n, self.cf32(-LN2_LO), r);
        let mut p = self.cf32(C[6]);
        for k in (0..6).rev() {
            p = self.fma_f32(m, p, r, self.cf32(C[k]));
        }
        let ni = self
            .b
            .build_float_to_signed_int(n, self.i32_t, "ni")
            .unwrap();
        let add = self
            .b
            .build_int_add(ni, self.i32_t.const_int(127, false), "e")
            .unwrap();
        let shl = self
            .b
            .build_left_shift(add, self.i32_t.const_int(23, false), "bits")
            .unwrap();
        let pf = self
            .b
            .build_bit_cast(shl, self.f32_t, "p2")
            .unwrap()
            .into_float_value();
        self.b.build_float_mul(p, pf, "exp").unwrap()
    }
}

/// Emit the geometry-specialized hspan attention function. `head_dim` must
/// be a multiple of 8 (existing kernel contract); `n_heads % n_kv_heads == 0`.
pub(crate) fn emit_attn_hspan_fn<'c>(
    ctx: &'c Context,
    module: &Module<'c>,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    name: &str,
    linkage: Linkage,
) -> FunctionValue<'c> {
    assert!(
        head_dim.is_multiple_of(8),
        "head_dim must be a multiple of 8"
    );
    assert!(n_heads.is_multiple_of(n_kv_heads), "GQA group must divide");
    let e = AttnEmitCtx::new(ctx);
    let group = (n_heads / n_kv_heads) as u64;
    let kv_dim = (n_kv_heads * head_dim) as u64;
    let chunks = head_dim / 8;
    // Baked exactly as the kernel computes it at runtime (deterministic).
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let fn_ty = ctx.void_type().fn_type(
        &[
            e.ptr_t.into(),
            e.ptr_t.into(),
            e.ptr_t.into(),
            e.ptr_t.into(), // out q kv scores
            e.i64_t.into(),
            e.i64_t.into(),
            e.i64_t.into(), // kv_base v_off pos
            e.i64_t.into(),
            e.i64_t.into(),
            e.i64_t.into(),
            e.i64_t.into(), // (ignored geometry)
            e.i64_t.into(),
            e.i64_t.into(), // h_start h_end
        ],
        false,
    );
    let f = module.add_function(name, fn_ty, Some(linkage));
    let entry = ctx.append_basic_block(f, "entry");
    e.b.position_at_end(entry);

    let out = f.get_nth_param(0).unwrap().into_pointer_value();
    let q = f.get_nth_param(1).unwrap().into_pointer_value();
    let kv = f.get_nth_param(2).unwrap().into_pointer_value();
    let scores = f.get_nth_param(3).unwrap().into_pointer_value();
    let kv_base = f.get_nth_param(4).unwrap().into_int_value();
    let v_off = f.get_nth_param(5).unwrap().into_int_value();
    let pos = f.get_nth_param(6).unwrap().into_int_value();
    // params 7..=10 (kv_dim, n_heads, n_kv_heads, head_dim): IGNORED — baked.
    let h_start = f.get_nth_param(11).unwrap().into_int_value();
    let h_end = f.get_nth_param(12).unwrap().into_int_value();

    let visible = e.b.build_int_add(pos, e.ci64(1), "visible").unwrap();
    let kreg = kv_base;
    let vreg = e.b.build_int_add(kv_base, v_off, "vreg").unwrap();

    // Entry allocas (hoisted out of loops so mem2reg promotes them).
    let maxa = e.b.build_alloca(e.f32_t, "max").unwrap();
    let dena = e.b.build_alloca(e.f32_t, "denom").unwrap();
    let acc =
        e.b.build_alloca(e.v8_t().array_type(chunks as u32), "avacc")
            .unwrap();

    e.loop_range(f, h_start, h_end, 1, "head", |e, h| {
        let g = e.b.build_int_unsigned_div(h, e.ci64(group), "g").unwrap();
        // NOTE: `q`/`out` here are the RAW ABI pointers exactly as passed to
        // `inferno_attention_f32_scalar_hspan` (unshifted) — not the
        // span-local slices `attn_core_scalar` indexes internally via `hl =
        // h - h_start`. The public wrapper pre-advances its `q`/`out`
        // pointers by `h_start*head_dim` before calling the h_start-relative
        // core (`q.add(h_start * head_dim)`), so from the RAW pointer's
        // perspective the net element offset for head `h` is
        // `h_start*head_dim + hl*head_dim == h*head_dim`. Addressing by
        // `hl*head_dim` directly against the raw pointer (as an earlier
        // draft did) reproduces the wrong head whenever `h_start != 0` —
        // caught by a partial-span spot check, not the brief's own
        // full-span-only test case.
        let hoff =
            e.b.build_int_mul(h, e.ci64(head_dim as u64), "hoff")
                .unwrap();
        let qh = e.fptr(q, hoff);
        let ghd =
            e.b.build_int_mul(g, e.ci64(head_dim as u64), "ghd")
                .unwrap();

        // -- scores[t] = reduce8(Σ_c fma(q8, k8)) * scale, t ascending --
        e.loop_range(f, e.ci64(0), visible, 1, "sc", |e, t| {
            let tkv = e.b.build_int_mul(t, e.ci64(kv_dim), "tkv").unwrap();
            let kb =
                e.b.build_int_add(e.b.build_int_add(kreg, tkv, "kb0").unwrap(), ghd, "kb")
                    .unwrap();
            let mut a8 = e.v8_t().const_zero();
            for c in 0..chunks {
                let q8 = e.load_v8(e.fptr(qh, e.ci64((c * 8) as u64)));
                let koff = e.b.build_int_add(kb, e.ci64((c * 8) as u64), "ko").unwrap();
                let k8 = e.load_v8(e.fptr(kv, koff));
                a8 = e.fma_v8(module, q8, k8, a8);
            }
            let dot = e.reduce8(a8);
            let s = e.b.build_float_mul(dot, e.cf32(scale), "s").unwrap();
            e.store_f32(e.fptr(scores, t), s);
        });

        // -- max: sequential f32::max fold from NEG_INFINITY --
        e.store_f32(maxa, e.cf32(f32::NEG_INFINITY));
        e.loop_range(f, e.ci64(0), visible, 1, "mx", |e, t| {
            let m0 = e.load_f32(maxa);
            let s = e.load_f32(e.fptr(scores, t));
            let mn = inkwell::intrinsics::Intrinsic::find("llvm.maxnum")
                .unwrap()
                .get_declaration(module, &[e.f32_t.into()])
                .unwrap();
            let m1 =
                e.b.build_call(mn, &[m0.into(), s.into()], "m")
                    .unwrap()
                    .try_as_basic_value()
                    .basic()
                    .unwrap()
                    .into_float_value();
            e.store_f32(maxa, m1);
        });
        let maxv = e.load_f32(maxa);
        let max8 = e.splat(maxv);

        // -- exp + denom: blocks of 8 with reduce8, then scalar tail --
        e.store_f32(dena, e.cf32(0.0));
        let blocks = e.b.build_and(visible, e.ci64(!7u64), "blk").unwrap(); // visible & !7
        e.loop_range(f, e.ci64(0), blocks, 8, "ex8", |e, t| {
            let sp = e.fptr(scores, t);
            let v = e.load_v8(sp);
            let xm = e.b.build_float_sub(v, max8, "xm").unwrap();
            let ev = e.expf_v8(module, xm);
            e.store_v8(sp, ev);
            let d0 = e.load_f32(dena);
            let d1 = e.b.build_float_add(d0, e.reduce8(ev), "d").unwrap();
            e.store_f32(dena, d1);
        });
        e.loop_range(f, blocks, visible, 1, "ext", |e, t| {
            let sp = e.fptr(scores, t);
            let xm = e.b.build_float_sub(e.load_f32(sp), maxv, "xm").unwrap();
            let ev = e.expf_f32(module, xm);
            e.store_f32(sp, ev);
            let d0 = e.load_f32(dena);
            let d1 = e.b.build_float_add(d0, ev, "d").unwrap();
            e.store_f32(dena, d1);
        });
        let denom = e.load_f32(dena);

        // -- AV: acc chunks zeroed; t ascending: acc_c = fma(splat(w/denom), v8, acc_c) --
        for c in 0..chunks {
            let slot = e.fptr(acc, e.ci64((c * 8) as u64));
            e.store_v8(slot, e.v8_t().const_zero());
        }
        e.loop_range(f, e.ci64(0), visible, 1, "av", |e, t| {
            let w = e.load_f32(e.fptr(scores, t));
            let wn = e.b.build_float_div(w, denom, "wn").unwrap();
            let w8 = e.splat(wn);
            let tkv = e.b.build_int_mul(t, e.ci64(kv_dim), "tkv").unwrap();
            let vb =
                e.b.build_int_add(e.b.build_int_add(vreg, tkv, "vb0").unwrap(), ghd, "vb")
                    .unwrap();
            for c in 0..chunks {
                let slot = e.fptr(acc, e.ci64((c * 8) as u64));
                let v8 = e.load_v8(e.fptr(
                    kv,
                    e.b.build_int_add(vb, e.ci64((c * 8) as u64), "vo").unwrap(),
                ));
                let a0 = e.load_v8(slot);
                let a1 = e.fma_v8(module, w8, v8, a0);
                e.store_v8(slot, a1);
            }
        });
        // store the accumulated row to out[h*head_dim ..] (raw ABI pointer;
        // same `hoff` computed above for `qh`)
        let oh = e.fptr(out, hoff);
        for c in 0..chunks {
            let slot = e.fptr(acc, e.ci64((c * 8) as u64));
            e.store_v8(e.fptr(oh, e.ci64((c * 8) as u64)), e.load_v8(slot));
        }
    });
    e.b.build_return(None).unwrap();
    f
}
