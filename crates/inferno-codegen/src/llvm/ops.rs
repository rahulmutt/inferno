//! Per-op lowering: LoopIr `Step`s -> LLVM IR, appended into the two entry
//! points (`prefill`, `decode_step`).
//!
//! Every op mirrors [`inferno_graph::ops`] (the scalar oracle) exactly —
//! operation order, eps placement, rope pairing, sigmoid form — so Task 12's
//! differential sees matching logits. Only `embed`, `rmsnorm`, `rope`,
//! `swiglu`, and `add` are lowered here; `matmul`/`attention` (and their
//! `quantize`/`gemv`/`bias` expansion) are no-op stubs this task (Task 10).
//!
//! # Arena addressing
//! `arena` is an opaque `ptr` to an f32 base. Value `v`'s slot is
//! `plan.arena.slots[v-1]`, whose `offset` is in **f32 elements**. A value
//! shaped `[Seq, ..]` lays out `max_seq_len` rows row-major with
//! `row_len = product of the non-Seq dims`. Row `r`'s element `i` lives at
//! element index `offset + r*row_len + i`, i.e. byte `arena + 4*(that)`.
//! We compute addresses with `ptrtoint`/`inttoptr` (both safe in inkwell,
//! unlike `build_gep`) so the crate stays `unsafe`-free.
//!
//! # prefill vs decode
//! `prefill` wraps the whole per-token forward pass in a loop `r in 0..n`
//! with `pos = pos_off + r` and `token = tokens[r]`, operating on arena row
//! `r`. `decode_step` runs the same body once on arena row `0` with the given
//! `pos`/`token`. The two share every global constant.

use std::cell::RefCell;
use std::collections::HashMap;

use inkwell::AddressSpace;
use inkwell::IntPredicate;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::{Linkage, Module};
use inkwell::types::{FloatType, IntType, PointerType};
use inkwell::values::{FloatValue, FunctionValue, IntValue, PointerValue};

use inferno_formats::{ModelDesc, RopeStyle, quant, read_tensor_bytes};
use inferno_graph::{Dim, Graph};
use inferno_plan::Plan;

use super::LlvmModule;
use crate::Result;
use crate::loopir::{LoopIr, Step, build_loopir};

/// Build the full LLVM module for a planned model: the frozen kernel ABI, the
/// two entry-point signatures, and real op lowering for the five arithmetic
/// ops (matmul/attention stubbed until Task 10). The result `verify()`s.
pub fn build_full_module<'c>(
    ctx: &'c Context,
    plan: &Plan,
    graph: &Graph,
    desc: &ModelDesc,
) -> Result<LlvmModule<'c>> {
    let lm = LlvmModule::new(ctx, "model");
    lm.declare_kernels();
    let (prefill, decode) = lm.declare_entry_points();

    let cg = Codegen::new(ctx, lm.module(), plan, graph, desc);
    let loopir = build_loopir(plan, graph, desc);
    cg.lower_prefill(prefill, &loopir);
    cg.lower_decode(decode, &loopir);

    Ok(lm)
}

/// Per-row runtime context threaded through op lowering.
struct Frame<'c> {
    /// The `arena` base pointer (entry-point param).
    arena: PointerValue<'c>,
    /// Arena row index for this token (`r` in prefill, `0` in decode).
    row: IntValue<'c>,
    /// Absolute position of this token (`pos_off + r` / the `pos` param), i64.
    pos: IntValue<'c>,
    /// Token id for this row (used by `embed`), zero-extended to i64.
    token: IntValue<'c>,
}

/// Holds the builder, cached types/intrinsics, and the compile-time constant
/// globals (dequantized weights, rope frequency tables) that lowering emits.
struct Codegen<'c, 'a> {
    ctx: &'c Context,
    module: &'a Module<'c>,
    builder: Builder<'c>,
    plan: &'a Plan,
    graph: &'a Graph,
    desc: &'a ModelDesc,

    f32_t: FloatType<'c>,
    i32_t: IntType<'c>,
    i64_t: IntType<'c>,
    ptr_t: PointerType<'c>,

    sin_fn: FunctionValue<'c>,
    cos_fn: FunctionValue<'c>,
    exp_fn: FunctionValue<'c>,
    sqrt_fn: FunctionValue<'c>,

    /// tensor_index -> private `[N x float]` global of dequantized weights.
    weight_globals: RefCell<HashMap<usize, PointerValue<'c>>>,
    /// (theta bits, head_dim) -> `[half x float]` rope frequency table.
    rope_freqs: RefCell<HashMap<(u32, u64), PointerValue<'c>>>,
}

impl<'c, 'a> Codegen<'c, 'a> {
    fn new(
        ctx: &'c Context,
        module: &'a Module<'c>,
        plan: &'a Plan,
        graph: &'a Graph,
        desc: &'a ModelDesc,
    ) -> Self {
        let f32_t = ctx.f32_type();
        // Overloaded intrinsic declarations (llvm.sin.f32 etc.), created once.
        let decl = |name: &str| {
            Intrinsic::find(name)
                .expect("known intrinsic")
                .get_declaration(module, &[f32_t.into()])
                .expect("intrinsic declaration")
        };
        Self {
            ctx,
            module,
            builder: ctx.create_builder(),
            plan,
            graph,
            desc,
            f32_t,
            i32_t: ctx.i32_type(),
            i64_t: ctx.i64_type(),
            ptr_t: ctx.ptr_type(AddressSpace::default()),
            sin_fn: decl("llvm.sin"),
            cos_fn: decl("llvm.cos"),
            exp_fn: decl("llvm.exp"),
            sqrt_fn: decl("llvm.sqrt"),
            weight_globals: RefCell::new(HashMap::new()),
            rope_freqs: RefCell::new(HashMap::new()),
        }
    }

    // ---- small constant / value helpers -------------------------------------

    fn const_i64(&self, v: u64) -> IntValue<'c> {
        self.i64_t.const_int(v, false)
    }
    fn const_f32(&self, v: f32) -> FloatValue<'c> {
        self.f32_t.const_float(v as f64)
    }
    fn add(&self, a: IntValue<'c>, b: IntValue<'c>) -> IntValue<'c> {
        self.builder.build_int_add(a, b, "idx").unwrap()
    }

    /// Byte address `base + 4*index`, as a fresh pointer. Works for both f32
    /// and i32/u32 element arrays (both 4-byte). Uses ptrtoint/inttoptr to
    /// avoid inkwell's `unsafe` GEP.
    fn elem_ptr(&self, base: PointerValue<'c>, index: IntValue<'c>) -> PointerValue<'c> {
        let b = self
            .builder
            .build_ptr_to_int(base, self.i64_t, "p2i")
            .unwrap();
        let bytes = self
            .builder
            .build_int_mul(index, self.const_i64(4), "bytes")
            .unwrap();
        let addr = self.builder.build_int_add(b, bytes, "addr").unwrap();
        self.builder
            .build_int_to_ptr(addr, self.ptr_t, "i2p")
            .unwrap()
    }

    fn load_f32(&self, ptr: PointerValue<'c>) -> FloatValue<'c> {
        self.builder
            .build_load(self.f32_t, ptr, "ld")
            .unwrap()
            .into_float_value()
    }
    fn store_f32(&self, ptr: PointerValue<'c>, val: FloatValue<'c>) {
        self.builder.build_store(ptr, val).unwrap();
    }

    /// Build an `alloca` in the current function's **entry block** (before its
    /// first instruction), regardless of where the builder currently sits, then
    /// restore the prior insert position. Hoisting keeps every stack slot
    /// allocated once per call — an alloca left in a loop body (e.g. inside
    /// prefill's per-token loop) would allocate a fresh slot each iteration and
    /// leak until return, growing prefill stack to O(n × allocas/token).
    fn entry_alloca<T: inkwell::types::BasicType<'c>>(
        &self,
        ty: T,
        name: &str,
    ) -> PointerValue<'c> {
        let saved = self.builder.get_insert_block().unwrap();
        let entry = saved.get_parent().unwrap().get_first_basic_block().unwrap();
        match entry.get_first_instruction() {
            Some(first) => self.builder.position_before(&first),
            None => self.builder.position_at_end(entry),
        }
        let slot = self.builder.build_alloca(ty, name).unwrap();
        self.builder.position_at_end(saved);
        slot
    }

    fn call_unary(&self, f: FunctionValue<'c>, x: FloatValue<'c>) -> FloatValue<'c> {
        self.builder
            .build_call(f, &[x.into()], "call")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value()
    }

    // ---- arena / weight addressing ------------------------------------------

    /// Per-row element count: product of the value's non-`Seq` output dims.
    fn row_len(&self, v: usize) -> u64 {
        self.graph.nodes[v - 1]
            .out_shape
            .0
            .iter()
            .filter_map(|d| match d {
                Dim::Const(c) => Some(*c),
                Dim::Seq => None,
            })
            .product()
    }

    /// Element index of value `v`'s row for this frame: `offset + row*row_len`.
    fn row_base(&self, frame: &Frame<'c>, v: usize) -> IntValue<'c> {
        // Invariant: every arena activation routed through here is `[Seq, ..]`,
        // so `row_len` (non-Seq dims) is a genuine per-row stride and `row`
        // (0..max_seq_len) indexes distinct rows. A shape without `Seq` would
        // make `row > 0` alias past the slot.
        debug_assert!(
            self.graph.nodes[v - 1]
                .out_shape
                .0
                .iter()
                .any(|d| matches!(d, Dim::Seq)),
            "value {v} routed through row_base has no Seq dim"
        );
        let off = self.const_i64(self.plan.arena.slots[v - 1].offset as u64);
        let rl = self.const_i64(self.row_len(v));
        let ro = self.builder.build_int_mul(frame.row, rl, "rowoff").unwrap();
        self.builder.build_int_add(off, ro, "rowbase").unwrap()
    }

    /// Pointer to element `base_index + i` of the arena (f32).
    fn arena_ptr(
        &self,
        frame: &Frame<'c>,
        base_index: IntValue<'c>,
        i: IntValue<'c>,
    ) -> PointerValue<'c> {
        let idx = self.add(base_index, i);
        self.elem_ptr(frame.arena, idx)
    }

    /// A private `[N x float]` global holding tensor `tensor_index` dequantized
    /// at compile time with the same `quant::dequant` the interpreter uses —
    /// guaranteeing bit-parity with the oracle. Cached per tensor.
    fn weight_global(&self, tensor_index: usize) -> PointerValue<'c> {
        if let Some(p) = self.weight_globals.borrow().get(&tensor_index) {
            return *p;
        }
        let td = &self.desc.tensors[tensor_index];
        let n: u64 = td.shape.iter().product();
        let bytes = read_tensor_bytes(self.desc, td).expect("weight tensor bytes readable");
        let vals = quant::dequant(&td.dtype, &bytes, n as usize).expect("weight dequant");
        let ptr = self.emit_f32_global(&format!("w.t{tensor_index}"), &vals);
        self.weight_globals.borrow_mut().insert(tensor_index, ptr);
        ptr
    }

    /// A `[half x float]` global of rope frequencies
    /// `freq[i] = theta^(-2i/head_dim)`, computed with `f32::powf` so it is
    /// bit-identical to the oracle's per-`i` `theta.powf(..)`. Cached per
    /// (theta, head_dim).
    fn rope_freq_table(&self, theta: f32, head_dim: u64) -> PointerValue<'c> {
        let key = (theta.to_bits(), head_dim);
        if let Some(p) = self.rope_freqs.borrow().get(&key) {
            return *p;
        }
        let half = (head_dim / 2) as usize;
        let vals: Vec<f32> = (0..half)
            .map(|i| theta.powf(-2.0 * i as f32 / head_dim as f32))
            .collect();
        let ptr = self.emit_f32_global(
            &format!("rope_freq.t{:08x}_{head_dim}", theta.to_bits()),
            &vals,
        );
        self.rope_freqs.borrow_mut().insert(key, ptr);
        ptr
    }

    fn emit_f32_global(&self, name: &str, vals: &[f32]) -> PointerValue<'c> {
        let arr_ty = self.f32_t.array_type(vals.len() as u32);
        let g = self
            .module
            .add_global(arr_ty, Some(AddressSpace::default()), name);
        g.set_constant(true);
        g.set_linkage(Linkage::Private);
        g.set_unnamed_addr(true);
        let consts: Vec<FloatValue> = vals.iter().map(|v| self.const_f32(*v)).collect();
        g.set_initializer(&self.f32_t.const_array(&consts));
        g.as_pointer_value()
    }

    // ---- control flow -------------------------------------------------------

    /// Emit `for i in 0..count { body(self, i) }` with an alloca'd counter.
    /// `body` may itself emit blocks (nested loops); it must leave the builder
    /// on a terminator-free block, which we close with the back-edge.
    fn range_loop(&self, count: IntValue<'c>, body: impl FnOnce(&Self, IntValue<'c>)) {
        let func = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();
        let header = self.ctx.append_basic_block(func, "loop.header");
        let body_bb = self.ctx.append_basic_block(func, "loop.body");
        let exit = self.ctx.append_basic_block(func, "loop.exit");

        let idx = self.entry_alloca(self.i64_t, "i");
        self.builder
            .build_store(idx, self.i64_t.const_zero())
            .unwrap();
        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(header);
        let i = self
            .builder
            .build_load(self.i64_t, idx, "i.load")
            .unwrap()
            .into_int_value();
        let cond = self
            .builder
            .build_int_compare(IntPredicate::ULT, i, count, "i.lt")
            .unwrap();
        self.builder
            .build_conditional_branch(cond, body_bb, exit)
            .unwrap();

        self.builder.position_at_end(body_bb);
        body(self, i);
        let next = self
            .builder
            .build_int_add(i, self.const_i64(1), "i.next")
            .unwrap();
        self.builder.build_store(idx, next).unwrap();
        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(exit);
    }

    // ---- entry points -------------------------------------------------------

    fn lower_prefill(&self, func: FunctionValue<'c>, loopir: &LoopIr) {
        let entry = self.ctx.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        let tokens = func.get_nth_param(0).unwrap().into_pointer_value();
        let n = func.get_nth_param(1).unwrap().into_int_value();
        let pos_off = func.get_nth_param(2).unwrap().into_int_value();
        let arena = func.get_nth_param(5).unwrap().into_pointer_value();

        self.range_loop(n, |cg, r| {
            let pos = cg.builder.build_int_add(pos_off, r, "pos").unwrap();
            let tok_ptr = cg.elem_ptr(tokens, r);
            let tok = cg
                .builder
                .build_load(cg.i32_t, tok_ptr, "tok")
                .unwrap()
                .into_int_value();
            let token = cg
                .builder
                .build_int_z_extend(tok, cg.i64_t, "tok64")
                .unwrap();
            let frame = Frame {
                arena,
                row: r,
                pos,
                token,
            };
            cg.lower_body(loopir, &frame);
        });

        self.builder.build_return(None).unwrap();
    }

    fn lower_decode(&self, func: FunctionValue<'c>, loopir: &LoopIr) {
        let entry = self.ctx.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        let tok = func.get_nth_param(0).unwrap().into_int_value();
        let pos = func.get_nth_param(1).unwrap().into_int_value();
        let arena = func.get_nth_param(4).unwrap().into_pointer_value();
        let token = self
            .builder
            .build_int_z_extend(tok, self.i64_t, "tok64")
            .unwrap();
        let frame = Frame {
            arena,
            row: self.const_i64(0),
            pos,
            token,
        };
        self.lower_body(loopir, &frame);
        self.builder.build_return(None).unwrap();
    }

    fn lower_body(&self, loopir: &LoopIr, frame: &Frame<'c>) {
        for island in &loopir.islands {
            for step in &island.steps {
                self.lower_step(frame, step);
            }
        }
    }

    // ---- per-op lowering ----------------------------------------------------

    fn lower_step(&self, frame: &Frame<'c>, step: &Step) {
        match step {
            Step::Embed { table, out } => self.lower_embed(frame, *table, *out),
            Step::RmsNorm {
                src,
                weight,
                eps,
                out,
                head_dim,
            } => self.lower_rmsnorm(frame, *src, *weight, *eps, *out, *head_dim),
            Step::Rope {
                src,
                out,
                n_heads,
                head_dim,
                theta,
                style,
            } => self.lower_rope(frame, *src, *out, *n_heads, *head_dim, *theta, *style),
            Step::SwiGlu { gate, up, out } => self.lower_swiglu(frame, *gate, *up, *out),
            Step::Add { a, b, out } => self.lower_add(frame, *a, *b, *out),
            // Task 10 fills these; a no-op that never writes its out-slot still
            // verifies (numeric correctness is Task 12's differential).
            Step::Quantize { .. }
            | Step::Gemv { .. }
            | Step::Bias { .. }
            | Step::Attention { .. } => {}
        }
    }

    /// `out[i] = table[token*hidden + i]` for `i in 0..hidden` (oracle
    /// `ops::embed`). The embed table is a compile-time global constant.
    fn lower_embed(&self, frame: &Frame<'c>, table: usize, out: usize) {
        let hidden = self.row_len(out);
        let tbl = self.weight_global(table);
        let out_base = self.row_base(frame, out);
        let tok_base = self
            .builder
            .build_int_mul(frame.token, self.const_i64(hidden), "tokbase")
            .unwrap();
        self.range_loop(self.const_i64(hidden), |cg, i| {
            let src = cg.elem_ptr(tbl, cg.add(tok_base, i));
            let v = cg.load_f32(src);
            let dst = cg.arena_ptr(frame, out_base, i);
            cg.store_f32(dst, v);
        });
    }

    /// `y[i] = x[i] * (1/sqrt(mean(x^2)+eps)) * w[i]` (oracle `ops::rmsnorm`),
    /// per `unit`-sized chunk (`unit = head_dim.unwrap_or(cols)`), weight
    /// cycling over `0..unit`. Mean = sum-of-squares/unit, eps inside the sqrt,
    /// multiply order `(x*inv)*w`.
    #[allow(clippy::too_many_arguments)]
    fn lower_rmsnorm(
        &self,
        frame: &Frame<'c>,
        src: usize,
        weight: usize,
        eps: f32,
        out: usize,
        head_dim: Option<usize>,
    ) {
        let cols = self.row_len(src);
        let unit = head_dim.map(|d| d as u64).unwrap_or(cols);
        let n_chunks = cols / unit;
        let w = self.weight_global(weight);
        let src_base = self.row_base(frame, src);
        let out_base = self.row_base(frame, out);
        let eps_c = self.const_f32(eps);
        let unit_c = self.const_f32(unit as f32);

        for c in 0..n_chunks {
            let chunk = self.const_i64(c * unit);
            let src_chunk = self.add(src_base, chunk);
            let out_chunk = self.add(out_base, chunk);

            // sum-of-squares, accumulated left-to-right (matches oracle .sum()).
            let acc = self.entry_alloca(self.f32_t, "ss");
            self.builder
                .build_store(acc, self.f32_t.const_zero())
                .unwrap();
            self.range_loop(self.const_i64(unit), |cg, j| {
                let x = cg.load_f32(cg.arena_ptr(frame, src_chunk, j));
                let sq = cg.builder.build_float_mul(x, x, "sq").unwrap();
                let cur = cg
                    .builder
                    .build_load(cg.f32_t, acc, "acc")
                    .unwrap()
                    .into_float_value();
                let sum = cg.builder.build_float_add(cur, sq, "sum").unwrap();
                cg.builder.build_store(acc, sum).unwrap();
            });
            let ss = self
                .builder
                .build_load(self.f32_t, acc, "ss.v")
                .unwrap()
                .into_float_value();
            let ms = self.builder.build_float_div(ss, unit_c, "ms").unwrap();
            let arg = self.builder.build_float_add(ms, eps_c, "ms.eps").unwrap();
            let root = self.call_unary(self.sqrt_fn, arg);
            let inv = self
                .builder
                .build_float_div(self.const_f32(1.0), root, "inv")
                .unwrap();

            self.range_loop(self.const_i64(unit), |cg, j| {
                let x = cg.load_f32(cg.arena_ptr(frame, src_chunk, j));
                let wv = cg.load_f32(cg.elem_ptr(w, j));
                let xi = cg.builder.build_float_mul(x, inv, "xinv").unwrap();
                let o = cg.builder.build_float_mul(xi, wv, "y").unwrap();
                cg.store_f32(cg.arena_ptr(frame, out_chunk, j), o);
            });
        }
    }

    /// Rotate each `(head, i<half)` pair by `angle = pos * theta^(-2i/head_dim)`
    /// (oracle `ops::rope`). We first copy `src -> out` (the oracle's `clone`),
    /// then rotate in place: `out[a] = x0*cos - x1*sin`,
    /// `out[b] = x0*sin + x1*cos`, with `(a,b)` per `RopeStyle`.
    #[allow(clippy::too_many_arguments)]
    fn lower_rope(
        &self,
        frame: &Frame<'c>,
        src: usize,
        out: usize,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        style: RopeStyle,
    ) {
        let cols = self.row_len(src);
        let hd = head_dim as u64;
        let half = hd / 2;
        let src_base = self.row_base(frame, src);
        let out_base = self.row_base(frame, out);

        // out = src.clone()
        self.range_loop(self.const_i64(cols), |cg, i| {
            let v = cg.load_f32(cg.arena_ptr(frame, src_base, i));
            cg.store_f32(cg.arena_ptr(frame, out_base, i), v);
        });

        let freq = self.rope_freq_table(theta, hd);
        let pos_f = self
            .builder
            .build_signed_int_to_float(frame.pos, self.f32_t, "posf")
            .unwrap();

        for h in 0..n_heads as u64 {
            let head_base = self.add(out_base, self.const_i64(h * hd));
            self.range_loop(self.const_i64(half), |cg, i| {
                let fq = cg.load_f32(cg.elem_ptr(freq, i));
                let angle = cg.builder.build_float_mul(pos_f, fq, "angle").unwrap();
                let sin = cg.call_unary(cg.sin_fn, angle);
                let cos = cg.call_unary(cg.cos_fn, angle);

                let (a_local, b_local) = match style {
                    RopeStyle::Interleaved => {
                        let two_i = cg.builder.build_int_mul(i, cg.const_i64(2), "2i").unwrap();
                        let a = two_i;
                        let b = cg.add(two_i, cg.const_i64(1));
                        (a, b)
                    }
                    RopeStyle::HalfSplit => {
                        let a = i;
                        let b = cg.add(i, cg.const_i64(half));
                        (a, b)
                    }
                };
                let pa = cg.arena_ptr(frame, head_base, a_local);
                let pb = cg.arena_ptr(frame, head_base, b_local);
                let x0 = cg.load_f32(pa);
                let x1 = cg.load_f32(pb);
                // out[a] = x0*cos - x1*sin
                let ac = cg.builder.build_float_mul(x0, cos, "x0cos").unwrap();
                let as_ = cg.builder.build_float_mul(x1, sin, "x1sin").unwrap();
                let na = cg.builder.build_float_sub(ac, as_, "na").unwrap();
                // out[b] = x0*sin + x1*cos
                let bs = cg.builder.build_float_mul(x0, sin, "x0sin").unwrap();
                let bc = cg.builder.build_float_mul(x1, cos, "x1cos").unwrap();
                let nb = cg.builder.build_float_add(bs, bc, "nb").unwrap();
                cg.store_f32(pa, na);
                cg.store_f32(pb, nb);
            });
        }
    }

    /// `out[i] = (g[i] / (1 + exp(-g[i]))) * u[i]` (oracle `ops::swiglu`).
    fn lower_swiglu(&self, frame: &Frame<'c>, gate: usize, up: usize, out: usize) {
        let cols = self.row_len(gate);
        let gb = self.row_base(frame, gate);
        let ub = self.row_base(frame, up);
        let ob = self.row_base(frame, out);
        self.range_loop(self.const_i64(cols), |cg, i| {
            let g = cg.load_f32(cg.arena_ptr(frame, gb, i));
            let u = cg.load_f32(cg.arena_ptr(frame, ub, i));
            let neg = cg.builder.build_float_neg(g, "neg").unwrap();
            let e = cg.call_unary(cg.exp_fn, neg);
            let denom = cg
                .builder
                .build_float_add(cg.const_f32(1.0), e, "denom")
                .unwrap();
            let silu = cg.builder.build_float_div(g, denom, "silu").unwrap();
            let o = cg.builder.build_float_mul(silu, u, "o").unwrap();
            cg.store_f32(cg.arena_ptr(frame, ob, i), o);
        });
    }

    /// `out[i] = a[i] + b[i]` (oracle `ops::add`).
    fn lower_add(&self, frame: &Frame<'c>, a: usize, b: usize, out: usize) {
        let cols = self.row_len(out);
        let ab = self.row_base(frame, a);
        let bb = self.row_base(frame, b);
        let ob = self.row_base(frame, out);
        self.range_loop(self.const_i64(cols), |cg, i| {
            let x = cg.load_f32(cg.arena_ptr(frame, ab, i));
            let y = cg.load_f32(cg.arena_ptr(frame, bb, i));
            let o = cg.builder.build_float_add(x, y, "sum").unwrap();
            cg.store_f32(cg.arena_ptr(frame, ob, i), o);
        });
    }
}
