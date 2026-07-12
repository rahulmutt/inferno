//! Per-op lowering: LoopIr `Step`s -> LLVM IR, appended into the two entry
//! points (`prefill`, `decode_step`).
//!
//! Every op mirrors [`inferno_graph::ops`] (the scalar oracle) exactly —
//! operation order, eps placement, rope pairing, sigmoid form, attention
//! scale/softmax/GQA — so Task 12's differential sees matching logits. All ops
//! are lowered here: `embed`, `rmsnorm`, `rope`, `swiglu`, `add`, the MatMul
//! kernel calls (`quantize?`/`gemv`/`bias?`), and causal GQA `attention`.
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
/// two entry-point signatures, and real op lowering for every op (the
/// arithmetic ops, MatMul kernel calls, and attention). The result `verify()`s.
pub fn build_full_module<'c>(
    ctx: &'c Context,
    plan: &Plan,
    graph: &Graph,
    desc: &ModelDesc,
    opts: &crate::CompileOptions,
    slots: &crate::profile::ProfileSlots,
) -> Result<LlvmModule<'c>> {
    let lm = LlvmModule::new(ctx, "model");
    lm.declare_kernels();
    let prof = if opts.profile {
        lm.declare_prof_counters(slots.len())
    } else {
        None
    };
    let (prefill, decode) = lm.declare_entry_points();

    let mut cg = Codegen::new(ctx, lm.module(), plan, graph, desc);
    cg.prof_counters = prof;
    if opts.profile {
        cg.prof_slots = slots.index_map();
    }
    let loopir = build_loopir(plan, graph, desc);
    cg.lower_prefill(prefill, &loopir);
    cg.lower_decode(decode, &loopir);

    Ok(lm)
}

/// Per-row runtime context threaded through op lowering.
struct Frame<'c> {
    /// The `arena` base pointer (entry-point param).
    arena: PointerValue<'c>,
    /// The packed-weight image base pointer (entry-point param). GEMV weight
    /// pointers are `weights + PackedWeight.offset` (a byte offset).
    weights: PointerValue<'c>,
    /// The KV-cache base pointer (entry-point param). f32 K/V storage.
    kv: PointerValue<'c>,
    /// Arena row index for this token (`r` in prefill, `0` in decode).
    row: IntValue<'c>,
    /// Absolute position of this token (`pos_off + r` / the `pos` param), i64.
    pos: IntValue<'c>,
    /// Token id for this row (used by `embed`), zero-extended to i64.
    token: IntValue<'c>,
}

/// Prefill entry-point pointers + `pos_off`, bundled once so a tile's per-row
/// [`Frame`]s are cheap to materialize (`env.frame(cg, row)`). Unlike a
/// [`Frame`] it is row-agnostic — it holds the loop-invariant parameters only.
struct TileEnv<'c> {
    /// The `tokens` param (`ptr` to i32 token ids).
    tokens: PointerValue<'c>,
    /// The `pos_off` param (absolute position of arena row 0), i64.
    pos_off: IntValue<'c>,
    /// The packed-weight image base pointer.
    weights: PointerValue<'c>,
    /// The KV-cache base pointer.
    kv: PointerValue<'c>,
    /// The activation-arena base pointer.
    arena: PointerValue<'c>,
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
    /// Monotonic counter naming outlined `tok_body.*` functions uniquely.
    outlined: RefCell<usize>,

    /// Profiler counter array base (`[N x i64]`), or None when not profiling.
    prof_counters: Option<PointerValue<'c>>,
    /// step-label -> slot index (empty when not profiling).
    prof_slots: HashMap<String, usize>,
    /// `llvm.readcyclecounter` declaration (always declared; only *called*
    /// when `prof_counters` is Some).
    readcyc_fn: FunctionValue<'c>,
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
            outlined: RefCell::new(0),
            readcyc_fn: Intrinsic::find("llvm.readcyclecounter")
                .expect("readcyclecounter intrinsic")
                .get_declaration(module, &[])
                .expect("readcyclecounter declaration"),
            prof_counters: None,
            prof_slots: HashMap::new(),
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

    /// Byte address `base + byte_off` (a raw byte offset, *not* scaled by 4).
    /// Used for the packed-weight image and the act-scratch region, whose
    /// offsets are already in bytes.
    fn byte_ptr(&self, base: PointerValue<'c>, byte_off: IntValue<'c>) -> PointerValue<'c> {
        let b = self
            .builder
            .build_ptr_to_int(base, self.i64_t, "p2i")
            .unwrap();
        let addr = self.builder.build_int_add(b, byte_off, "baddr").unwrap();
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

    // ---- profiler helpers ---------------------------------------------------

    fn load_i64(&self, ptr: PointerValue<'c>) -> IntValue<'c> {
        self.builder
            .build_load(self.i64_t, ptr, "ld64")
            .unwrap()
            .into_int_value()
    }

    /// Read the CPU cycle counter (`llvm.readcyclecounter`, an i64).
    fn readcyc(&self) -> IntValue<'c> {
        self.builder
            .build_call(self.readcyc_fn, &[], "rdtsc")
            .unwrap()
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value()
    }

    /// Run `emit` (one op's lowering) bracketed by cycle-counter reads that
    /// accumulate `t1 - t0` into this op's profiler slot. When not profiling
    /// (`prof_counters` is None) it just runs `emit` — zero overhead, and the
    /// emitted math is byte-for-byte identical (readcyclecounter is pure and
    /// only reads a clock). The accumulation runs on the entry-point thread
    /// (GEMV/GEMM shards join before the wrapper returns), so the plain
    /// load/add/store needs no atomics; profiled artifacts are driven by the
    /// single-threaded CLI generate loop.
    fn profiled(&self, label: &str, emit: impl FnOnce(&Self)) {
        let Some(base) = self.prof_counters else {
            return emit(self);
        };
        let slot = self.prof_slots[label];
        let t0 = self.readcyc();
        emit(self);
        let t1 = self.readcyc();
        let delta = self.builder.build_int_sub(t1, t0, "cyc").unwrap();
        let p = self.byte_ptr(base, self.const_i64((slot * 8) as u64));
        let cur = self.load_i64(p);
        let next = self.builder.build_int_add(cur, delta, "acc64").unwrap();
        self.builder.build_store(p, next).unwrap();
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

    /// Pointer to element 0 of value `v`'s row for this frame (the row start).
    fn arena_row_ptr(&self, frame: &Frame<'c>, v: usize) -> PointerValue<'c> {
        let base = self.row_base(frame, v);
        self.arena_ptr(frame, base, self.i64_t.const_zero())
    }

    /// Pointer to the quantized-activation scratch region: `arena` advanced by
    /// `act_scratch_off` *bytes* (the offset is already a byte offset from the
    /// f32 arena base, so it is added directly, not scaled by 4).
    fn act_scratch_ptr(&self, frame: &Frame<'c>) -> PointerValue<'c> {
        let off = self.const_i64(self.plan.arena.act_scratch_off as u64);
        self.byte_ptr(frame.arena, off)
    }

    /// Pointer to element 0 of value `v`'s row `row` in the given `arena`
    /// (f32). Like [`arena_row_ptr`](Self::arena_row_ptr) but takes an explicit
    /// arena pointer + row value rather than a [`Frame`] — the batched GEMM
    /// panel path needs to address rows `tile_start..tile_start+m` directly.
    fn arena_row_ptr_at(
        &self,
        arena: PointerValue<'c>,
        v: usize,
        row: IntValue<'c>,
    ) -> PointerValue<'c> {
        let off = self.const_i64(self.plan.arena.slots[v - 1].offset as u64);
        let rl = self.const_i64(self.row_len(v));
        let ro = self.builder.build_int_mul(row, rl, "rowoff").unwrap();
        let base = self.builder.build_int_add(off, ro, "rowbase").unwrap();
        self.elem_ptr(arena, base)
    }

    /// Base pointer of the quantized-activation scratch region for the given
    /// `arena` (row 0 of the T-row GEMM panel). Same byte offset as
    /// [`act_scratch_ptr`](Self::act_scratch_ptr) without a [`Frame`].
    fn act_scratch_ptr_row0(&self, arena: PointerValue<'c>) -> PointerValue<'c> {
        let off = self.const_i64(self.plan.arena.act_scratch_off as u64);
        self.byte_ptr(arena, off)
    }

    /// Packed-activation byte length for a MatMul weight's stored dtype
    /// (mirrors the kernel's internal `act_len(k)` and the planner's
    /// `packed_act_bytes`). This is THIS matmul's own panel stride — the GEMM
    /// kernel recomputes the identical value, so the panel must be packed at
    /// exactly this stride (see [`lower_gemm`](Self::lower_gemm)).
    fn packed_act_bytes(dtype: &inferno_formats::DType, k: usize) -> u64 {
        use inferno_formats::DType::*;
        (match dtype {
            F32 => k * 4,
            Q8_0 => inferno_kernels::act::q8a_len(k),
            Q4_K => inferno_kernels::act::q8k_len(k),
            other => unreachable!("non-matmul dtype {other:?} reached packed_act_bytes"),
        }) as u64
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

    /// Emit `for tile_start in (0..count).step_by(step) { body(self,
    /// tile_start, m) }` where `m = min(step, count - tile_start)` is the live
    /// row count for this tile. A strided variant of [`range_loop`]; `body`
    /// may itself emit blocks and must leave the builder on a terminator-free
    /// block, which we close with the back-edge.
    fn tile_loop(
        &self,
        count: IntValue<'c>,
        step: IntValue<'c>,
        body: impl FnOnce(&Self, IntValue<'c>, IntValue<'c>),
    ) {
        let func = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();
        let header = self.ctx.append_basic_block(func, "tile.header");
        let body_bb = self.ctx.append_basic_block(func, "tile.body");
        let exit = self.ctx.append_basic_block(func, "tile.exit");

        let idx = self.entry_alloca(self.i64_t, "ts");
        self.builder
            .build_store(idx, self.i64_t.const_zero())
            .unwrap();
        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(header);
        let ts = self
            .builder
            .build_load(self.i64_t, idx, "ts.load")
            .unwrap()
            .into_int_value();
        let cond = self
            .builder
            .build_int_compare(IntPredicate::ULT, ts, count, "ts.lt")
            .unwrap();
        self.builder
            .build_conditional_branch(cond, body_bb, exit)
            .unwrap();

        self.builder.position_at_end(body_bb);
        // m = min(step, count - ts): the tail tile is short when count % step != 0.
        let rem = self.builder.build_int_sub(count, ts, "rem").unwrap();
        let short = self
            .builder
            .build_int_compare(IntPredicate::ULT, rem, step, "m.short")
            .unwrap();
        let m = self
            .builder
            .build_select(short, rem, step, "m")
            .unwrap()
            .into_int_value();
        body(self, ts, m);
        let next = self.builder.build_int_add(ts, step, "ts.next").unwrap();
        self.builder.build_store(idx, next).unwrap();
        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(exit);
    }

    /// Outline `per_token` into a private `void tok_body.<label>.<n>(ptr
    /// ctx, i64 t0, i64 t1)` function and emit ONE
    /// `inferno_par_token_loop(body, ctx, m)` dispatch sharding the tile's
    /// `m` tokens across pool lanes (M4b.9). The ctx pack is 6 i64 words on
    /// the caller's stack — ptrtoint(tokens), pos_off, ptrtoint(weights),
    /// ptrtoint(kv), ptrtoint(arena), tile_start — rebuilt per dispatch;
    /// the pool treats ctx as opaque, only this emitter knows the layout.
    ///
    /// `per_token(cg, env, ti, row)` is emitted POSITIONED INSIDE the
    /// outlined function, once, looped over the span: `ti` is the
    /// tile-local token index, `row = tile_start + ti` the arena row. It
    /// must derive every runtime value from the rebuilt `env`/`ti`/`row`,
    /// module-level globals/functions, or constants — referencing an SSA
    /// value from the calling function is malformed IR (module
    /// verification fails). Never call `profiled` inside `per_token`: the
    /// counter accumulation is non-atomic and belongs to the dispatcher
    /// thread; brackets wrap the dispatch call in the caller instead.
    ///
    /// Bit-neutrality: each token's writes are disjoint rows produced by
    /// the identical loop-body IR the serial `range_loop(m)` emitted, and
    /// exactly one lane runs each token — thread count and shard layout
    /// cannot change output bits (M4b.8 argument, verbatim).
    fn par_token_loop(
        &self,
        env: &TileEnv<'c>,
        tile_start: IntValue<'c>,
        m: IntValue<'c>,
        label: &str,
        per_token: impl FnOnce(&Self, &TileEnv<'c>, IntValue<'c>, IntValue<'c>),
    ) {
        // Caller side: pack ctx on the stack (6 stores per dispatch).
        let ctx = self.entry_alloca(self.i64_t.array_type(6), "tokctx");
        let p2i =
            |p: PointerValue<'c>| self.builder.build_ptr_to_int(p, self.i64_t, "p2i").unwrap();
        let fields = [
            p2i(env.tokens),
            env.pos_off,
            p2i(env.weights),
            p2i(env.kv),
            p2i(env.arena),
            tile_start,
        ];
        for (i, v) in fields.iter().enumerate() {
            let slot = self.byte_ptr(ctx, self.const_i64((i * 8) as u64));
            self.builder.build_store(slot, *v).unwrap();
        }

        // Emit the outlined body function.
        let n = {
            let mut c = self.outlined.borrow_mut();
            *c += 1;
            *c
        };
        let sanitized: String = label
            .chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
            .collect();
        let fn_ty = self.ctx.void_type().fn_type(
            &[self.ptr_t.into(), self.i64_t.into(), self.i64_t.into()],
            false,
        );
        let f = self.module.add_function(
            &format!("tok_body.{sanitized}.{n}"),
            fn_ty,
            Some(Linkage::Private),
        );
        let saved = self.builder.get_insert_block().unwrap();
        let entry = self.ctx.append_basic_block(f, "entry");
        self.builder.position_at_end(entry);

        let ctx_p = f.get_nth_param(0).unwrap().into_pointer_value();
        let t0 = f.get_nth_param(1).unwrap().into_int_value();
        let t1 = f.get_nth_param(2).unwrap().into_int_value();
        let field = |i: usize| {
            let p = self.byte_ptr(ctx_p, self.const_i64((i * 8) as u64));
            self.load_i64(p)
        };
        let i2p = |v: IntValue<'c>| self.builder.build_int_to_ptr(v, self.ptr_t, "i2p").unwrap();
        let body_env = TileEnv {
            tokens: i2p(field(0)),
            pos_off: field(1),
            weights: i2p(field(2)),
            kv: i2p(field(3)),
            arena: i2p(field(4)),
        };
        let ts = field(5);
        let span = self.builder.build_int_sub(t1, t0, "span").unwrap();
        self.range_loop(span, |cg, i| {
            let ti = cg.add(i, t0);
            let row = cg.add(ts, ti);
            per_token(cg, &body_env, ti, row);
        });
        self.builder.build_return(None).unwrap();
        self.builder.position_at_end(saved);

        // Caller side: one dispatch over the tile.
        let pfn = self
            .module
            .get_function("inferno_par_token_loop")
            .expect("par token-loop dispatcher declared");
        self.builder
            .build_call(
                pfn,
                &[
                    f.as_global_value().as_pointer_value().into(),
                    ctx.into(),
                    m.into(),
                ],
                "par_token_loop",
            )
            .unwrap();
    }

    // ---- entry points -------------------------------------------------------

    fn lower_prefill(&self, func: FunctionValue<'c>, loopir: &LoopIr) {
        let entry = self.ctx.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        let tokens = func.get_nth_param(0).unwrap().into_pointer_value();
        let n = func.get_nth_param(1).unwrap().into_int_value();
        let pos_off = func.get_nth_param(2).unwrap().into_int_value();
        let weights = func.get_nth_param(3).unwrap().into_pointer_value();
        let kv = func.get_nth_param(4).unwrap().into_pointer_value();
        let arena = func.get_nth_param(5).unwrap().into_pointer_value();
        let logits_out = func.get_nth_param(6).unwrap().into_pointer_value();
        let t = self.const_i64(self.plan.prefill_tile as u64);

        // Process tokens in tiles of `T = prefill_tile`: for each tile of
        // `m <= T` rows, run the forward pass once — one batched GEMM per
        // matmul (each weight strip read once for the whole tile) with every
        // other op looped over the tile's `m` rows. Output is bitwise-invariant
        // to `T` because each output row is computed independently.
        let env = TileEnv {
            tokens,
            pos_off,
            weights,
            kv,
            arena,
        };
        self.tile_loop(n, t, |cg, tile_start, m| {
            cg.lower_tile(loopir, &env, tile_start, m);
        });

        // Copy the LAST token's logits row (row n-1) of the graph output slot
        // into the caller's `logits_out` buffer. The forward pass leaves logits
        // in the arena slot; the entry-point contract is to surface the final
        // token's row. `n >= 1` for any real prefill call.
        let last = self
            .builder
            .build_int_sub(n, self.const_i64(1), "last")
            .unwrap();
        self.emit_logits_copy(arena, logits_out, last);

        self.builder.build_return(None).unwrap();
    }

    /// Copy `row` of the graph output slot (`[Seq, vocab]`) into `logits_out`
    /// (`vocab` f32). Emitted after the forward pass so the caller sees logits
    /// in the parameter buffer rather than an internal arena slot.
    fn emit_logits_copy(
        &self,
        arena: PointerValue<'c>,
        logits_out: PointerValue<'c>,
        row: IntValue<'c>,
    ) {
        let out_v = self.graph.output;
        let vocab = self.row_len(out_v);
        let off = self.const_i64(self.plan.arena.slots[out_v - 1].offset as u64);
        let rl = self.const_i64(vocab);
        let row_off = self.builder.build_int_mul(row, rl, "logrow").unwrap();
        let base = self.builder.build_int_add(off, row_off, "logbase").unwrap();
        self.range_loop(self.const_i64(vocab), |cg, i| {
            let src = cg.elem_ptr(arena, cg.add(base, i));
            let v = cg.load_f32(src);
            let dst = cg.elem_ptr(logits_out, i);
            cg.store_f32(dst, v);
        });
    }

    fn lower_decode(&self, func: FunctionValue<'c>, loopir: &LoopIr) {
        let entry = self.ctx.append_basic_block(func, "entry");
        self.builder.position_at_end(entry);

        let tok = func.get_nth_param(0).unwrap().into_int_value();
        let pos = func.get_nth_param(1).unwrap().into_int_value();
        let weights = func.get_nth_param(2).unwrap().into_pointer_value();
        let kv = func.get_nth_param(3).unwrap().into_pointer_value();
        let arena = func.get_nth_param(4).unwrap().into_pointer_value();
        let logits_out = func.get_nth_param(5).unwrap().into_pointer_value();
        let token = self
            .builder
            .build_int_z_extend(tok, self.i64_t, "tok64")
            .unwrap();
        let frame = Frame {
            arena,
            weights,
            kv,
            row: self.const_i64(0),
            pos,
            token,
        };
        self.lower_body(loopir, &frame);
        // decode_step operates on arena row 0; copy that single row to logits.
        self.emit_logits_copy(arena, logits_out, self.const_i64(0));
        self.builder.build_return(None).unwrap();
    }

    fn lower_body(&self, loopir: &LoopIr, frame: &Frame<'c>) {
        for island in &loopir.islands {
            for step in &island.steps {
                let label = crate::profile::step_label(step, self.plan, self.desc);
                self.profiled(&label, |cg| cg.lower_step(frame, step));
            }
        }
    }

    /// Materialize a per-row [`Frame`] for arena row `row` from the tile's
    /// loop-invariant [`TileEnv`]: `pos = pos_off + row`, `token = tokens[row]`.
    fn tile_frame(&self, env: &TileEnv<'c>, row: IntValue<'c>) -> Frame<'c> {
        let pos = self.builder.build_int_add(env.pos_off, row, "pos").unwrap();
        let tok_ptr = self.elem_ptr(env.tokens, row);
        let tok = self
            .builder
            .build_load(self.i32_t, tok_ptr, "tok")
            .unwrap()
            .into_int_value();
        let token = self
            .builder
            .build_int_z_extend(tok, self.i64_t, "tok64")
            .unwrap();
        Frame {
            arena: env.arena,
            weights: env.weights,
            kv: env.kv,
            row,
            pos,
            token,
        }
    }

    /// One tile's forward pass over rows `tile_start..tile_start+m`: iterate the
    /// program's steps once, batching each matmul into a single `inferno_par_gemm`
    /// and looping every other op over the tile's `m` rows. Step order is
    /// identical to the old per-token body. Attention appends the whole tile's
    /// k/v via one token-loop dispatch, then issues a single parallel read;
    /// both are bit-identical and T-invariant because each token's rows are
    /// written by exactly one lane and a token's causal read never reaches
    /// KV rows past its own position.
    ///
    /// Profiling attributes per op-kind: the matmul is timed once per tile; each
    /// elementwise/attention op is timed once per tile across its whole m-loop.
    fn lower_tile(
        &self,
        loopir: &LoopIr,
        env: &TileEnv<'c>,
        tile_start: IntValue<'c>,
        m: IntValue<'c>,
    ) {
        for island in &loopir.islands {
            for step in &island.steps {
                let label = crate::profile::step_label(step, self.plan, self.desc);
                match step {
                    Step::Gemv { .. } => {
                        self.profiled(&label, |cg| cg.lower_gemm(env, step, tile_start, m));
                    }
                    // Quantization is folded into `lower_gemm` (it happens inline
                    // while filling the activation panel for a quantized-weight
                    // Gemv). `lower_step`'s own `Step::Quantize` arm is a no-op
                    // for the same reason, so emitting a profiled range-loop here
                    // would just be a dead m-loop plus a spurious ~0-cycle
                    // "quantize" profiler accumulation. Skip it entirely.
                    Step::Quantize { .. } => {}
                    Step::Attention { .. } => {
                        self.profiled(crate::profile::KV_APPEND_LABEL, |cg| {
                            cg.lower_tile_kv_append(env, step, tile_start, m)
                        });
                        self.profiled(&label, |cg| {
                            cg.lower_tile_attention(env, step, tile_start, m)
                        });
                    }
                    _ => {
                        self.profiled(&label, |cg| {
                            cg.par_token_loop(env, tile_start, m, &label, |cg, benv, _ti, row| {
                                let frame = cg.tile_frame(benv, row);
                                cg.lower_step(&frame, step);
                            });
                        });
                    }
                }
            }
        }
    }

    /// Batched matmul for a tile of `m` tokens starting at `tile_start`: fill an
    /// `m`-row activation panel, then issue ONE `inferno_par_gemm` (the kernel
    /// reads each weight strip once for the whole tile).
    ///
    /// Panel stride MUST equal the GEMM kernel's internal `act_len(k)` — for a
    /// quantized weight we quantize token `ti`'s source row into
    /// `act_scratch + ti * packed_act_bytes(dtype, k)` (this matmul's OWN
    /// `act_len(k)`, NOT the arena's `max_act_row`), so the panel is tightly
    /// packed for this GEMM. The scratch region is sized
    /// `prefill_tile * max_act_row >= prefill_tile * this_act_row`, so it fits.
    /// For an F32 weight the arena source rows are already a contiguous panel
    /// (stride `k` f32) — no copy. Output is written token-major
    /// (`y[ti*rows + r]`), which coincides with the arena rows of `out` starting
    /// at `tile_start`.
    fn lower_gemm(
        &self,
        env: &TileEnv<'c>,
        step: &Step,
        tile_start: IntValue<'c>,
        m: IntValue<'c>,
    ) {
        let Step::Gemv {
            weight,
            out,
            rows,
            k,
            ..
        } = step
        else {
            unreachable!("lower_gemm called on non-Gemv step")
        };
        let pw = &self.plan.weights.weights[*weight];
        // The activation feeding this matmul is the MatMul node's first input;
        // the node produces value `out`, so it is `graph.nodes[out-1]`.
        let src = self.graph.nodes[*out - 1].inputs[0];
        let k_c = self.const_i64(*k as u64);
        let rows_c = self.const_i64(*rows as u64);
        let gemm_sym = crate::loopir::gemm_symbol(&pw.dtype, pw.isa);

        let panel_ptr = if pw.dtype != inferno_formats::DType::F32 {
            // Quantize each token's source row into scratch[ti * act_len(k)].
            let act_row = Self::packed_act_bytes(&pw.dtype, *k);
            let scratch = self.act_scratch_ptr_row0(env.arena);
            let qsym = Self::quantize_symbol(&pw.dtype, pw.isa);
            let qfn = self
                .module
                .get_function(&qsym)
                .expect("quantize kernel declared (Task 8)");
            self.range_loop(m, |cg, ti| {
                let row = cg.add(tile_start, ti);
                let src_ptr = cg.arena_row_ptr_at(env.arena, src, row);
                let off = cg
                    .builder
                    .build_int_mul(ti, cg.const_i64(act_row), "actoff")
                    .unwrap();
                let dst = cg.byte_ptr(scratch, off);
                cg.builder
                    .build_call(qfn, &[src_ptr.into(), dst.into(), k_c.into()], "quantize")
                    .unwrap();
            });
            scratch
        } else {
            // F32 weight: the source rows are already a contiguous f32 panel.
            self.arena_row_ptr_at(env.arena, src, tile_start)
        };

        let y_ptr = self.arena_row_ptr_at(env.arena, *out, tile_start);
        let w_ptr = self.byte_ptr(env.weights, self.const_i64(pw.offset as u64));
        let gfn = self
            .module
            .get_function(&gemm_sym)
            .expect("gemm kernel declared (Task 5)");
        let pfn = self
            .module
            .get_function("inferno_par_gemm")
            .expect("par gemm dispatcher declared");
        self.builder
            .build_call(
                pfn,
                &[
                    gfn.as_global_value().as_pointer_value().into(),
                    y_ptr.into(),
                    panel_ptr.into(),
                    w_ptr.into(),
                    k_c.into(),
                    m.into(),
                    rows_c.into(),
                ],
                "par_gemm",
            )
            .unwrap();
        // Bias (if any) is a separate Step handled in the elementwise m-loop.
    }

    /// The write half of the tile's attention step (M4b.9): shard the
    /// tile's `m` KV-appends across pool lanes. Bit-safe by the same
    /// argument as the parallel read: each token writes only its own KV
    /// row (`pos0 + t`), rows are disjoint across tokens, and the
    /// dispatch joins before `lower_tile_attention` issues the parallel
    /// read — so every KV row `<= pos_i` is in place when token i's
    /// causal read runs, exactly as with the old serial append loop.
    fn lower_tile_kv_append(
        &self,
        env: &TileEnv<'c>,
        step: &Step,
        tile_start: IntValue<'c>,
        m: IntValue<'c>,
    ) {
        let Step::Attention { k, v, layer, .. } = step else {
            unreachable!("lower_tile_kv_append called on non-Attention step")
        };
        let (k, v, layer) = (*k, *v, *layer);
        self.par_token_loop(
            env,
            tile_start,
            m,
            crate::profile::KV_APPEND_LABEL,
            |cg, benv, _ti, row| {
                let frame = cg.tile_frame(benv, row);
                cg.lower_kv_append(&frame, k, v, layer);
            },
        );
    }

    /// Tiled prefill attention read (M4b.8/M4b.9): the tile's k/v was
    /// appended by `lower_tile_kv_append`'s dispatch (already joined), then
    /// ONE `inferno_par_attention` call shards the tile's `m` tokens across
    /// pool lanes.
    fn lower_tile_attention(
        &self,
        env: &TileEnv<'c>,
        step: &Step,
        tile_start: IntValue<'c>,
        m: IntValue<'c>,
    ) {
        let Step::Attention {
            q,
            layer,
            n_heads,
            n_kv_heads,
            head_dim,
            out,
            ..
        } = step
        else {
            unreachable!("lower_tile_attention called on non-Attention step")
        };
        let kv_dim = self.plan.kv.kv_dim as u64;
        let seq_len = self.plan.max_seq_len as u64;
        let kv_base = *layer as u64 * seq_len * kv_dim * 2;
        let q_ptr = self.arena_row_ptr_at(env.arena, *q, tile_start);
        let out_ptr = self.arena_row_ptr_at(env.arena, *out, tile_start);
        let pos0 = self.add(env.pos_off, tile_start);
        let sym = crate::loopir::attention_symbol(self.module_isa());
        let afn = self
            .module
            .get_function(&sym)
            .expect("attention kernel declared (Task 6)");
        let pfn = self
            .module
            .get_function("inferno_par_attention")
            .expect("par attention dispatcher declared");
        self.builder
            .build_call(
                pfn,
                &[
                    afn.as_global_value().as_pointer_value().into(),
                    out_ptr.into(),
                    q_ptr.into(),
                    env.kv.into(),
                    pos0.into(),
                    m.into(),
                    self.const_i64(kv_base).into(),
                    self.const_i64(seq_len * kv_dim).into(),
                    self.const_i64(kv_dim).into(),
                    self.const_i64(*n_heads as u64).into(),
                    self.const_i64(*n_kv_heads as u64).into(),
                    self.const_i64(*head_dim as u64).into(),
                    self.const_i64(self.row_len(*q)).into(),
                    self.const_i64(self.row_len(*out)).into(),
                ],
                "par_attention",
            )
            .unwrap();
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
            Step::Gemv {
                symbol,
                weight,
                out,
                rows,
                k,
            } => self.lower_gemv(frame, symbol, *weight, *out, *rows, *k),
            Step::Bias {
                bias_tensor,
                out,
                rows,
            } => self.lower_bias(frame, *bias_tensor, *out, *rows),
            Step::Attention {
                q,
                k,
                v,
                layer,
                n_heads,
                n_kv_heads,
                head_dim,
                out,
            } => self.lower_attention(
                frame,
                *q,
                *k,
                *v,
                *layer,
                *n_heads,
                *n_kv_heads,
                *head_dim,
                *out,
            ),
            // The activation-quantize is folded into the `Gemv` anchor (which
            // carries the weight index → dtype/isa needed to pick the kernel).
            Step::Quantize { .. } => {}
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

    /// `inferno_quantize_row_<q>_<isa>`: the activation-quantize kernel for a
    /// quantized weight's *stored* dtype. Weight Q8_0 → activation `q8a`;
    /// weight Q4_K → activation `q8k` (M2 kernel design). Only reached for
    /// quantized weights (F32 skips quantize entirely).
    fn quantize_symbol(dtype: &inferno_formats::DType, isa: inferno_kernels::KernelIsa) -> String {
        let q = match dtype {
            inferno_formats::DType::Q8_0 => "q8a",
            inferno_formats::DType::Q4_K => "q8k",
            other => unreachable!("non-quantized dtype {other:?} reached quantize_symbol"),
        };
        let i = match isa {
            inferno_kernels::KernelIsa::Scalar => "scalar",
            inferno_kernels::KernelIsa::Avx2 => "avx2",
        };
        format!("inferno_quantize_row_{q}_{i}")
    }

    /// Lower one MatMul's `Gemv` (folding in the preceding `Quantize`): compute
    /// `out[0..rows]` for this token via the packed-weight kernel. The
    /// activation source is the MatMul's input-0 value (node `out-1`). For a
    /// quantized weight the source row is quantized into the shared act-scratch
    /// region first; for an F32 (native or widened F16/BF16) weight the raw f32
    /// source row is passed straight through. `w_ptr = weights + offset`,
    /// dispatched through `inferno_par_gemv`, which shards `[0, rows)` across
    /// the host thread pool (serial when the pool is uninitialized).
    fn lower_gemv(
        &self,
        frame: &Frame<'c>,
        symbol: &str,
        weight: usize,
        out: usize,
        rows: usize,
        k: usize,
    ) {
        let pw = &self.plan.weights.weights[weight];
        // The activation feeding this GEMV is the MatMul node's first input;
        // the node produces value `out`, so it is `graph.nodes[out-1]`.
        let src = self.graph.nodes[out - 1].inputs[0];
        let k_c = self.const_i64(k as u64);

        let xq_ptr = if pw.dtype != inferno_formats::DType::F32 {
            // Quantize the f32 source row into the act-scratch region.
            let scratch = self.act_scratch_ptr(frame);
            let src_ptr = self.arena_row_ptr(frame, src);
            let qsym = Self::quantize_symbol(&pw.dtype, pw.isa);
            let qfn = self
                .module
                .get_function(&qsym)
                .expect("quantize kernel declared (Task 8)");
            self.builder
                .build_call(
                    qfn,
                    &[src_ptr.into(), scratch.into(), k_c.into()],
                    "quantize",
                )
                .unwrap();
            scratch
        } else {
            // F32 weight: the raw f32 activation row is the kernel input.
            self.arena_row_ptr(frame, src)
        };

        let w_ptr = self.byte_ptr(frame.weights, self.const_i64(pw.offset as u64));
        let out_ptr = self.arena_row_ptr(frame, out);
        let gfn = self
            .module
            .get_function(symbol)
            .expect("gemv kernel declared (Task 8)");
        let pfn = self
            .module
            .get_function("inferno_par_gemv")
            .expect("par gemv dispatcher declared");
        let rows_c = self.const_i64(rows as u64);
        self.builder
            .build_call(
                pfn,
                &[
                    gfn.as_global_value().as_pointer_value().into(),
                    out_ptr.into(),
                    xq_ptr.into(),
                    w_ptr.into(),
                    k_c.into(),
                    rows_c.into(),
                ],
                "par_gemv",
            )
            .unwrap();
    }

    /// `out[i] += bias[i]` for `i in 0..rows` (oracle `matmul`'s `+ bias[n]`).
    /// The bias tensor is dequantized at compile time into a private global,
    /// same as the norm/embed weights.
    fn lower_bias(&self, frame: &Frame<'c>, bias_tensor: usize, out: usize, rows: usize) {
        let bias = self.weight_global(bias_tensor);
        let out_base = self.row_base(frame, out);
        self.range_loop(self.const_i64(rows as u64), |cg, i| {
            let o_ptr = cg.arena_ptr(frame, out_base, i);
            let cur = cg.load_f32(o_ptr);
            let bv = cg.load_f32(cg.elem_ptr(bias, i));
            let sum = cg.builder.build_float_add(cur, bv, "bias").unwrap();
            cg.store_f32(o_ptr, sum);
        });
    }

    /// Append this token's k/v rows into the f32 KV cache at `frame.pos` —
    /// the write half of `Step::Attention`. Called per token by the decode
    /// path (`lower_attention`) and by the prefill tile arm
    /// (`lower_tile_attention`), which appends the WHOLE tile before its
    /// parallel attention read. That reordering is bit-safe: token i's
    /// causal read never reaches rows past `pos_i`.
    fn lower_kv_append(&self, frame: &Frame<'c>, k: usize, v: usize, layer: usize) {
        let kv_dim = self.plan.kv.kv_dim as u64;
        let seq_len = self.plan.max_seq_len as u64;
        let kv_base = layer as u64 * seq_len * kv_dim * 2;
        let k_region = kv_base;
        let v_region = kv_base + seq_len * kv_dim;
        let kv_dim_c = self.const_i64(kv_dim);
        let pos_kv = self
            .builder
            .build_int_mul(frame.pos, kv_dim_c, "poskv")
            .unwrap();
        let k_row = self.row_base(frame, k);
        let v_row = self.row_base(frame, v);
        let k_dst = self.add(self.const_i64(k_region), pos_kv);
        let v_dst = self.add(self.const_i64(v_region), pos_kv);
        self.range_loop(kv_dim_c, |cg, c| {
            let kval = cg.load_f32(cg.arena_ptr(frame, k_row, c));
            cg.store_f32(cg.elem_ptr(frame.kv, cg.add(k_dst, c)), kval);
            let vval = cg.load_f32(cg.arena_ptr(frame, v_row, c));
            cg.store_f32(cg.elem_ptr(frame.kv, cg.add(v_dst, c)), vval);
        });
    }

    /// Every PackedWeight carries the same target-derived ISA; use it as
    /// the module ISA (attention has no PackedWeight of its own).
    fn module_isa(&self) -> inferno_kernels::KernelIsa {
        self.plan
            .weights
            .weights
            .first()
            .map(|w| w.isa)
            .unwrap_or(inferno_kernels::KernelIsa::Scalar)
    }

    /// Causal GQA attention for the current token, mirroring
    /// `inferno_graph::ops::attention`. First appends this token's k/v vectors
    /// into the f32 KV cache at position `pos`, then reads: for each head `h`
    /// (kv group `g = h / (n_heads/n_kv_heads)`) computes
    /// `scores[t] = dot(q_head, kcache[t,g]) * (1/sqrt(head_dim))` for
    /// `t in 0..=pos`, softmaxes with max-subtraction, and accumulates
    /// `out_head = Σ_t (scores[t]/denom) * vcache[t,g]`.
    #[allow(clippy::too_many_arguments)]
    fn lower_attention(
        &self,
        frame: &Frame<'c>,
        q: usize,
        k: usize,
        v: usize,
        layer: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        out: usize,
    ) {
        let hd = head_dim as u64;
        let kv_dim = self.plan.kv.kv_dim as u64; // == n_kv_heads * head_dim
        let seq_len = self.plan.max_seq_len as u64;
        // Per-layer KV region base (f32 elements); K then V, each [seq_len × kv_dim].
        let kv_base = layer as u64 * seq_len * kv_dim * 2;

        // --- KV append: write this token's k/v vectors at position `pos`. ---
        self.lower_kv_append(frame, k, v, layer);

        // --- Single-token attention read via the kernel (M4b.3). q is this
        // token's query row; k/v for this token were appended above, so the
        // kernel only reads the KV cache. `scores` is per-call scratch. ---
        let scores = self.entry_alloca(self.f32_t.array_type(seq_len as u32), "scores");
        let q_ptr = self.arena_row_ptr(frame, q);
        let out_ptr = self.arena_row_ptr(frame, out);
        let isa = self.module_isa();
        let sym = crate::loopir::attention_symbol(isa);
        let f = self
            .module
            .get_function(&sym)
            .expect("attention kernel declared (Task 6)");
        let kv_dim_c = self.const_i64(kv_dim);
        let v_off = self.const_i64(seq_len * kv_dim);
        let kv_base_c = self.const_i64(kv_base);
        self.builder
            .build_call(
                f,
                &[
                    out_ptr.into(),
                    q_ptr.into(),
                    frame.kv.into(),
                    scores.into(),
                    kv_base_c.into(),
                    v_off.into(),
                    frame.pos.into(),
                    kv_dim_c.into(),
                    self.const_i64(n_heads as u64).into(),
                    self.const_i64(n_kv_heads as u64).into(),
                    self.const_i64(hd).into(),
                ],
                "attention",
            )
            .unwrap();
    }
}
