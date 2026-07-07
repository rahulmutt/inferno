use crate::{Result, weights::WeightImageLayout};
use inferno_graph::{Dim, Graph};

/// One value's placement in the shared f32 activation arena.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueSlot {
    pub value: usize,
    pub offset: usize,
    pub len_elems: usize,
}

/// The full activation-memory layout: a liveness-packed f32 arena plus the
/// scratch region reserved for quantized-activation buffers (rs8 GEMV
/// inputs), placed immediately after the arena.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ArenaLayout {
    /// One slot per graph value (f32 elements), in producer order.
    pub slots: Vec<ValueSlot>,
    /// Arena size in f32 elements, at `max_seq_len`.
    pub total_f32: usize,
    /// Byte offset of the quantized-activation scratch region (after the
    /// f32 arena).
    pub act_scratch_off: usize,
    /// Max packed-activation bytes over all MatMuls.
    pub act_scratch_bytes: usize,
}

/// Value `v` is live from its producer (node `v - 1`) to its last consumer.
/// The graph's output value has no consumer node, so it lives to the end.
fn live_range(graph: &Graph, v: usize) -> (usize, usize) {
    let producer = v - 1;
    let last_use = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.inputs.contains(&v))
        .map(|(i, _)| i)
        .max()
        .unwrap_or(producer);
    (producer, last_use)
}

/// Whether values `a` and `b` are simultaneously live at any node.
pub(crate) fn live_overlaps(graph: &Graph, a: usize, b: usize) -> bool {
    let (a0, a1) = live_range(graph, a);
    let (b0, b1) = live_range(graph, b);
    a0 <= b1 && b0 <= a1
}

/// A value's element count: the product of its `out_shape`, with `Dim::Seq`
/// bound to `max_seq_len` (the prefill footprint; decode's `Seq = 1` always
/// fits within it).
fn value_len(graph: &Graph, v: usize, max_seq_len: usize) -> usize {
    graph.nodes[v - 1]
        .out_shape
        .0
        .iter()
        .map(|d| match d {
            Dim::Const(c) => *c as usize,
            Dim::Seq => max_seq_len,
        })
        .product()
}

/// Packed activation byte length for a MatMul weight's stored dtype (mirrors
/// the kernel's `act_len`). Widened F16/BF16 weights are stored as `F32`
/// (Task 3), so they fall through to the `F32` arm and consume raw f32
/// activations, which is correct.
fn packed_act_bytes(dtype: &inferno_formats::DType, k: usize) -> usize {
    use inferno_formats::DType::*;
    match dtype {
        F32 => k * 4,
        Q8_0 => inferno_kernels::act::q8a_len(k),
        Q4_K => inferno_kernels::act::q8k_len(k),
        _ => 0,
    }
}

/// Plan the shared activation arena: a liveness-packed f32 region (one slot
/// per graph value) followed by a quantized-activation scratch region.
///
/// Placement is a correct first-fit gap scan: for each value `v` (processed
/// in producer order), collect the half-open element intervals of every
/// *already-placed* slot whose live range overlaps `v`'s, sort them by
/// offset, and walk the gaps between them (starting at 0) for the first gap
/// `>= len_elems(v)`. If no such gap exists, place `v` past the highest live
/// interval. This never places two simultaneously-live values at overlapping
/// offsets, because the search only ever avoids *live* intervals, and it
/// still reuses dead values' offsets whenever a gap is free.
pub fn plan_arena(
    graph: &Graph,
    weights: &WeightImageLayout,
    max_seq_len: usize,
    prefill_tile: usize,
) -> Result<ArenaLayout> {
    // The ×T multiply that sizes `act_scratch` for prefill tiling is Task 8;
    // this task only threads the parameter through. Behavior must stay
    // byte-identical (the plan snapshot depends on it).
    let _ = prefill_tile;
    let n = graph.nodes.len();
    let mut slots: Vec<ValueSlot> = Vec::with_capacity(n);
    let mut total_f32 = 0usize;

    for v in 1..=n {
        let len = value_len(graph, v, max_seq_len);

        // Every already-placed slot that is simultaneously live with `v`.
        let mut live: Vec<(usize, usize)> = slots
            .iter()
            .filter(|s| live_overlaps(graph, v, s.value))
            .map(|s| (s.offset, s.offset + s.len_elems))
            .collect();
        live.sort_by_key(|&(off, _)| off);

        // Gap scan: walk the sorted live intervals looking for a gap of at
        // least `len` between the current cursor and the next interval's
        // start. If none fits, place past the last live interval's end.
        let mut cursor = 0usize;
        let mut offset = None;
        for &(start, end) in &live {
            if start >= cursor && start - cursor >= len {
                offset = Some(cursor);
                break;
            }
            cursor = cursor.max(end);
        }
        let off = offset.unwrap_or(cursor);

        total_f32 = total_f32.max(off + len);
        slots.push(ValueSlot {
            value: v,
            offset: off,
            len_elems: len,
        });
    }

    let act_scratch_bytes = weights
        .weights
        .iter()
        .map(|w| packed_act_bytes(&w.dtype, w.k))
        .max()
        .unwrap_or(0);
    let act_scratch_off = total_f32 * 4;

    Ok(ArenaLayout {
        slots,
        total_f32,
        act_scratch_off,
        act_scratch_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::build_weight_image;
    use inferno_formats::load_desc;
    use inferno_graph::build_graph;
    use inferno_target::TargetDesc;
    use std::path::Path;

    fn setup() -> (inferno_graph::Graph, crate::weights::WeightImageLayout) {
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let graph = build_graph(&desc).unwrap();
        let target = TargetDesc::detect().unwrap();
        let weights = build_weight_image(&desc, &graph, &target).unwrap();
        (graph, weights)
    }

    #[test]
    fn every_value_has_a_slot() {
        let (graph, weights) = setup();
        let a = plan_arena(&graph, &weights, 128, 64).unwrap();
        assert_eq!(a.slots.len(), graph.nodes.len());
        for (i, s) in a.slots.iter().enumerate() {
            assert_eq!(s.value, i + 1);
        }
    }

    #[test]
    fn no_two_live_values_overlap() {
        let (graph, weights) = setup();
        let a = plan_arena(&graph, &weights, 128, 64).unwrap();
        // For each pair with overlapping live ranges, byte ranges must be disjoint.
        for (i, si) in a.slots.iter().enumerate() {
            for sj in a.slots.iter().skip(i + 1) {
                if live_overlaps(&graph, si.value, sj.value) {
                    let ri = si.offset..si.offset + si.len_elems;
                    let rj = sj.offset..sj.offset + sj.len_elems;
                    assert!(
                        ri.end <= rj.start || rj.end <= ri.start,
                        "values {} and {} overlap in the arena",
                        si.value,
                        sj.value
                    );
                }
            }
        }
    }

    #[test]
    fn reuse_shrinks_arena_below_bump() {
        // Liveness packing must be <= naive bump (sum of all value sizes).
        let (graph, weights) = setup();
        let a = plan_arena(&graph, &weights, 128, 64).unwrap();
        let bump: usize = a.slots.iter().map(|s| s.len_elems).sum();
        assert!(a.total_f32 <= bump);
    }
}
