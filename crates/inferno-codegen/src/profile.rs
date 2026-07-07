//! Per-op profiler slot assignment (pure, no LLVM). Each lowered `Step`
//! maps to a stable label; matmuls aggregate across layers by their weight
//! tensor's role (the numeric `blk.N.` segment normalized to `*`), so the
//! slot count is op-kind-sized, not per-layer (spec: op-kind totals).

use std::collections::HashMap;

use inferno_formats::ModelDesc;
use inferno_plan::Plan;

use crate::loopir::{LoopIr, Step};

/// Ordered, de-duplicated profiler slots. `labels[i]` names slot `i`.
#[derive(Debug, Clone, Default)]
pub struct ProfileSlots {
    pub labels: Vec<String>,
    pub(crate) index: HashMap<String, usize>,
}

impl ProfileSlots {
    pub fn slot(&self, label: &str) -> usize {
        self.index[label]
    }
    /// The full label -> slot-index map (used by codegen to look up each
    /// step's counter slot from its `step_label`).
    pub(crate) fn index_map(&self) -> HashMap<String, usize> {
        self.index.clone()
    }
    pub fn len(&self) -> usize {
        self.labels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
    fn intern(&mut self, label: String) {
        if !self.index.contains_key(&label) {
            self.index.insert(label.clone(), self.labels.len());
            self.labels.push(label);
        }
    }
}

/// Normalize a weight tensor name so all layers share one slot: any dotted
/// segment that is a bare integer becomes `*` (e.g. `blk.7.attn_q.weight`
/// -> `blk.*.attn_q.weight`).
fn normalize_weight_name(name: &str) -> String {
    name.split('.')
        .map(|seg| if seg.parse::<u64>().is_ok() { "*" } else { seg })
        .collect::<Vec<_>>()
        .join(".")
}

/// The profiler label for one step. Gemv is keyed by normalized weight name
/// (its matmul "site"); every other op by its kind.
pub fn step_label(step: &Step, plan: &Plan, desc: &ModelDesc) -> String {
    match step {
        Step::Gemv { weight, .. } => {
            let ti = plan.weights.weights[*weight].tensor_index;
            format!("matmul:{}", normalize_weight_name(&desc.tensors[ti].name))
        }
        Step::Quantize { .. } => "quantize".into(),
        Step::Bias { .. } => "bias".into(),
        Step::Embed { .. } => "embed".into(),
        Step::RmsNorm { .. } => "rmsnorm".into(),
        Step::Rope { .. } => "rope".into(),
        Step::SwiGlu { .. } => "swiglu".into(),
        Step::Add { .. } => "add".into(),
        Step::Attention { .. } => "attention".into(),
    }
}

/// Assign a slot to every distinct step label, in first-seen program order.
pub fn assign_slots(loopir: &LoopIr, plan: &Plan, desc: &ModelDesc) -> ProfileSlots {
    let mut slots = ProfileSlots::default();
    for island in &loopir.islands {
        for step in &island.steps {
            slots.intern(step_label(step, plan, desc));
        }
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loopir::build_loopir;
    use inferno_formats::load_desc;
    use inferno_graph::build_graph;
    use inferno_target::TargetDesc;
    use std::path::Path;

    #[test]
    fn slots_aggregate_matmuls_across_layers() {
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        let graph = build_graph(&desc).unwrap();
        let target = TargetDesc::detect().unwrap();
        let plan = inferno_plan::plan(&desc, &graph, &target, 64, 64).unwrap();
        let lir = build_loopir(&plan, &graph, &desc);
        let slots = assign_slots(&lir, &plan, &desc);
        // Two transformer layers, but matmul slots must not double: every
        // label is unique, and no label contains a bare layer index.
        let unique: std::collections::HashSet<_> = slots.labels.iter().collect();
        assert_eq!(
            unique.len(),
            slots.labels.len(),
            "labels: {:?}",
            slots.labels
        );
        assert!(
            slots
                .labels
                .iter()
                .all(|l| !l.contains(".0.") && !l.contains(".1."))
        );
        // Sanity: the elementwise kinds each collapse to one slot.
        for kind in ["rmsnorm", "rope", "swiglu", "add", "attention"] {
            assert_eq!(
                slots.labels.iter().filter(|l| *l == kind).count(),
                1,
                "{kind}"
            );
        }
    }

    #[test]
    fn normalize_strips_layer_index() {
        assert_eq!(
            normalize_weight_name("blk.7.attn_q.weight"),
            "blk.*.attn_q.weight"
        );
        assert_eq!(normalize_weight_name("output.weight"), "output.weight");
    }
}
