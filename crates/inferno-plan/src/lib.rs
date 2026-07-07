//! Target-aware planning: Graph + ModelDesc + TargetDesc -> Plan (pure data).
//! No LLVM. See docs/superpowers/specs/2026-07-05-m3-compiler-design.md.

pub mod error;
pub mod island;
pub mod kv;
pub mod memory;
pub mod plan;
pub mod weights;

pub use error::{PlanError, Result};
pub use plan::Plan;

use inferno_formats::ModelDesc;
use inferno_graph::Graph;
use inferno_target::TargetDesc;

/// Build the full plan. Assembled in Task 5 once the pieces exist.
pub fn plan(
    desc: &ModelDesc,
    graph: &Graph,
    target: &TargetDesc,
    max_seq_len: usize,
    prefill_tile: usize,
) -> Result<Plan> {
    let islands = island::partition(graph);
    let weights = weights::build_weight_image(desc, graph, target)?;
    let arena = memory::plan_arena(graph, &weights, max_seq_len, prefill_tile)?;
    let kv = kv::plan_kv(graph, max_seq_len)?;
    Ok(Plan {
        islands,
        weights,
        arena,
        kv,
        max_seq_len,
    })
}
