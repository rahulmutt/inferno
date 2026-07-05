use crate::{Result, weights::WeightImageLayout};
use inferno_graph::Graph;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ArenaLayout;

pub fn plan_arena(_g: &Graph, _w: &WeightImageLayout, _max: usize) -> Result<ArenaLayout> {
    Ok(ArenaLayout)
}
