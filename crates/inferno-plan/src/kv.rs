use crate::Result;
use inferno_graph::Graph;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct KvLayout;

pub fn plan_kv(_g: &Graph, _max: usize) -> Result<KvLayout> {
    Ok(KvLayout)
}
