use crate::Result;
use inferno_formats::ModelDesc;
use inferno_graph::Graph;
use inferno_target::TargetDesc;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct WeightImageLayout;

pub fn build_weight_image(
    _d: &ModelDesc,
    _g: &Graph,
    _t: &TargetDesc,
) -> Result<WeightImageLayout> {
    Ok(WeightImageLayout)
}
