use inferno_graph::{Graph, Op};
use std::ops::Range;

/// Fusion-island kind (spec §Fusion islands): the group of ops a single
/// codegen'd function body covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IslandKind {
    /// The leading token embedding lookup.
    Embed,
    /// Pre-attention norm + qkv projections + rope, up to (not including) the
    /// `Attention` node.
    AttnProj,
    /// A single `Attention` node, always its own island (spec: attention is
    /// never fused with neighbors).
    Attention,
    /// Post-attention o-proj plus residual add, ffn norm plus gate/up/down
    /// plus swiglu plus residual add: everything from after one `Attention`
    /// up to and including the residual add that closes the FFN block.
    Ffn,
    /// The trailing output norm + logits projection.
    Logits,
}

/// A contiguous run of graph nodes fused into one codegen unit.
#[derive(Debug, Clone, PartialEq)]
pub struct Island {
    pub kind: IslandKind,
    pub nodes: Range<usize>,
    pub label: String,
}

fn label_for(kind: IslandKind, nodes: &Range<usize>) -> String {
    match kind {
        IslandKind::Attention => format!("attention@{}", nodes.start),
        _ => format!("{kind:?}").to_lowercase(),
    }
}

fn push_island(out: &mut Vec<Island>, kind: IslandKind, nodes: Range<usize>) {
    if nodes.is_empty() {
        return;
    }
    let label = label_for(kind, &nodes);
    out.push(Island { kind, nodes, label });
}

/// Single-forward-pass fusion-island partitioner (spec §Fusion islands): no
/// cost model, no lookahead — boundaries are keyed purely on the `Op`
/// variant of the node being visited.
///
/// Rules:
/// - The leading `Embed` node is its own `Embed` island.
/// - Every `Attention` node is its own `Attention` island.
/// - The run of ops between `Embed`/the previous FFN close and the next
///   `Attention` (norm, qkv projections, rope) is an `AttnProj` island.
/// - The run after an `Attention` up to and including the second `Add` seen
///   since (the o-proj residual add, then the FFN's closing residual add) is
///   an `Ffn` island.
/// - Whatever is left after the last `Ffn` island (the trailing output norm +
///   logits `MatMul`) is the `Logits` island.
pub fn partition(g: &Graph) -> Vec<Island> {
    let n = g.nodes.len();
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }

    let mut start = 0usize;
    let mut kind = IslandKind::AttnProj;
    let mut ffn_adds = 0u32;

    for i in 0..n {
        match &g.nodes[i].op {
            Op::Embed { .. } => {
                push_island(&mut out, kind, start..i);
                push_island(&mut out, IslandKind::Embed, i..i + 1);
                start = i + 1;
                kind = IslandKind::AttnProj;
                ffn_adds = 0;
            }
            Op::Attention { .. } => {
                push_island(&mut out, kind, start..i);
                push_island(&mut out, IslandKind::Attention, i..i + 1);
                start = i + 1;
                kind = IslandKind::Ffn;
                ffn_adds = 0;
            }
            Op::Add if kind == IslandKind::Ffn => {
                ffn_adds += 1;
                if ffn_adds == 2 {
                    push_island(&mut out, IslandKind::Ffn, start..i + 1);
                    start = i + 1;
                    kind = IslandKind::AttnProj;
                    ffn_adds = 0;
                }
            }
            _ => {}
        }
    }

    // Whatever is left after the last Ffn close is the trailing output norm
    // + logits projection (the node producing `g.output`), regardless of the
    // (now stale) `kind` state left over from the loop.
    push_island(&mut out, IslandKind::Logits, start..n);

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferno_formats::load_desc;
    use inferno_graph::build_graph;
    use std::path::Path;

    fn tiny_graph() -> inferno_graph::Graph {
        let desc = load_desc(Path::new("../inferno-formats/tests/fixtures/tiny.gguf")).unwrap();
        build_graph(&desc).unwrap()
    }

    #[test]
    fn islands_cover_all_nodes_in_order() {
        let g = tiny_graph();
        let islands = partition(&g);
        // Contiguous, gapless cover of 0..nodes.len().
        assert_eq!(islands.first().unwrap().nodes.start, 0);
        assert_eq!(islands.last().unwrap().nodes.end, g.nodes.len());
        for w in islands.windows(2) {
            assert_eq!(w[0].nodes.end, w[1].nodes.start);
        }
    }

    #[test]
    fn each_attention_is_its_own_island() {
        let g = tiny_graph();
        let islands = partition(&g);
        let attn = islands
            .iter()
            .filter(|i| i.kind == IslandKind::Attention)
            .count();
        assert_eq!(attn as u64, g.n_layers);
        for isl in islands.iter().filter(|i| i.kind == IslandKind::Attention) {
            assert_eq!(isl.nodes.len(), 1); // exactly the Attention node
        }
    }

    #[test]
    fn first_island_is_embed_last_is_logits() {
        let g = tiny_graph();
        let islands = partition(&g);
        assert_eq!(islands.first().unwrap().kind, IslandKind::Embed);
        assert_eq!(islands.last().unwrap().kind, IslandKind::Logits);
    }
}
