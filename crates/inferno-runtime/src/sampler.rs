//! Sampling. M1 ships greedy only; the trait is the M4 extension point
//! (temperature/top-k/top-p slot in without touching the generation loop).

pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> u32;
}

pub struct Greedy;

impl Sampler for Greedy {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, v) in logits.iter().enumerate() {
            if *v > logits[best] {
                best = i; // strict > keeps the lowest index on ties
            }
        }
        best as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_argmax_lowest_index_tie_break() {
        let mut g = Greedy;
        assert_eq!(g.sample(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(g.sample(&[0.5, 0.9, 0.9]), 1); // tie → lowest index
        assert_eq!(g.sample(&[f32::NEG_INFINITY, -1.0]), 1);
    }
}
