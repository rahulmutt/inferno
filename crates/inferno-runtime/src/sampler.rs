//! Sampling. M1 ships greedy only; the trait is the M4 extension point
//! (temperature/top-k/top-p slot in without touching the generation loop).

pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> u32;
    /// Observe a token appended to the sequence (prompt or sampled).
    /// Default no-op; `ChainSampler` uses it for the repeat-penalty window.
    fn accept(&mut self, _token: u32) {}
}

pub struct Greedy;

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i; // strict > keeps the lowest index on ties
        }
    }
    best as u32
}

impl Sampler for Greedy {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        argmax(logits)
    }
}

/// Configuration for [`ChainSampler`]. Defaults are all-neutral: a default
/// config behaves exactly like [`Greedy`] (tested), so `inferno run`'s
/// defaults stay bit-identical to the pre-M4a greedy behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    /// 0.0 = greedy argmax (short-circuits the whole chain).
    pub temperature: f32,
    /// Keep the k highest-logit tokens; 0 = disabled.
    pub top_k: usize,
    /// Keep the smallest prefix of descending-probability tokens with
    /// cumulative mass >= top_p; 1.0 = disabled.
    pub top_p: f32,
    /// Drop tokens with probability < min_p * max_probability; 0.0 = disabled.
    pub min_p: f32,
    /// Divide positive / multiply negative logits of recent tokens; 1.0 = disabled.
    pub repeat_penalty: f32,
    /// How many recent tokens the penalty window holds.
    pub repeat_last_n: usize,
    /// RNG seed for the final draw.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> SamplerConfig {
        SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

impl SamplerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.temperature.is_nan() || self.temperature < 0.0 {
            return Err(format!(
                "temperature must be >= 0, got {}",
                self.temperature
            ));
        }
        if self.top_p.is_nan() || !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(format!("top-p must be in (0, 1], got {}", self.top_p));
        }
        if self.min_p.is_nan() || !(self.min_p >= 0.0 && self.min_p < 1.0) {
            return Err(format!("min-p must be in [0, 1), got {}", self.min_p));
        }
        if self.repeat_penalty.is_nan() || self.repeat_penalty <= 0.0 {
            return Err(format!(
                "repeat-penalty must be > 0, got {}",
                self.repeat_penalty
            ));
        }
        Ok(())
    }
}

/// The M4a sampling chain (spec order): repeat penalty → top-k → top-p →
/// min-p → temperature → seeded draw. `temperature == 0` short-circuits to
/// argmax over the penalized logits.
pub struct ChainSampler {
    cfg: SamplerConfig,
    rng: crate::rng::Xoshiro256StarStar,
    recent: std::collections::VecDeque<u32>,
}

impl ChainSampler {
    pub fn new(cfg: SamplerConfig) -> ChainSampler {
        let rng = crate::rng::Xoshiro256StarStar::new(cfg.seed);
        ChainSampler {
            cfg,
            rng,
            recent: std::collections::VecDeque::new(),
        }
    }

    fn penalized(&self, logits: &[f32]) -> Vec<f32> {
        let mut out = logits.to_vec();
        if self.cfg.repeat_penalty != 1.0 {
            // Penalize once per distinct token in the window.
            let distinct: std::collections::HashSet<u32> = self.recent.iter().copied().collect();
            for t in distinct {
                if let Some(l) = out.get_mut(t as usize) {
                    *l = if *l > 0.0 {
                        *l / self.cfg.repeat_penalty
                    } else {
                        *l * self.cfg.repeat_penalty
                    };
                }
            }
        }
        out
    }
}

/// Numerically stable softmax over `n` values (max-subtracted, f64).
fn softmax(vals: impl Iterator<Item = f64>, n: usize) -> Vec<f64> {
    let vals: Vec<f64> = vals.collect();
    debug_assert_eq!(vals.len(), n);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = vals.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

impl Sampler for ChainSampler {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        let logits = self.penalized(logits);
        if self.cfg.temperature == 0.0 {
            return argmax(&logits);
        }

        // Candidates sorted by (logit desc, index asc) — index tiebreak
        // keeps every downstream truncation deterministic.
        let mut cand: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as u32, l))
            .collect();
        cand.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        // Top-k.
        if self.cfg.top_k > 0 && self.cfg.top_k < cand.len() {
            cand.truncate(self.cfg.top_k);
        }

        // Softmax at temperature 1 (f64, max-subtracted) for the
        // mass-based filters. `cand` is sorted, so probs are descending.
        let mut probs = softmax(cand.iter().map(|&(_, l)| l as f64), cand.len());

        // Top-p: smallest prefix with cumulative mass >= top_p.
        if self.cfg.top_p < 1.0 {
            let mut cum = 0.0;
            let mut keep = cand.len();
            for (i, p) in probs.iter().enumerate() {
                cum += p;
                // Epsilon tolerance for floating-point comparison: handle rounding in cumulative sum
                // when adding probabilities computed from softmax with f32→f64 conversion.
                if cum > self.cfg.top_p as f64 - 1e-6 {
                    keep = i + 1;
                    break;
                }
            }
            cand.truncate(keep);
            probs.truncate(keep);
        }

        // Min-p: drop tokens with prob < min_p * max_prob. probs[0] is the
        // max because the list is sorted. Always keep at least one.
        if self.cfg.min_p > 0.0 {
            let cutoff = self.cfg.min_p as f64 * probs[0];
            let keep = probs.iter().take_while(|&&p| p >= cutoff).count().max(1);
            cand.truncate(keep);
        }

        // Temperature, final softmax over survivors, seeded draw.
        let t = self.cfg.temperature as f64;
        let final_probs = softmax(cand.iter().map(|&(_, l)| l as f64 / t), cand.len());
        let u = self.rng.next_f64();
        let mut cum = 0.0;
        for (&(idx, _), p) in cand.iter().zip(&final_probs) {
            cum += p;
            if u < cum {
                return idx;
            }
        }
        // Float roundoff can leave cum fractionally below 1.0.
        cand.last().expect("candidate list is never empty").0
    }

    fn accept(&mut self, token: u32) {
        if self.cfg.repeat_last_n == 0 {
            return;
        }
        self.recent.push_back(token);
        if self.recent.len() > self.cfg.repeat_last_n {
            self.recent.pop_front();
        }
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

    fn pick(cfg: SamplerConfig, accepted: &[u32], logits: &[f32]) -> u32 {
        let mut s = ChainSampler::new(cfg);
        for &t in accepted {
            s.accept(t);
        }
        s.sample(logits)
    }

    /// Neutral config must behave exactly like `Greedy`, ties included.
    #[test]
    fn neutral_chain_equals_greedy() {
        for logits in [
            vec![0.1, 0.9, 0.3],
            vec![0.5, 0.9, 0.9], // tie → lowest index
            vec![f32::NEG_INFINITY, -1.0],
            vec![-2.0, -1.0, -3.0, -1.0], // negative-only, tie
        ] {
            let want = Greedy.sample(&logits);
            assert_eq!(pick(SamplerConfig::default(), &[], &logits), want);
        }
    }

    /// llama.cpp sign convention: positive logits divided by the penalty,
    /// negative logits multiplied.
    #[test]
    fn repeat_penalty_sign_convention() {
        let cfg = SamplerConfig {
            repeat_penalty: 2.0,
            ..Default::default()
        };
        // 2.0/2 = 1.0 < 1.5 → penalized argmax flips to index 1.
        assert_eq!(pick(cfg.clone(), &[0], &[2.0, 1.5]), 1);
        // -1.0*2 = -2.0 < -1.5 → flips to index 1.
        assert_eq!(pick(cfg.clone(), &[0], &[-1.0, -1.5]), 1);
        // Unpenalized (token 0 never accepted): argmax stays 0.
        assert_eq!(pick(cfg, &[], &[2.0, 1.5]), 0);
    }

    /// Tokens evicted from the `repeat_last_n` window are no longer
    /// penalized; a token is penalized once, not per occurrence.
    #[test]
    fn repeat_window_evicts_oldest() {
        let cfg = SamplerConfig {
            repeat_penalty: 1000.0,
            repeat_last_n: 2,
            ..Default::default()
        };
        // accept 1,2,3 with window 2 → only {2,3} penalized; 1 survives.
        assert_eq!(pick(cfg, &[1, 2, 3], &[0.0, 5.0, 6.0, 7.0]), 1);
    }

    #[test]
    fn validate_rejects_out_of_range() {
        assert!(SamplerConfig::default().validate().is_ok());
        let bad = [
            SamplerConfig {
                temperature: -1.0,
                ..Default::default()
            },
            SamplerConfig {
                temperature: f32::NAN,
                ..Default::default()
            },
            SamplerConfig {
                top_p: 0.0,
                ..Default::default()
            },
            SamplerConfig {
                top_p: 1.5,
                ..Default::default()
            },
            SamplerConfig {
                top_p: f32::NAN,
                ..Default::default()
            },
            SamplerConfig {
                min_p: 1.0,
                ..Default::default()
            },
            SamplerConfig {
                min_p: -0.1,
                ..Default::default()
            },
            SamplerConfig {
                min_p: f32::NAN,
                ..Default::default()
            },
            SamplerConfig {
                repeat_penalty: 0.0,
                ..Default::default()
            },
            SamplerConfig {
                repeat_penalty: f32::NAN,
                ..Default::default()
            },
        ];
        for cfg in bad {
            assert!(cfg.validate().is_err(), "{cfg:?} should be rejected");
        }
    }

    /// Four draws over four equally-likely tokens with seed 42 walk the
    /// pinned uniform sequence 0.0838→0.3789→0.6800→0.9246, so the picks
    /// are exactly 0,1,2,3 (cumulative bins of width 0.25).
    #[test]
    fn seeded_exact_picks_uniform_logits() {
        let cfg = SamplerConfig {
            temperature: 1.0,
            seed: 42,
            ..Default::default()
        };
        let mut s = ChainSampler::new(cfg);
        let logits = [3.0, 3.0, 3.0, 3.0];
        let picks: Vec<u32> = (0..4).map(|_| s.sample(&logits)).collect();
        assert_eq!(picks, vec![0, 1, 2, 3]);
    }

    /// A dominant logit wins under any of the pinned seed-42 draws:
    /// p(idx 2) = e^10 / (3 + e^10) ≈ 0.99986 > 0.9246 (the largest draw).
    #[test]
    fn seeded_exact_pick_dominant_logit() {
        let cfg = SamplerConfig {
            temperature: 1.0,
            seed: 42,
            ..Default::default()
        };
        let mut s = ChainSampler::new(cfg);
        for _ in 0..4 {
            assert_eq!(s.sample(&[0.0, 0.0, 10.0, 0.0]), 2);
        }
    }

    /// top-k = 1 must be greedy regardless of temperature or seed.
    #[test]
    fn top_k_one_is_greedy() {
        for seed in [0, 1, 42, 999] {
            let cfg = SamplerConfig {
                temperature: 2.5,
                top_k: 1,
                seed,
                ..Default::default()
            };
            assert_eq!(pick(cfg, &[], &[0.1, 0.9, 0.3]), 1);
        }
    }

    /// Probabilities 0.5/0.3/0.2 (logits ln p). top_p = 0.5 keeps exactly
    /// the first token (cumulative 0.5 >= 0.5) → always index 0.
    #[test]
    fn top_p_cumulative_mass_edge() {
        let logits = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        for seed in 0..20 {
            let cfg = SamplerConfig {
                temperature: 1.0,
                top_p: 0.5,
                seed,
                ..Default::default()
            };
            assert_eq!(pick(cfg, &[], &logits), 0);
        }
        // top_p = 0.8: cumulative hits 0.8 at the second token → index 2
        // is never drawn.
        for seed in 0..20 {
            let cfg = SamplerConfig {
                temperature: 1.0,
                top_p: 0.8,
                seed,
                ..Default::default()
            };
            assert_ne!(pick(cfg, &[], &logits), 2);
        }
    }

    /// min_p = 0.5 with max prob 0.5 → cutoff 0.25: drops the 0.2 token,
    /// keeps 0.5 and 0.3. min_p = 0.7 → cutoff 0.35: only the max survives.
    #[test]
    fn min_p_relative_cutoff() {
        let logits = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        for seed in 0..20 {
            let cfg = SamplerConfig {
                temperature: 1.0,
                min_p: 0.5,
                seed,
                ..Default::default()
            };
            assert_ne!(pick(cfg, &[], &logits), 2);
            let cfg = SamplerConfig {
                temperature: 1.0,
                min_p: 0.7,
                seed,
                ..Default::default()
            };
            assert_eq!(pick(cfg, &[], &logits), 0);
        }
    }
}
