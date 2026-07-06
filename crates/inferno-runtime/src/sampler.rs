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
    #[allow(dead_code)] // consumed in Task 4
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

impl Sampler for ChainSampler {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        let logits = self.penalized(logits);
        if self.cfg.temperature == 0.0 {
            return argmax(&logits);
        }
        // Stochastic stages land in the next commit (top-k/top-p/min-p/
        // temperature/draw); greedy covers everything reachable so far.
        argmax(&logits)
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
}
