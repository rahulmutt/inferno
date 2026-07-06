# M4a — Bench Protocol + Sampling Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full sampling suite (`ChainSampler` behind the existing `Sampler` trait) and a real `inferno bench` that compares inferno against the devenv-pinned llama.cpp via `llama-bench -o json`, per the approved spec `docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md`.

**Architecture:** No new crates. `inferno-runtime` gains a private xoshiro256\*\* RNG module and a `ChainSampler` (stages: repeat penalty → top-k → top-p → min-p → temperature → seeded draw; `temperature = 0` short-circuits to argmax). The CLI gains sampling flags on `run` and a new `bench` subcommand that measures the compiled backend directly (no tokenizer in the timed path) and parses `llama-bench` JSON for the comparison report.

**Tech Stack:** Rust workspace; cargo-nextest via `mise run test`; insta snapshots; assert_cmd for CLI tests; serde/serde_json (already workspace deps — **no new dependencies anywhere in this plan**).

## Global Constraints

- **No new crate dependencies.** The RNG is hand-rolled precisely to avoid one (spec §Scope Decisions).
- Workspace lints: `unsafe_code = "deny"` everywhere touched here; `cargo clippy --workspace --all-targets -- -D warnings` must stay clean (`mise run lint`).
- Tests run via `mise run test` (= `cargo nextest run --workspace --no-tests=pass`). Single-test runs below use `cargo nextest run -p <crate> <filter>`.
- **Never modify** `Greedy`, `cli/src/diff.rs`, `bench_compiled`/`MARGIN`, or any differential/nightly gate. The diff gates stay pinned to greedy by construction.
- `inferno run` defaults must stay bit-identical to today's greedy behavior (all sampling flags neutral by default).
- Perf numbers only from quiet hardware inside `devenv shell` (AGENTS.md); nothing in this plan adds a CI perf gate.
- Commit after every task (lefthook runs fmt-check + gitleaks on commit).

---

### Task 1: xoshiro256\*\* RNG module

**Files:**
- Create: `crates/inferno-runtime/src/rng.rs`
- Modify: `crates/inferno-runtime/src/lib.rs` (add `mod rng;`)

**Interfaces:**
- Consumes: nothing.
- Produces (crate-private, used by Task 3/4's `ChainSampler`):
  - `pub(crate) struct Xoshiro256StarStar` with `pub(crate) fn new(seed: u64) -> Self`, `pub(crate) fn next_u64(&mut self) -> u64`, `pub(crate) fn next_f64(&mut self) -> f64` (uniform in `[0, 1)`).

Reference algorithm: xoshiro256\*\* (Blackman & Vigna, public domain), state seeded from the `u64` seed by splitmix64 — the reference seeding procedure. The exact vectors below were computed from the published reference algorithms; they pin the implementation forever (a wrong constant or shift fails loudly).

- [ ] **Step 1: Write the failing tests**

Create `crates/inferno-runtime/src/rng.rs` with tests only (implementation comes in Step 3 — the module must exist for `mod rng;` to compile, so start with the test scaffold plus `use` lines):

```rust
//! Seedable deterministic RNG for sampling: xoshiro256** seeded via
//! splitmix64 (the reference procedure). Hand-rolled on purpose — `rand`'s
//! `SmallRng` documents that its algorithm may change between crate
//! versions, which would silently break the exact-pick sampler tests.

#[cfg(test)]
mod tests {
    use super::*;

    // Reference vectors computed from the published splitmix64 and
    // xoshiro256** algorithms (Blackman & Vigna). If any of these fail,
    // the implementation is wrong — never update the constants.
    #[test]
    fn splitmix64_reference_vector() {
        let mut state = 0u64;
        assert_eq!(splitmix64(&mut state), 0xE220A8397B1DCDAF);
        assert_eq!(splitmix64(&mut state), 0x6E789E6AA1B965F4);
        assert_eq!(splitmix64(&mut state), 0x06C45D188009454F);
    }

    #[test]
    fn xoshiro_reference_vectors() {
        let mut r = Xoshiro256StarStar::new(0);
        assert_eq!(r.next_u64(), 0x99EC5F36CB75F2B4);
        assert_eq!(r.next_u64(), 0xBF6E1F784956452A);
        assert_eq!(r.next_u64(), 0x1A5F849D4933E6E0);

        let mut r = Xoshiro256StarStar::new(42);
        assert_eq!(r.next_u64(), 0x15780B2E0C2EC716);
        assert_eq!(r.next_u64(), 0x6104D9866D113A7E);
        assert_eq!(r.next_u64(), 0xAE17533239E499A1);
    }

    #[test]
    fn next_f64_is_unit_interval_and_deterministic() {
        let mut r = Xoshiro256StarStar::new(42);
        let want = [
            0.08386297105988216,
            0.3789802506626686,
            0.6800434110281394,
            0.9246929453253876,
        ];
        for w in want {
            let got = r.next_f64();
            assert!((0.0..1.0).contains(&got));
            assert_eq!(got, w);
        }
    }

    #[test]
    fn same_seed_same_stream_different_seed_diverges() {
        let mut a = Xoshiro256StarStar::new(7);
        let mut b = Xoshiro256StarStar::new(7);
        let mut c = Xoshiro256StarStar::new(8);
        let sa: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let sb: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        let sc: Vec<u64> = (0..8).map(|_| c.next_u64()).collect();
        assert_eq!(sa, sb);
        assert_ne!(sa, sc);
    }
}
```

Add `mod rng;` to `crates/inferno-runtime/src/lib.rs` (after `mod error;`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo nextest run -p inferno-runtime rng`
Expected: COMPILE ERROR — `splitmix64` and `Xoshiro256StarStar` not found.

- [ ] **Step 3: Write the implementation**

Above the `tests` module in `rng.rs`:

```rust
/// splitmix64 step: advances `state` and returns the next output.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub(crate) struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    pub(crate) fn new(seed: u64) -> Xoshiro256StarStar {
        let mut state = seed;
        let s = [
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
        ];
        Xoshiro256StarStar { s }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1) from the top 53 bits (exactly representable in f64).
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run -p inferno-runtime rng`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-runtime/src/rng.rs crates/inferno-runtime/src/lib.rs
git commit -m "feat(runtime): xoshiro256** RNG pinned by reference vectors"
```

---

### Task 2: `Sampler::accept` + generator wiring

**Files:**
- Modify: `crates/inferno-runtime/src/sampler.rs` (trait)
- Modify: `crates/inferno-runtime/src/generate.rs:153-204` (`Generator::generate`)

**Interfaces:**
- Consumes: existing `Sampler` trait, `Generator::generate`.
- Produces: `Sampler` gains `fn accept(&mut self, _token: u32) {}` (default no-op). `Generator::generate` now calls `sampler.accept(t)` for every prompt token (before prefill) and for every sampled non-EOS token (right after it is pushed to `out_ids`). `Greedy` is untouched — the default method covers it.

- [ ] **Step 1: Write the failing test**

In `crates/inferno-runtime/src/generate.rs` `tests` module, add:

```rust
/// The generator must feed every prompt token and every sampled token to
/// `Sampler::accept` — that is how repeat penalty (M4a) sees context.
#[test]
fn generator_accepts_prompt_and_sampled_tokens() {
    struct Recording {
        inner: Greedy,
        accepted: Vec<u32>,
    }
    impl crate::sampler::Sampler for Recording {
        fn sample(&mut self, logits: &[f32]) -> u32 {
            self.inner.sample(logits)
        }
        fn accept(&mut self, token: u32) {
            self.accepted.push(token);
        }
    }

    let mut g = Generator::load(&fixture("tiny.gguf"), 64).unwrap();
    let prompt_ids = g.encode("the").unwrap();
    let mut s = Recording { inner: Greedy, accepted: Vec::new() };
    let (out_ids, _) = g
        .generate("the", 4, &mut s, &mut |_| ControlFlow::Continue(()))
        .unwrap();
    let mut want = prompt_ids;
    want.extend_from_slice(&out_ids);
    assert_eq!(s.accepted, want);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo nextest run -p inferno-runtime generator_accepts`
Expected: COMPILE ERROR — trait `Sampler` has no method `accept`.

- [ ] **Step 3: Implement**

In `crates/inferno-runtime/src/sampler.rs`, extend the trait:

```rust
pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> u32;
    /// Observe a token appended to the sequence (prompt or sampled).
    /// Default no-op; `ChainSampler` uses it for the repeat-penalty window.
    fn accept(&mut self, _token: u32) {}
}
```

In `Generator::generate` (`generate.rs`), after the `PromptTooLong` check and before `self.backend.reset()`:

```rust
        for &t in &prompt_ids {
            sampler.accept(t);
        }
```

And inside the decode loop, immediately after `out_ids.push(next);`:

```rust
            sampler.accept(next);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run -p inferno-runtime`
Expected: all PASS (including the existing `on_bytes_break_stops_generation_early`).

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-runtime/src/sampler.rs crates/inferno-runtime/src/generate.rs
git commit -m "feat(runtime): Sampler::accept hook; generator feeds prompt + sampled tokens"
```

---

### Task 3: `SamplerConfig` + validation + `ChainSampler` (penalty + greedy path)

**Files:**
- Modify: `crates/inferno-runtime/src/sampler.rs`
- Modify: `crates/inferno-runtime/src/lib.rs` (exports)

**Interfaces:**
- Consumes: `Xoshiro256StarStar` (Task 1), `Sampler` trait with `accept` (Task 2).
- Produces (public, used by Task 4/5):
  - `pub struct SamplerConfig { pub temperature: f32, pub top_k: usize, pub top_p: f32, pub min_p: f32, pub repeat_penalty: f32, pub repeat_last_n: usize, pub seed: u64 }` with `Default` (0, 0, 1.0, 0.0, 1.0, 64, 0 — all neutral) and `pub fn validate(&self) -> Result<(), String>`.
  - `pub struct ChainSampler` with `pub fn new(cfg: SamplerConfig) -> ChainSampler`, implementing `Sampler`.
  - This task implements only: penalty stage + `temperature == 0` argmax. Stochastic stages land in Task 4 (the `sample` body written here is completed there).

- [ ] **Step 1: Write the failing tests**

Append to the `tests` module in `sampler.rs`:

```rust
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
            vec![0.5, 0.9, 0.9],          // tie → lowest index
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
        let cfg = SamplerConfig { repeat_penalty: 2.0, ..Default::default() };
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
            SamplerConfig { temperature: -1.0, ..Default::default() },
            SamplerConfig { top_p: 0.0, ..Default::default() },
            SamplerConfig { top_p: 1.5, ..Default::default() },
            SamplerConfig { min_p: 1.0, ..Default::default() },
            SamplerConfig { min_p: -0.1, ..Default::default() },
            SamplerConfig { repeat_penalty: 0.0, ..Default::default() },
        ];
        for cfg in bad {
            assert!(cfg.validate().is_err(), "{cfg:?} should be rejected");
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo nextest run -p inferno-runtime sampler`
Expected: COMPILE ERROR — `SamplerConfig`, `ChainSampler` not found.

- [ ] **Step 3: Implement**

In `sampler.rs` (below `Greedy`):

```rust
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
        if !(self.temperature >= 0.0) {
            return Err(format!("temperature must be >= 0, got {}", self.temperature));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(format!("top-p must be in (0, 1], got {}", self.top_p));
        }
        if !(self.min_p >= 0.0 && self.min_p < 1.0) {
            return Err(format!("min-p must be in [0, 1), got {}", self.min_p));
        }
        if !(self.repeat_penalty > 0.0) {
            return Err(format!("repeat-penalty must be > 0, got {}", self.repeat_penalty));
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
        ChainSampler { cfg, rng, recent: std::collections::VecDeque::new() }
    }

    fn penalized(&self, logits: &[f32]) -> Vec<f32> {
        let mut out = logits.to_vec();
        if self.cfg.repeat_penalty != 1.0 {
            // Penalize once per distinct token in the window.
            let distinct: std::collections::HashSet<u32> =
                self.recent.iter().copied().collect();
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

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i; // strict > keeps the lowest index on ties
        }
    }
    best as u32
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
```

Also refactor `Greedy` to reuse the helper (keeps one argmax definition):

```rust
impl Sampler for Greedy {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        argmax(logits)
    }
}
```

Update `crates/inferno-runtime/src/lib.rs`:

```rust
pub use sampler::{ChainSampler, Greedy, Sampler, SamplerConfig};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run -p inferno-runtime`
Expected: all PASS (existing `greedy_argmax_lowest_index_tie_break` must still pass against the refactored `Greedy`).

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-runtime/src/sampler.rs crates/inferno-runtime/src/lib.rs
git commit -m "feat(runtime): SamplerConfig + ChainSampler (repeat penalty, greedy path)"
```

---

### Task 4: ChainSampler stochastic stages (top-k / top-p / min-p / temperature / draw)

**Files:**
- Modify: `crates/inferno-runtime/src/sampler.rs` (complete `sample`)

**Interfaces:**
- Consumes: Task 3's `ChainSampler` internals; Task 1's `next_f64`.
- Produces: the final `Sampler::sample` for `ChainSampler`. No signature changes.
- Seed-42 draw sequence (from Task 1's pinned `next_f64` test): `0.0838…, 0.3789…, 0.6800…, 0.9246…` — the exact-pick tests below are hand-derivable from these.

- [ ] **Step 1: Write the failing tests**

Append to the `tests` module in `sampler.rs`:

```rust
    /// Four draws over four equally-likely tokens with seed 42 walk the
    /// pinned uniform sequence 0.0838→0.3789→0.6800→0.9246, so the picks
    /// are exactly 0,1,2,3 (cumulative bins of width 0.25).
    #[test]
    fn seeded_exact_picks_uniform_logits() {
        let cfg = SamplerConfig { temperature: 1.0, seed: 42, ..Default::default() };
        let mut s = ChainSampler::new(cfg);
        let logits = [3.0, 3.0, 3.0, 3.0];
        let picks: Vec<u32> = (0..4).map(|_| s.sample(&logits)).collect();
        assert_eq!(picks, vec![0, 1, 2, 3]);
    }

    /// A dominant logit wins under any of the pinned seed-42 draws:
    /// p(idx 2) = e^10 / (3 + e^10) ≈ 0.99986 > 0.9246 (the largest draw).
    #[test]
    fn seeded_exact_pick_dominant_logit() {
        let cfg = SamplerConfig { temperature: 1.0, seed: 42, ..Default::default() };
        let mut s = ChainSampler::new(cfg);
        for _ in 0..4 {
            assert_eq!(s.sample(&[0.0, 0.0, 10.0, 0.0]), 2);
        }
    }

    /// top-k = 1 must be greedy regardless of temperature or seed.
    #[test]
    fn top_k_one_is_greedy() {
        for seed in [0, 1, 42, 999] {
            let cfg = SamplerConfig { temperature: 2.5, top_k: 1, seed, ..Default::default() };
            assert_eq!(pick(cfg, &[], &[0.1, 0.9, 0.3]), 1);
        }
    }

    /// Probabilities 0.5/0.3/0.2 (logits ln p). top_p = 0.5 keeps exactly
    /// the first token (cumulative 0.5 >= 0.5) → always index 0.
    #[test]
    fn top_p_cumulative_mass_edge() {
        let logits = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        for seed in 0..20 {
            let cfg = SamplerConfig { temperature: 1.0, top_p: 0.5, seed, ..Default::default() };
            assert_eq!(pick(cfg, &[], &logits), 0);
        }
        // top_p = 0.8: cumulative hits 0.8 at the second token → index 2
        // is never drawn.
        for seed in 0..20 {
            let cfg = SamplerConfig { temperature: 1.0, top_p: 0.8, seed, ..Default::default() };
            assert_ne!(pick(cfg, &[], &logits), 2);
        }
    }

    /// min_p = 0.5 with max prob 0.5 → cutoff 0.25: drops the 0.2 token,
    /// keeps 0.5 and 0.3. min_p = 0.7 → cutoff 0.35: only the max survives.
    #[test]
    fn min_p_relative_cutoff() {
        let logits = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        for seed in 0..20 {
            let cfg = SamplerConfig { temperature: 1.0, min_p: 0.5, seed, ..Default::default() };
            assert_ne!(pick(cfg, &[], &logits), 2);
            let cfg = SamplerConfig { temperature: 1.0, min_p: 0.7, seed, ..Default::default() };
            assert_eq!(pick(cfg, &[], &logits), 0);
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo nextest run -p inferno-runtime sampler`
Expected: `seeded_exact_picks_uniform_logits` FAILS (picks are all 0 — the Task 3 stub argmaxes). `top_k_one_is_greedy` may pass by accident; the mass-based ones fail.

- [ ] **Step 3: Implement the full chain**

Replace `ChainSampler`'s `sample` body in `sampler.rs`:

```rust
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
                if cum >= self.cfg.top_p as f64 {
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
```

Add the helper (above `impl Sampler for ChainSampler`):

```rust
/// Numerically stable softmax over `n` values (max-subtracted, f64).
fn softmax(vals: impl Iterator<Item = f64>, n: usize) -> Vec<f64> {
    let vals: Vec<f64> = vals.collect();
    debug_assert_eq!(vals.len(), n);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = vals.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run -p inferno-runtime`
Expected: all PASS, including every Task 3 test (the neutral-≡-greedy property now guards the full chain).

- [ ] **Step 5: Commit**

```bash
git add crates/inferno-runtime/src/sampler.rs
git commit -m "feat(runtime): full sampling chain — top-k/top-p/min-p/temperature/seeded draw"
```

---

### Task 5: `inferno run` sampling flags + determinism tests

**Files:**
- Modify: `cli/src/main.rs:33-48` (Run variant + dispatch)
- Modify: `cli/src/run.rs:9-67` (`run` signature + sampler construction)
- Modify: `crates/inferno-runtime/src/generate.rs` (tests only)
- Test: `cli/tests/run.rs`

**Interfaces:**
- Consumes: `SamplerConfig`, `ChainSampler` (Tasks 3/4).
- Produces: `run::run(model: &Path, prompt: &str, max_tokens: usize, max_seq_len: usize, interp: bool, sampling: SamplerConfig) -> ExitCode`. New `run` flags: `--temperature` (f32, default 0.0), `--top-k` (usize, default 0), `--top-p` (f32, default 1.0), `--min-p` (f32, default 0.0), `--repeat-penalty` (f32, default 1.0), `--repeat-last-n` (usize, default 64), `--seed` (u64, default 0).

- [ ] **Step 1: Write the failing runtime determinism tests**

In `crates/inferno-runtime/src/generate.rs` `tests`:

```rust
    fn sampled_ids(seed: u64) -> Vec<u32> {
        use crate::sampler::{ChainSampler, SamplerConfig};
        let mut g = Generator::load(&fixture("tiny.gguf"), 64).unwrap();
        let mut s = ChainSampler::new(SamplerConfig {
            temperature: 5.0, // near-uniform: forces real draws
            seed,
            ..Default::default()
        });
        g.generate("the", 8, &mut s, &mut |_| ControlFlow::Continue(()))
            .unwrap()
            .0
    }

    /// Blocking-tier determinism gate from the M4a spec: same seed → same
    /// token sequence; different seeds diverge at temperature > 0.
    #[test]
    fn same_seed_same_tokens_different_seed_diverges() {
        assert_eq!(sampled_ids(7), sampled_ids(7));
        // At temperature 5 over the whole vocab, 8 identical draws across
        // two seeds is astronomically unlikely; a collision here means the
        // seed is being ignored.
        assert_ne!(sampled_ids(1), sampled_ids(2));
    }
```

- [ ] **Step 2: Run to verify it passes at runtime level**

Run: `cargo nextest run -p inferno-runtime same_seed`
Expected: PASS (Tasks 1–4 already make this true; this pins it). If `sampled_ids(1) == sampled_ids(2)` fails because generation hit EOS after 0–1 tokens, raise `max_tokens` to 16 — do not weaken the assertion.

- [ ] **Step 3: Write the failing CLI tests**

In `cli/tests/run.rs`:

```rust
#[test]
fn run_sampling_same_seed_is_reproducible() {
    let out = |seed: &str| {
        let a = Command::cargo_bin("inferno")
            .unwrap()
            .args([
                "run", &fixture("tiny.gguf"), "--interp",
                "--prompt", "the", "--max-tokens", "8", "--max-seq-len", "64",
                "--temperature", "5.0", "--seed", seed,
            ])
            .assert()
            .success();
        String::from_utf8(a.get_output().stdout.clone()).unwrap()
    };
    assert_eq!(out("7"), out("7"));
}

#[test]
fn run_rejects_invalid_sampling_flags() {
    Command::cargo_bin("inferno")
        .unwrap()
        .args([
            "run", &fixture("tiny.gguf"), "--interp",
            "--prompt", "the", "--top-p", "0",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("top-p"));
}
```

- [ ] **Step 4: Run to verify they fail**

Run: `cargo nextest run -p inferno run_sampling`
Expected: FAIL — unknown `--temperature` / `--top-p` flags (clap error, but `run_rejects_invalid_sampling_flags` fails for the wrong reason: clap's message doesn't mention validation; it will pass only after real validation exists — confirm the stderr assertion fails now).

- [ ] **Step 5: Implement**

`cli/src/main.rs` — extend the `Run` variant:

```rust
    Run {
        /// Path to a .gguf file, an MLX directory, or a .safetensors file.
        model: PathBuf,
        #[arg(long, short)]
        prompt: String,
        /// Maximum tokens to generate.
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// KV-cache capacity (clamped to the model's context length).
        #[arg(long, default_value_t = 4096)]
        max_seq_len: usize,
        /// Use the M1 scalar interpreter instead of the compiled path.
        #[arg(long)]
        interp: bool,
        /// Sampling temperature; 0 = greedy (the default).
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Keep only the k highest-logit tokens; 0 = disabled.
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        /// Nucleus sampling mass in (0, 1]; 1.0 = disabled.
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        /// Drop tokens below min-p × max-probability; 0 = disabled.
        #[arg(long, default_value_t = 0.0)]
        min_p: f32,
        /// Penalty for recently seen tokens; 1.0 = disabled.
        #[arg(long, default_value_t = 1.0)]
        repeat_penalty: f32,
        /// Repeat-penalty window length.
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
        /// RNG seed for sampling.
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
```

Dispatch (build the config in `main`, keep `run::run`'s surface small):

```rust
        Command::Run {
            model,
            prompt,
            max_tokens,
            max_seq_len,
            interp,
            temperature,
            top_k,
            top_p,
            min_p,
            repeat_penalty,
            repeat_last_n,
            seed,
        } => run::run(
            &model,
            &prompt,
            max_tokens,
            max_seq_len,
            interp,
            inferno_runtime::SamplerConfig {
                temperature,
                top_k,
                top_p,
                min_p,
                repeat_penalty,
                repeat_last_n,
                seed,
            },
        ),
```

`cli/src/run.rs` — new signature and sampler:

```rust
use inferno_runtime::{ChainSampler, Generator, SamplerConfig};

pub fn run(
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    max_seq_len: usize,
    interp: bool,
    sampling: SamplerConfig,
) -> ExitCode {
    if let Err(e) = sampling.validate() {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    let mut sampler = ChainSampler::new(sampling);
    // ... existing body unchanged, except the generate call:
    let result = generator.generate(prompt, max_tokens, &mut sampler, &mut |bytes| ...
```

(`Greedy` import goes away from `run.rs`; `bench.rs`/`diff.rs` keep theirs. A neutral `SamplerConfig` ≡ `Greedy` is a tested property from Task 3, so default `inferno run` output is bit-identical to before.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo nextest run -p inferno && cargo nextest run -p inferno-runtime`
Expected: all PASS, including the existing `run_streams_tokens_from_gguf_fixture` (defaults unchanged) and inspect snapshots.

- [ ] **Step 7: Lint + commit**

Run: `mise run lint`
Expected: clean.

```bash
git add cli/src/main.rs cli/src/run.rs cli/tests/run.rs crates/inferno-runtime/src/generate.rs
git commit -m "feat(cli): sampling flags on inferno run; seeded determinism pinned"
```

---

### Task 6: llama-bench JSON parsing + invocation

**Files:**
- Create: `cli/src/llama_bench.rs`
- Create: `cli/tests/fixtures/llama-bench.json` (golden fixture)
- Modify: `cli/src/main.rs` (add `mod llama_bench;`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Task 7/8):
  - `pub struct LlamaBenchRow { pub build_commit: String, pub cpu_info: String, pub model_type: String, pub n_prompt: u64, pub n_gen: u64, pub n_threads: u64, pub avg_ts: f64, pub stddev_ts: f64 }` (serde `Deserialize` + `Serialize`, unknown JSON fields ignored — serde's default).
  - `pub fn parse(json: &str) -> Result<Vec<LlamaBenchRow>, String>` — strict on the required fields: a missing field is a schema-drift error naming the field, per the spec's "fails loudly" requirement.
  - `pub fn run_llama_bench(bin: &Path, model: &Path, pp: u64, tg: u64, threads: &[u64], reps: u64) -> Result<Vec<LlamaBenchRow>, String>` — invokes `llama-bench -m <model> -p <pp> -n <tg> -t <t1,t2> -r <reps> -o json`, captures stdout, passes stderr through on non-zero exit; a `NotFound` spawn error maps to: `"llama-bench not found — run inside `devenv shell` (it provides the pinned llama.cpp) or pass --llama-bench <path>"`.
  - `pub fn find_row<'a>(rows: &'a [LlamaBenchRow], n_prompt: u64, n_gen: u64, n_threads: u64) -> Option<&'a LlamaBenchRow>`.

- [ ] **Step 1: Create the golden fixture**

`llama-bench -o json` emits a JSON array with one object per test: a prompt-processing row (`n_prompt` > 0, `n_gen` = 0) and a text-generation row (`n_prompt` = 0, `n_gen` > 0), repeated per `-t` value. Write `cli/tests/fixtures/llama-bench.json` with this realistic capture shape (extra fields present exactly so the test proves they're tolerated):

```json
[
  {
    "build_commit": "3ab8b3a9", "build_number": 4568,
    "cpu_info": "AMD Ryzen 9 3900 12-Core Processor",
    "gpu_info": "", "backends": "CPU",
    "model_filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
    "model_type": "qwen2 1B Q8_0", "model_size": 531068928,
    "model_n_params": 494032768,
    "n_batch": 2048, "n_ubatch": 512, "n_threads": 12,
    "cpu_mask": "0x0", "cpu_strict": false, "poll": 50,
    "type_k": "f16", "type_v": "f16", "n_gpu_layers": 99,
    "split_mode": "layer", "main_gpu": 0, "no_kv_offload": false,
    "flash_attn": false, "tensor_split": "0.00", "use_mmap": true,
    "embeddings": false,
    "n_prompt": 512, "n_gen": 0, "test_time": "2026-07-06T10:00:00Z",
    "avg_ns": 1052631578, "stddev_ns": 10526315,
    "avg_ts": 486.4, "stddev_ts": 4.9,
    "samples_ns": [ 1052631578 ], "samples_ts": [ 486.4 ]
  },
  {
    "build_commit": "3ab8b3a9", "build_number": 4568,
    "cpu_info": "AMD Ryzen 9 3900 12-Core Processor",
    "gpu_info": "", "backends": "CPU",
    "model_filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
    "model_type": "qwen2 1B Q8_0", "model_size": 531068928,
    "model_n_params": 494032768,
    "n_batch": 2048, "n_ubatch": 512, "n_threads": 12,
    "cpu_mask": "0x0", "cpu_strict": false, "poll": 50,
    "type_k": "f16", "type_v": "f16", "n_gpu_layers": 99,
    "split_mode": "layer", "main_gpu": 0, "no_kv_offload": false,
    "flash_attn": false, "tensor_split": "0.00", "use_mmap": true,
    "embeddings": false,
    "n_prompt": 0, "n_gen": 128, "test_time": "2026-07-06T10:00:30Z",
    "avg_ns": 1523809523, "stddev_ns": 15238095,
    "avg_ts": 84.0, "stddev_ts": 0.8,
    "samples_ns": [ 1523809523 ], "samples_ts": [ 84.0 ]
  }
]
```

**Fixture provenance caveat:** this shape is written from the pinned llama.cpp's `llama-bench` JSON schema. During Task 9's protocol run (the first time real `llama-bench` output flows through the parser), if the real output has different field names, replace this fixture with a real capture (`llama-bench -m <model> -p 8 -n 4 -r 1 -o json` inside `devenv shell`) and adjust `LlamaBenchRow` — that is a fixture correction, not schema drift.

- [ ] **Step 2: Write the failing tests**

In `cli/src/llama_bench.rs` (tests module at the bottom):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_json() -> String {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/llama-bench.json"
        ))
        .unwrap()
    }

    #[test]
    fn parses_golden_fixture() {
        let rows = parse(&fixture_json()).unwrap();
        assert_eq!(rows.len(), 2);
        let pp = find_row(&rows, 512, 0, 12).unwrap();
        assert_eq!(pp.avg_ts, 486.4);
        assert_eq!(pp.stddev_ts, 4.9);
        assert_eq!(pp.build_commit, "3ab8b3a9");
        let tg = find_row(&rows, 0, 128, 12).unwrap();
        assert_eq!(tg.avg_ts, 84.0);
        assert!(find_row(&rows, 512, 0, 1).is_none());
    }

    /// Schema drift (a required field vanishing) must fail loudly with the
    /// field name, not produce a half-report.
    #[test]
    fn missing_required_field_is_a_loud_error() {
        let broken = fixture_json().replace("\"avg_ts\"", "\"renamed_ts\"");
        let err = parse(&broken).unwrap_err();
        assert!(err.contains("avg_ts"), "error should name the field: {err}");
    }

    #[test]
    fn non_json_input_is_an_error() {
        assert!(parse("ggml_init: using CPU backend\n").is_err());
    }
}
```

- [ ] **Step 3: Run to verify they fail**

Run: `cargo nextest run -p inferno llama_bench`
Expected: COMPILE ERROR — module functions not defined.

- [ ] **Step 4: Implement**

`cli/src/llama_bench.rs`:

```rust
//! Drive the devenv-pinned `llama-bench` and parse its `-o json` output.
//! The parser is strict on required fields (schema drift fails loudly,
//! citing the field) and tolerant of extra fields (serde's default).

use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LlamaBenchRow {
    pub build_commit: String,
    pub cpu_info: String,
    pub model_type: String,
    pub n_prompt: u64,
    pub n_gen: u64,
    pub n_threads: u64,
    /// Average tokens/sec for this test row.
    pub avg_ts: f64,
    pub stddev_ts: f64,
}

pub fn parse(json: &str) -> Result<Vec<LlamaBenchRow>, String> {
    serde_json::from_str(json).map_err(|e| {
        format!(
            "unparseable llama-bench JSON (schema drift vs the devenv-pinned \
             llama.cpp? see the M4a spec): {e}"
        )
    })
}

pub fn find_row<'a>(
    rows: &'a [LlamaBenchRow],
    n_prompt: u64,
    n_gen: u64,
    n_threads: u64,
) -> Option<&'a LlamaBenchRow> {
    rows.iter().find(|r| {
        r.n_prompt == n_prompt && r.n_gen == n_gen && r.n_threads == n_threads
    })
}

pub fn run_llama_bench(
    bin: &Path,
    model: &Path,
    pp: u64,
    tg: u64,
    threads: &[u64],
    reps: u64,
) -> Result<Vec<LlamaBenchRow>, String> {
    let t_list = threads
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let out = Command::new(bin)
        .arg("-m")
        .arg(model)
        .args(["-p", &pp.to_string(), "-n", &tg.to_string()])
        .args(["-t", &t_list, "-r", &reps.to_string(), "-o", "json"])
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!(
                    "llama-bench not found at `{}` — run inside `devenv shell` \
                     (it provides the pinned llama.cpp) or pass --llama-bench <path>",
                    bin.display()
                )
            } else {
                format!("failed to spawn llama-bench: {e}")
            }
        })?;
    if !out.status.success() {
        return Err(format!(
            "llama-bench exited with {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    parse(&String::from_utf8_lossy(&out.stdout))
}
```

Add `mod llama_bench;` to `cli/src/main.rs` (next to `mod bench;`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo nextest run -p inferno llama_bench`
Expected: 3 tests PASS. (`missing_required_field_is_a_loud_error` passes because serde_json's "missing field \`avg_ts\`" message contains the field name.)

- [ ] **Step 6: Commit**

```bash
git add cli/src/llama_bench.rs cli/tests/fixtures/llama-bench.json cli/src/main.rs
git commit -m "feat(cli): llama-bench JSON invocation + strict parser with golden fixture"
```

---

### Task 7: Inferno-side measurement (backend-direct, warmup + repeats)

**Files:**
- Modify: `cli/src/bench.rs` (add below the untouched `bench_compiled`/`MARGIN`)

**Interfaces:**
- Consumes: `inferno_core::Engine` (`Engine::load(model, max_seq_len)`, `.compiled_backend()`), `inferno_runtime::{Backend, Greedy, Sampler}`, `clamp_max_seq_len` from `crate::run`.
- Produces (used by Task 8):
  - `pub struct Measurement { pub mean_tok_s: f64, pub stddev_tok_s: f64 }`
  - `pub struct InfernoNumbers { pub pp: Measurement, pub tg: Measurement }`
  - `pub fn measure_inferno(model: &Path, pp: usize, tg: usize, reps: usize) -> Result<InfernoNumbers, Box<dyn std::error::Error>>`
  - `fn mean_stddev(samples: &[f64]) -> (f64, f64)` (sample stddev, n−1; 0.0 when n < 2)

Measurement drives the `Backend` directly — no tokenizer, no UTF-8 streaming, no EOS handling in the timed path (spec: synthetic token ids, content irrelevant to speed). Compile cost lands in `Engine::load`, outside timing; one untimed warmup rep before the timed reps warms the mmap'd weight pages so rep 1 doesn't pay page faults.

- [ ] **Step 1: Write the failing tests**

In `cli/src/bench.rs` tests module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_stddev_matches_hand_computation() {
        let (m, s) = mean_stddev(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((m - 5.0).abs() < 1e-12);
        // Sample stddev (n-1) of that set is sqrt(32/7).
        assert!((s - (32.0f64 / 7.0).sqrt()).abs() < 1e-12);
        let (m1, s1) = mean_stddev(&[3.5]);
        assert_eq!((m1, s1), (3.5, 0.0));
    }

    /// End-to-end smoke on the tiny fixture: compiles once (artifact cache),
    /// then measures. Numbers must be finite and positive; anything else
    /// means the timed path is broken, not slow.
    #[test]
    fn measure_inferno_smoke_on_fixture() {
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crates/inferno-formats/tests/fixtures/tiny.gguf");
        let n = measure_inferno(&model, 8, 4, 2).unwrap();
        for m in [&n.pp, &n.tg] {
            assert!(m.mean_tok_s.is_finite() && m.mean_tok_s > 0.0);
            assert!(m.stddev_tok_s.is_finite() && m.stddev_tok_s >= 0.0);
        }
    }

    #[test]
    fn measure_inferno_rejects_prompt_beyond_context() {
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crates/inferno-formats/tests/fixtures/tiny.gguf");
        // tiny.gguf's context length is far below 1<<20.
        assert!(measure_inferno(&model, 1 << 20, 4, 1).is_err());
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo nextest run -p inferno measure_inferno mean_stddev`
Expected: COMPILE ERROR — functions not defined.

- [ ] **Step 3: Implement**

Append to `cli/src/bench.rs`:

```rust
use std::time::Instant;

use inferno_core::Engine;
use inferno_runtime::{Backend, Greedy, Sampler};

use crate::run::clamp_max_seq_len;

pub struct Measurement {
    pub mean_tok_s: f64,
    pub stddev_tok_s: f64,
}

pub struct InfernoNumbers {
    pub pp: Measurement,
    pub tg: Measurement,
}

/// Sample mean and sample (n-1) standard deviation; stddev is 0 for n < 2.
fn mean_stddev(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    if samples.len() < 2 {
        return (mean, 0.0);
    }
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

/// Measure compiled prefill (pp synthetic tokens) and decode (tg greedy
/// steps) throughput, `reps` timed repetitions after one untimed warmup.
/// Drives the `Backend` directly: no tokenizer/EOS/UTF-8 in the timed path.
pub fn measure_inferno(
    model: &Path,
    pp: usize,
    tg: usize,
    reps: usize,
) -> Result<InfernoNumbers, Box<dyn std::error::Error>> {
    let needed = pp + tg;
    let max_seq_len = clamp_max_seq_len(model, needed)?;
    if max_seq_len < needed {
        return Err(format!(
            "model context length {max_seq_len} is too small for pp={pp} + tg={tg}"
        )
        .into());
    }
    let desc = inferno_formats::load_desc(model)?;
    // vocab_size is u64 in ModelDesc; token ids are u32.
    let vocab = u32::try_from(desc.hyperparams.vocab_size)
        .map_err(|_| "vocab size exceeds u32")?;
    if vocab < 2 {
        return Err("model vocab too small for synthetic prompt".into());
    }
    // Synthetic prompt: valid ids cycling [1, vocab). Content is irrelevant
    // to throughput (mirrors llama-bench's approach).
    let ids: Vec<u32> = (0..pp).map(|i| 1 + (i as u32 % (vocab - 1))).collect();

    // Compile (or cache-hit) happens here, outside any timed region.
    let engine = Engine::load(model, max_seq_len)?;
    let mut backend = engine.compiled_backend()?;

    let mut run_once = |backend: &mut dyn Backend| -> Result<(f64, f64), Box<dyn std::error::Error>> {
        backend.reset();
        let t0 = Instant::now();
        let mut last = backend.forward(&ids)?;
        let pp_secs = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for _ in 0..tg {
            let next = Greedy.sample(&last);
            last = backend.forward(&[next])?;
        }
        let tg_secs = t1.elapsed().as_secs_f64();
        Ok((
            pp as f64 / pp_secs.max(1e-9),
            tg as f64 / tg_secs.max(1e-9),
        ))
    };

    run_once(&mut backend)?; // warmup: touches every mmap'd weight page
    let mut pp_samples = Vec::with_capacity(reps);
    let mut tg_samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let (p, t) = run_once(&mut backend)?;
        pp_samples.push(p);
        tg_samples.push(t);
    }
    let (pm, ps) = mean_stddev(&pp_samples);
    let (tm, ts) = mean_stddev(&tg_samples);
    Ok(InfernoNumbers {
        pp: Measurement { mean_tok_s: pm, stddev_tok_s: ps },
        tg: Measurement { mean_tok_s: tm, stddev_tok_s: ts },
    })
}
```

Note: `Greedy.sample(&last)` needs a `mut` binding per clippy — if it complains, use `let mut greedy = Greedy;` outside the loop and `greedy.sample(&last)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo nextest run -p inferno measure_inferno mean_stddev`
Expected: 3 tests PASS. The smoke test compiles `tiny.gguf` (needs the devenv LLVM, same as the existing `inferno-core` artifact tests — run inside `devenv shell`).

- [ ] **Step 5: Commit**

```bash
git add cli/src/bench.rs
git commit -m "feat(cli): inferno-side bench measurement — backend-direct, warmup + repeats"
```

---

### Task 8: `inferno bench` subcommand, report, mise task

**Files:**
- Modify: `cli/src/bench.rs` (report + command entry)
- Modify: `cli/src/main.rs` (Bench variant + dispatch)
- Modify: `mise.toml` (`[tasks.bench]`)
- Test: `cli/src/bench.rs` unit tests (report rendering is pure)

**Interfaces:**
- Consumes: `measure_inferno`/`InfernoNumbers` (Task 7), `llama_bench::{run_llama_bench, find_row, LlamaBenchRow}` (Task 6), `inferno_target::TargetDesc::detect()` (`.topology.physical_cores`, `.topology.logical_cores`).
- Produces:
  - CLI: `inferno bench <model> [--pp 512] [--tg 128] [--reps 5] [--threads 0=auto] [--llama-bench <path>] [--json]`
  - `pub struct BenchReport` (serde `Serialize`) — fields below.
  - `pub fn bench(model: &Path, pp: u64, tg: u64, reps: u64, threads: u64, llama_bench_bin: Option<&Path>, json: bool) -> ExitCode`
  - `fn render_table(r: &BenchReport) -> String` (pure; insta-snapshot-tested)

- [ ] **Step 1: Write the failing rendering test**

In `cli/src/bench.rs` tests:

```rust
    #[test]
    fn render_table_snapshot() {
        let r = BenchReport {
            model: "qwen2.5-0.5b-instruct-q8_0.gguf".into(),
            model_type: "qwen2 1B Q8_0".into(),
            cpu_info: "AMD Ryzen 9 3900 12-Core Processor".into(),
            physical_cores: 12,
            logical_cores: 24,
            inferno_version: "0.1.0".into(),
            inferno_git: "0b09ece".into(),
            llama_build_commit: "3ab8b3a9".into(),
            pp: 512,
            tg: 128,
            reps: 5,
            inferno_threads: 1,
            llama_threads: 12,
            inferno_pp_tok_s: 110.2,
            inferno_pp_stddev: 1.4,
            inferno_tg_tok_s: 26.1,
            inferno_tg_stddev: 0.3,
            llama_pp_tok_s: 486.4,
            llama_pp_stddev: 4.9,
            llama_tg_tok_s: 84.0,
            llama_tg_stddev: 0.8,
            llama_t1_pp_tok_s: Some(52.1),
            llama_t1_pp_stddev: Some(0.5),
            llama_t1_tg_tok_s: Some(9.3),
            llama_t1_tg_stddev: Some(0.1),
        };
        insta::assert_snapshot!("bench_report_table", render_table(&r));
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo nextest run -p inferno render_table`
Expected: COMPILE ERROR — `BenchReport`, `render_table` not defined.

- [ ] **Step 3: Implement report + command**

Append to `cli/src/bench.rs`:

```rust
/// One recorded comparison data point (the `--json` shape; the human table
/// is rendered from the same struct). Recorded in the M4a spec's
/// Amendments section per the protocol.
#[derive(serde::Serialize)]
pub struct BenchReport {
    pub model: String,
    pub model_type: String,
    pub cpu_info: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub inferno_version: String,
    pub inferno_git: String,
    pub llama_build_commit: String,
    pub pp: u64,
    pub tg: u64,
    pub reps: u64,
    /// M3 generated code is single-threaded; recorded so old data points
    /// stay interpretable after M4b lands threading.
    pub inferno_threads: u64,
    pub llama_threads: u64,
    pub inferno_pp_tok_s: f64,
    pub inferno_pp_stddev: f64,
    pub inferno_tg_tok_s: f64,
    pub inferno_tg_stddev: f64,
    pub llama_pp_tok_s: f64,
    pub llama_pp_stddev: f64,
    pub llama_tg_tok_s: f64,
    pub llama_tg_stddev: f64,
    /// The `-t 1` per-thread-parity diagnostic rows (None when the
    /// full-thread run already was `-t 1`).
    pub llama_t1_pp_tok_s: Option<f64>,
    pub llama_t1_pp_stddev: Option<f64>,
    pub llama_t1_tg_tok_s: Option<f64>,
    pub llama_t1_tg_stddev: Option<f64>,
}

fn render_table(r: &BenchReport) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "model: {} ({})", r.model, r.model_type);
    let _ = writeln!(
        s,
        "cpu: {} ({} physical / {} logical cores)",
        r.cpu_info, r.physical_cores, r.logical_cores
    );
    let _ = writeln!(
        s,
        "inferno {} ({}) vs llama.cpp {} | pp={} tg={} reps={}",
        r.inferno_version, r.inferno_git, r.llama_build_commit, r.pp, r.tg, r.reps
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "{:<22} {:>7} {:>18} {:>18}",
        "engine", "threads", format!("pp{} tok/s", r.pp), format!("tg{} tok/s", r.tg)
    );
    let mut row = |name: &str, threads: u64, pp: f64, pps: f64, tg: f64, tgs: f64| {
        let _ = writeln!(
            s,
            "{:<22} {:>7} {:>11.2} ± {:<5.2} {:>11.2} ± {:<5.2}",
            name, threads, pp, pps, tg, tgs
        );
    };
    row(
        "inferno (compiled)",
        r.inferno_threads,
        r.inferno_pp_tok_s,
        r.inferno_pp_stddev,
        r.inferno_tg_tok_s,
        r.inferno_tg_stddev,
    );
    row(
        "llama.cpp",
        r.llama_threads,
        r.llama_pp_tok_s,
        r.llama_pp_stddev,
        r.llama_tg_tok_s,
        r.llama_tg_stddev,
    );
    if let (Some(pp), Some(pps), Some(tg), Some(tgs)) = (
        r.llama_t1_pp_tok_s,
        r.llama_t1_pp_stddev,
        r.llama_t1_tg_tok_s,
        r.llama_t1_tg_stddev,
    ) {
        row("llama.cpp (t=1 diag)", 1, pp, pps, tg, tgs);
    }
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "ratio (inferno/llama.cpp): pp {:.2}x | tg {:.2}x",
        r.inferno_pp_tok_s / r.llama_pp_tok_s.max(1e-9),
        r.inferno_tg_tok_s / r.llama_tg_tok_s.max(1e-9),
    );
    s
}

fn git_short_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// `inferno bench`: the M4a manual comparison protocol (see the spec —
/// quiet hardware, devenv shell, release build; data points recorded in
/// the spec's Amendments, never a CI gate).
#[allow(clippy::too_many_arguments)]
pub fn bench(
    model: &Path,
    pp: u64,
    tg: u64,
    reps: u64,
    threads: u64,
    llama_bench_bin: Option<&Path>,
    json: bool,
) -> ExitCode {
    let inner = || -> Result<BenchReport, Box<dyn std::error::Error>> {
        if pp == 0 || tg == 0 || reps == 0 {
            return Err("--pp, --tg, and --reps must all be > 0".into());
        }
        let target = inferno_target::TargetDesc::detect()?;
        let threads = if threads == 0 {
            u64::from(target.topology.physical_cores)
        } else {
            threads
        };
        let inferno = measure_inferno(model, pp as usize, tg as usize, reps as usize)?;
        let bin = llama_bench_bin
            .map(Path::to_path_buf)
            .unwrap_or_else(|| "llama-bench".into());
        let t_list: Vec<u64> = if threads == 1 { vec![1] } else { vec![threads, 1] };
        let rows = crate::llama_bench::run_llama_bench(&bin, model, pp, tg, &t_list, reps)?;
        let pick = |n_prompt: u64, n_gen: u64, t: u64| {
            crate::llama_bench::find_row(&rows, n_prompt, n_gen, t)
                .ok_or_else(|| {
                    format!("llama-bench output missing the (pp={n_prompt}, tg={n_gen}, t={t}) row")
                })
        };
        let lpp = pick(pp, 0, threads)?;
        let ltg = pick(0, tg, threads)?;
        let (t1pp, t1tg) = if threads == 1 {
            (None, None)
        } else {
            (Some(pick(pp, 0, 1)?), Some(pick(0, tg, 1)?))
        };
        Ok(BenchReport {
            model: model
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| model.display().to_string()),
            model_type: lpp.model_type.clone(),
            cpu_info: lpp.cpu_info.clone(),
            physical_cores: target.topology.physical_cores,
            logical_cores: target.topology.logical_cores,
            inferno_version: env!("CARGO_PKG_VERSION").into(),
            inferno_git: git_short_hash(),
            llama_build_commit: lpp.build_commit.clone(),
            pp,
            tg,
            reps,
            inferno_threads: 1, // M3 generated code is single-threaded
            llama_threads: threads,
            inferno_pp_tok_s: inferno.pp.mean_tok_s,
            inferno_pp_stddev: inferno.pp.stddev_tok_s,
            inferno_tg_tok_s: inferno.tg.mean_tok_s,
            inferno_tg_stddev: inferno.tg.stddev_tok_s,
            llama_pp_tok_s: lpp.avg_ts,
            llama_pp_stddev: lpp.stddev_ts,
            llama_tg_tok_s: ltg.avg_ts,
            llama_tg_stddev: ltg.stddev_ts,
            llama_t1_pp_tok_s: t1pp.map(|r| r.avg_ts),
            llama_t1_pp_stddev: t1pp.map(|r| r.stddev_ts),
            llama_t1_tg_tok_s: t1tg.map(|r| r.avg_ts),
            llama_t1_tg_stddev: t1tg.map(|r| r.stddev_ts),
        })
    };
    match inner() {
        Ok(report) => {
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report)
                        .expect("BenchReport serializes: plain numbers and strings")
                );
            } else {
                print!("{}", render_table(&report));
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
```

`cli/src/main.rs` — add the variant (after `Compile`):

```rust
    /// Compare inferno's compiled path against the devenv-pinned llama.cpp
    /// (`llama-bench`) on the same model: prefill (pp) and decode (tg)
    /// tok/s, mean ± stddev. Manual protocol — quiet hardware, devenv
    /// shell, release build; see the M4a spec. Never a CI gate.
    Bench {
        /// Path to a .gguf file (must be loadable by both engines).
        model: PathBuf,
        /// Synthetic prompt length (prefill test size).
        #[arg(long, default_value_t = 512)]
        pp: u64,
        /// Decode test length in tokens.
        #[arg(long, default_value_t = 128)]
        tg: u64,
        /// Timed repetitions per engine (after one untimed warmup).
        #[arg(long, default_value_t = 5)]
        reps: u64,
        /// llama.cpp thread count; 0 = physical cores. A t=1 diagnostic
        /// row is recorded alongside unless this is 1.
        #[arg(long, default_value_t = 0)]
        threads: u64,
        /// Path to llama-bench (default: found on PATH via devenv shell).
        #[arg(long)]
        llama_bench: Option<PathBuf>,
        /// Emit the machine-readable data point instead of the table.
        #[arg(long)]
        json: bool,
    },
```

Dispatch arm:

```rust
        Command::Bench {
            model,
            pp,
            tg,
            reps,
            threads,
            llama_bench,
            json,
        } => bench::bench(&model, pp, tg, reps, threads, llama_bench.as_deref(), json),
```

Check `cli/Cargo.toml` has `inferno-target` — if not (it currently doesn't), add `inferno-target.workspace = true` to `[dependencies]` (it is already a workspace member dep of other crates; this adds no external dependency).

`mise.toml` — after `[tasks.bench-kernels]`:

```toml
[tasks.bench]
description = "inferno vs llama.cpp comparison (M4a protocol: run inside devenv shell, quiet hardware; record data points in the M4a spec): mise run bench -- <model.gguf>"
run = "cargo run --release -p inferno -- bench"
```

(mise appends everything after `--` to the command, so `mise run bench -- model.gguf --json` works.)

- [ ] **Step 4: Run tests, review the snapshot**

Run: `cargo nextest run -p inferno && cargo insta review`
Expected: `render_table_snapshot` produces a new snapshot — review it (readable table, both engines, t=1 row, ratio line), accept if right. **Never blind-accept** (AGENTS.md).

- [ ] **Step 5: Add the CLI-level validation test**

Create `cli/tests/bench.rs`:

```rust
use assert_cmd::Command;
use predicates::prelude::*;

fn fixture(p: &str) -> String {
    format!(
        "{}/../crates/inferno-formats/tests/fixtures/{p}",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// Validation must reject zero sizes before any measurement or llama-bench
/// lookup happens (so this test needs neither LLVM-compiled artifacts nor
/// a llama-bench binary).
#[test]
fn bench_rejects_zero_sizes() {
    for flag in ["--pp", "--tg", "--reps"] {
        Command::cargo_bin("inferno")
            .unwrap()
            .args(["bench", &fixture("tiny.gguf"), flag, "0"])
            .assert()
            .failure()
            .stderr(predicate::str::contains("must all be > 0"));
    }
}
```

Run: `cargo nextest run -p inferno bench_rejects`
Expected: PASS.

- [ ] **Step 6: Verify the command surface manually**

Run: `cargo run -p inferno -- bench --help`
Expected: help text shows the new flags with defaults (pp 512, tg 128, reps 5).

Run: `cargo run -p inferno -- bench /nonexistent.gguf`
Expected: clean `error: …` (from `measure_inferno`'s model load), exit code 1.

- [ ] **Step 7: Lint + full test + commit**

Run: `mise run lint && mise run test`
Expected: clean, all PASS.

```bash
git add cli/src/bench.rs cli/src/main.rs cli/src/snapshots cli/tests/bench.rs cli/Cargo.toml mise.toml
git commit -m "feat(cli): inferno bench — llama.cpp comparison report + mise task"
```

---

### Task 9: Protocol run — first data point + front-door note

**Files:**
- Modify: `docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md` (Amendments)
- Modify: `AGENTS.md` (one line)

**Interfaces:** none — this executes the manual protocol the spec defines.

**Hardware gate:** this task needs the quiet dev machine, the devenv shell, and the pinned nightly model (Qwen2.5-0.5B-Instruct Q8_0 — `scripts/nightly-speedup.sh` shows where it downloads from). **If you are not on real, quiet hardware with that model available (e.g. you are a sandboxed agent), do steps 1–2 only, then STOP and hand back to the human — never fabricate or estimate a data point** (AGENTS.md: perf numbers come only from real runs).

- [ ] **Step 1: Add the AGENTS.md line**

In `AGENTS.md`'s non-obvious-constraints list, after the `bench-kernels` bullet:

```markdown
- **`inferno bench` (vs llama.cpp) is a manual protocol**, never a CI gate:
  quiet hardware, devenv shell, release build; record each report in the
  M4a spec's Amendments section
  (`docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md`) and
  never edit a recorded data point.
```

- [ ] **Step 2: Commit the doc change**

```bash
git add AGENTS.md
git commit -m "docs: AGENTS.md note — inferno bench is a manual protocol"
```

- [ ] **Step 3 (quiet hardware only): Run the protocol**

Inside `devenv shell`, with the nightly model at `<MODEL>`:

```bash
mise run bench -- <MODEL>
mise run bench -- <MODEL> --json
```

Expected: the table renders with all rows (inferno, llama.cpp at physical cores, llama.cpp t=1) and the ratio line. If `llama-bench` rejects the model or the JSON parse fails, fix per Task 6's fixture-provenance caveat before recording anything.

- [ ] **Step 4 (quiet hardware only): Record the data point**

Append to the spec's `## Amendments` section: date, machine (CPU model, core counts), model file + quant, the full table output, and the `--json` blob in a fenced block. State plainly whether inferno wins or loses each of pp/tg — a loss is the expected M4a outcome and is the input to M4b's plan.

- [ ] **Step 5 (quiet hardware only): Commit**

```bash
git add docs/superpowers/specs/2026-07-06-m4a-bench-sampling-design.md
git commit -m "docs(spec): first inferno-vs-llama.cpp data point (M4a protocol)"
```
