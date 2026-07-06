//! Seedable deterministic RNG for sampling: xoshiro256** seeded via
//! splitmix64 (the reference procedure). Hand-rolled on purpose — `rand`'s
//! `SmallRng` documents that its algorithm may change between crate
//! versions, which would silently break the exact-pick sampler tests.

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

    #[allow(dead_code)] // consumed in Task 4
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
    #[allow(dead_code)] // consumed in Task 4
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

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
