//! Vectorized polynomial expf shared by the attention softmax. The scalar
//! lane and the AVX2 lane evaluate the *identical* constants and FMA order,
//! so a softmax built on them is bit-identical across ISAs (rig invariant).
//! Accuracy target: << 1 ULP-ish; the interpreter's std::exp stays the
//! ground truth, bounded by attn_rel_tol (see inferno-graph tolerance.rs).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// clippy::approx_constant (deny-by-default) flags the literal digit-for-digit
// as an approximation of this exact std constant; using it directly avoids
// the lint while producing the identical f32 bit pattern (0x3fb8aa3b).
// pub: inferno-codegen's attn_emit reads these to bake the identical
// polynomial into emitted IR (M4b.16). One source of truth — never copy
// the values into codegen.
pub const LOG2E: f32 = std::f32::consts::LOG2_E;
pub const LN2_HI: f32 = 0.693_359_4;
pub const LN2_LO: f32 = -2.121_944_4e-4;
// C0..C6: Taylor coefficients of exp(r) = 1 + r + r^2/2! + ... + r^6/6!,
// i.e. a degree-6 polynomial in r (7 terms, C[6] = 1/720). The brief's
// original 6-entry array (C0..C5) was only degree-5 despite the doc comment
// above calling for "degree-6"; that truncation left relative error above
// the 1e-6 test tolerance for ~12% of the -88..88 sweep (up to ~3.3e-6, e.g.
// x=-69.0). Adding the natural next term (1/720) restores the degree-6 the
// brief specifies and brings every swept point under the 1e-6 bound.
pub const C: [f32; 7] = [
    1.0,
    1.0,
    0.5,
    0.166_666_67,
    0.041_666_67,
    0.008_333_34,
    0.001_388_888_9,
];

#[inline]
pub(crate) fn expf_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    // round_ties_even matches the AVX2 _MM_FROUND_TO_NEAREST_INT rounding
    // mode exactly (round-half-to-even), keeping the two lanes bit-identical
    // even at exact half-boundaries of x*log2e.
    let n = (x * LOG2E).round_ties_even();
    let r = n.mul_add(-LN2_LO, n.mul_add(-LN2_HI, x));
    // Horner: p = C0 + r*(C1 + r*(C2 + ... )). C0==C1==1 => exp(r) series.
    let mut p = C[6];
    p = p.mul_add(r, C[5]);
    p = p.mul_add(r, C[4]);
    p = p.mul_add(r, C[3]);
    p = p.mul_add(r, C[2]);
    p = p.mul_add(r, C[1]);
    p = p.mul_add(r, C[0]);
    // scale by 2^n via exponent bits: (n as i32 + 127) << 23.
    let pow2n = f32::from_bits((((n as i32) + 127) << 23) as u32);
    p * pow2n
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn expf_avx2(x: __m256) -> __m256 {
    let x = _mm256_min_ps(
        _mm256_set1_ps(88.0),
        _mm256_max_ps(_mm256_set1_ps(-88.0), x),
    );
    // round-to-nearest-even matches f32::round_ties_even for the |x*log2e|
    // range here (n is an integer well under 2^23); use the AVX2 round
    // intrinsic.
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(LOG2E),
    ));
    let r = _mm256_fmadd_ps(
        n,
        _mm256_set1_ps(-LN2_LO),
        _mm256_fmadd_ps(n, _mm256_set1_ps(-LN2_HI), x),
    );
    let mut p = _mm256_set1_ps(C[6]);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[5]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[4]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[3]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[2]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[1]));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C[0]));
    // pow2n = ((n_i32 + 127) << 23) reinterpreted as f32, per lane.
    let ni = _mm256_cvtps_epi32(n);
    let bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(ni, _mm256_set1_epi32(127)));
    _mm256_mul_ps(p, _mm256_castsi256_ps(bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_matches_std_within_relative_1e_6() {
        for i in -880..=880 {
            let x = i as f32 * 0.1;
            let got = expf_scalar(x);
            let want = x.exp();
            let rel = (got - want).abs() / want.max(1e-30);
            assert!(rel <= 1e-6, "x={x}: got {got}, want {want}, rel {rel}");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_lane_is_bitwise_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let xs = [-88.0f32, -3.3, -0.7, 0.0, 0.2, 1.5, 11.0, 87.9];
        // SAFETY: avx2 detected above.
        let out = unsafe {
            let v = _mm256_loadu_ps(xs.as_ptr());
            let r = expf_avx2(v);
            let mut o = [0f32; 8];
            _mm256_storeu_ps(o.as_mut_ptr(), r);
            o
        };
        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(out[i].to_bits(), expf_scalar(x).to_bits(), "lane {i} x={x}");
        }
    }
}
