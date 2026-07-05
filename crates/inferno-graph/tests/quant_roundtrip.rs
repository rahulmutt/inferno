//! Property tests: pack → dequant stays within the per-dtype tolerance
//! defined in inferno_graph::tolerance (the single home for these numbers).

use inferno_formats::{DType, quant};
use inferno_graph::tolerance::roundtrip_rel_tol;
use proptest::prelude::*;

fn check_roundtrip(dtype: &DType, vals: &[f32]) {
    let packed = quant::pack(dtype, vals).unwrap();
    let out = quant::dequant(dtype, &packed, vals.len()).unwrap();
    let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
    let tol = roundtrip_rel_tol(dtype) * amax;
    for (i, (a, b)) in vals.iter().zip(&out).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "{dtype:?}[{i}]: {a} vs {b} (tol {tol})"
        );
    }
}

proptest! {
    #[test]
    fn f16_roundtrip(vals in proptest::collection::vec(-100f32..100.0, 1..64)) {
        check_roundtrip(&DType::F16, &vals);
    }
    #[test]
    fn bf16_roundtrip(vals in proptest::collection::vec(-100f32..100.0, 1..64)) {
        check_roundtrip(&DType::BF16, &vals);
    }
    #[test]
    fn q8_0_roundtrip(vals in proptest::collection::vec(-10f32..10.0, 1..8)) {
        let vals: Vec<f32> = vals.into_iter().cycle().take(64).collect(); // 2 blocks
        check_roundtrip(&DType::Q8_0, &vals);
    }
    #[test]
    fn q4_k_roundtrip(vals in proptest::collection::vec(-10f32..10.0, 256..=512)) {
        let n = (vals.len() / 256) * 256;
        check_roundtrip(&DType::Q4_K, &vals[..n]);
    }
}
