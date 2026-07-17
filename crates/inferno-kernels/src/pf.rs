//! Compile-time parsing for the `INFERNO_PF_DIST` prefetch-distance
//! override (M4b.7 quiet-hardware sweep). `option_env!` makes the env var
//! a compilation input, so a rebuild with a different value re-evaluates
//! the kernel consts — no runtime cost at any value.

/// Parse a decimal `usize` at const-eval time. Panics — a compile error
/// in const context — on anything but a plain decimal integer.
pub(crate) const fn parse_pf_dist(s: &str) -> usize {
    let b = s.as_bytes();
    assert!(!b.is_empty(), "INFERNO_PF_DIST must be a decimal integer");
    let mut v = 0usize;
    let mut i = 0;
    while i < b.len() {
        assert!(
            b[i].is_ascii_digit(),
            "INFERNO_PF_DIST must be a decimal integer"
        );
        v = v * 10 + (b[i] - b'0') as usize;
        i += 1;
    }
    v
}

/// Compile-time parsing for the `INFERNO_GEMM_MR` register-tile width
/// (M4b.13 µbench sweep). 1..=16; pure loop restructuring, so output bits
/// never depend on the value.
pub(crate) const fn parse_gemm_mr(s: &str) -> usize {
    let b = s.as_bytes();
    assert!(!b.is_empty(), "INFERNO_GEMM_MR must be a decimal integer");
    let mut v = 0usize;
    let mut i = 0;
    while i < b.len() {
        assert!(
            b[i] >= b'0' && b[i] <= b'9',
            "INFERNO_GEMM_MR must be a decimal integer"
        );
        v = v * 10 + (b[i] - b'0') as usize;
        i += 1;
    }
    assert!(v >= 1 && v <= 16, "INFERNO_GEMM_MR must be in 1..=16");
    v
}

#[cfg(test)]
mod tests {
    use super::{parse_gemm_mr, parse_pf_dist};

    #[test]
    fn parses_decimal() {
        assert_eq!(parse_pf_dist("0"), 0);
        assert_eq!(parse_pf_dist("4"), 4);
        assert_eq!(parse_pf_dist("12"), 12);
    }

    #[test]
    #[should_panic(expected = "decimal integer")]
    fn rejects_empty() {
        parse_pf_dist("");
    }

    #[test]
    #[should_panic(expected = "decimal integer")]
    fn rejects_non_digit() {
        parse_pf_dist("4x");
    }

    #[test]
    fn gemm_mr_parses_decimal() {
        assert_eq!(parse_gemm_mr("4"), 4);
        assert_eq!(parse_gemm_mr("16"), 16);
    }

    #[test]
    #[should_panic(expected = "decimal integer")]
    fn gemm_mr_rejects_empty() {
        parse_gemm_mr("");
    }

    #[test]
    #[should_panic(expected = "decimal integer")]
    fn gemm_mr_rejects_non_digit() {
        parse_gemm_mr("4x");
    }

    #[test]
    #[should_panic(expected = "1..=16")]
    fn gemm_mr_rejects_out_of_range() {
        parse_gemm_mr("17");
    }
}
