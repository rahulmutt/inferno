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

#[cfg(test)]
mod tests {
    use super::parse_pf_dist;

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
}
