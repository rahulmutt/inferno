//! Native BPE must agree with the HF `tokenizers` crate on the same vocab
//! (spec §Testing). ASCII-focused generator plus unicode spot checks — the
//! fixture vocab covers all 256 bytes, so any disagreement is a merge/pre-
//! tokenizer bug, not a coverage artifact.

use std::path::Path;

use inferno_formats::fixtures;
use inferno_formats::{SpecialTokens, TokenizerKind, TokenizerSpec};
use proptest::prelude::*;

fn native() -> Box<dyn inferno_runtime::Tokenizer> {
    let (tokens, merges) = fixtures::tiny_vocab();
    let mut token_types = vec![1i32; 256];
    token_types.extend([3, 3, 1, 1]);
    inferno_runtime::tokenizer_for(&TokenizerSpec::Embedded {
        kind: TokenizerKind::Bpe,
        tokens,
        scores: vec![],
        token_types,
        merges,
        pre: Some("default".into()),
        special: SpecialTokens {
            bos: Some(256),
            eos: Some(257),
        },
        add_bos: false,
    })
    .unwrap()
}

fn hf() -> Box<dyn inferno_runtime::Tokenizer> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../inferno-formats/tests/fixtures/mlx/tokenizer.json");
    inferno_runtime::tokenizer_for(&TokenizerSpec::HfJson { path }).unwrap()
}

proptest! {
    #[test]
    fn native_bpe_matches_hf(text in "[ -~\\n\\t]{0,64}") {
        let (n, h) = (native(), hf());
        prop_assert_eq!(n.encode(&text, false).unwrap(), h.encode(&text, false).unwrap());
    }

    #[test]
    fn native_bpe_roundtrips_arbitrary_unicode(text in "\\PC{0,32}") {
        let n = native();
        let ids = n.encode(&text, false).unwrap();
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| n.decode_token(i)).collect();
        prop_assert_eq!(bytes, text.as_bytes());
    }
}
