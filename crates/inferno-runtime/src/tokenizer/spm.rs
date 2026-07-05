//! Native SentencePiece-style tokenizer (score-driven greedy bigram merging,
//! llama.cpp's SPM semantics) for GGUF-embedded Llama-2/Mistral vocabs.

use std::collections::HashMap;

use inferno_formats::SpecialTokens;

use crate::tokenizer::bytes::spm_token_to_bytes;
use crate::{Result, RuntimeError};

const GGUF_TOKEN_CONTROL: i32 = 3;

pub(crate) struct SpmTokenizer {
    tokens: Vec<String>,
    vocab: HashMap<String, u32>,
    scores: Vec<f32>,
    token_types: Vec<i32>,
    special: SpecialTokens,
    add_bos: bool,
}

impl SpmTokenizer {
    pub(crate) fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        token_types: Vec<i32>,
        special: SpecialTokens,
        add_bos: bool,
    ) -> Result<SpmTokenizer> {
        if scores.len() != tokens.len() {
            return Err(RuntimeError::Tokenizer(format!(
                "spm vocab has {} tokens but {} scores",
                tokens.len(),
                scores.len()
            )));
        }
        let vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        Ok(SpmTokenizer {
            vocab,
            scores,
            token_types,
            special,
            add_bos,
            tokens,
        })
    }

    fn is_special(&self, id: u32) -> bool {
        self.token_types.get(id as usize) == Some(&GGUF_TOKEN_CONTROL)
    }
}

impl crate::Tokenizer for SpmTokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if add_bos && let Some(b) = self.special.bos {
            ids.push(b);
        }
        if text.is_empty() {
            return Ok(ids);
        }
        // SPM whitespace escaping + add_space_prefix (llama.cpp default).
        let escaped = format!("\u{2581}{}", text.replace(' ', "\u{2581}"));
        // Symbols start as single characters.
        let mut syms: Vec<String> = escaped.chars().map(String::from).collect();
        // Greedy: always merge the adjacent pair with the highest score.
        loop {
            let best = syms
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    let cat = format!("{}{}", w[0], w[1]);
                    self.vocab
                        .get(&cat)
                        .map(|&id| (self.scores[id as usize], i))
                })
                .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
            match best {
                Some((_, i)) => {
                    let merged = format!("{}{}", syms[i], syms[i + 1]);
                    syms.splice(i..=i + 1, [merged]);
                }
                None => break,
            }
        }
        for s in &syms {
            match self.vocab.get(s) {
                Some(&id) => ids.push(id),
                None => {
                    // Byte fallback: emit <0xNN> per UTF-8 byte.
                    for b in s.bytes() {
                        match self.vocab.get(&format!("<0x{b:02X}>")) {
                            Some(&id) => ids.push(id),
                            None => {
                                return Err(RuntimeError::Tokenizer(format!(
                                    "no vocab entry or byte fallback for {s:?}"
                                )));
                            }
                        }
                    }
                }
            }
        }
        Ok(ids)
    }

    fn decode_token(&self, id: u32) -> Vec<u8> {
        let Some(tok) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        if self.is_special(id) {
            return tok.as_bytes().to_vec();
        }
        spm_token_to_bytes(tok)
    }

    fn bos(&self) -> Option<u32> {
        self.special.bos
    }
    fn eos(&self) -> Option<u32> {
        self.special.eos
    }
    fn default_add_bos(&self) -> bool {
        self.add_bos
    }
}

#[cfg(test)]
mod tests {
    use inferno_formats::{SpecialTokens, TokenizerKind, TokenizerSpec};

    /// Hand-built SPM vocab: byte-fallback tokens 0..=255 as <0xNN> (type 6),
    /// then ▁(256), h(257), e(258), he(259, score 1.0), ▁he(260, score 2.0),
    /// <s>(261, control), </s>(262, control).
    fn spm() -> Box<dyn crate::Tokenizer> {
        let mut tokens: Vec<String> = (0u16..256).map(|b| format!("<0x{b:02X}>")).collect();
        let mut token_types = vec![6i32; 256];
        let mut scores = vec![0f32; 256];
        for (t, ty, sc) in [
            ("\u{2581}", 1, 0.0),
            ("h", 1, 0.0),
            ("e", 1, 0.0),
            ("he", 1, 1.0),
            ("\u{2581}he", 1, 2.0),
            ("<s>", 3, 0.0),
            ("</s>", 3, 0.0),
        ] {
            tokens.push(t.into());
            token_types.push(ty);
            scores.push(sc);
        }
        crate::tokenizer_for(&TokenizerSpec::Embedded {
            kind: TokenizerKind::Spm,
            tokens,
            scores,
            token_types,
            merges: vec![],
            pre: None,
            special: SpecialTokens {
                bos: Some(261),
                eos: Some(262),
            },
            add_bos: true,
        })
        .unwrap()
    }

    #[test]
    fn merges_by_score_with_space_prefix() {
        // "he" → "▁he" after prefixing → single token 260 (score 2.0 beats
        // merging h+e first).
        assert_eq!(spm().encode("he", false).unwrap(), vec![260]);
    }

    #[test]
    fn unknown_chars_use_byte_fallback() {
        // "Z" is not in the vocab → ▁ then <0x5A>.
        assert_eq!(spm().encode("Z", false).unwrap(), vec![256, 0x5A]);
    }

    #[test]
    fn add_bos_default_is_true_for_spm() {
        let t = spm();
        assert!(t.default_add_bos());
        assert_eq!(t.encode("he", true).unwrap(), vec![261, 260]);
    }

    #[test]
    fn decode_restores_text() {
        let t = spm();
        let ids = t.encode("he he", false).unwrap();
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| t.decode_token(i)).collect();
        // SPM's leading ▁ decodes to a leading space; strip for comparison.
        assert_eq!(String::from_utf8(bytes).unwrap().trim_start(), "he he");
    }
}
