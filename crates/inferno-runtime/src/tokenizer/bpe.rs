//! Native byte-level BPE over GGUF-embedded vocab (merge-rank driven, GPT-2
//! byte↔unicode alphabet, fancy-regex pre-tokenization).

use std::collections::HashMap;

use inferno_formats::SpecialTokens;

use crate::tokenizer::bytes::{bpe_token_to_bytes, byte_to_unicode};
use crate::{Result, RuntimeError};

const GGUF_TOKEN_CONTROL: i32 = 3;
const GGUF_TOKEN_USER_DEFINED: i32 = 4;

fn pre_pattern(pre: Option<&str>) -> Result<&'static str> {
    match pre.unwrap_or("default") {
        "default" | "gpt-2" => {
            Ok(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        }
        "llama-bpe" | "llama3" => Ok(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ),
        "qwen2" => Ok(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ),
        other => Err(RuntimeError::Tokenizer(format!(
            "unsupported pre-tokenizer id {other:?} — add its pattern to bpe.rs"
        ))),
    }
}

pub(crate) struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    tokens: Vec<String>,
    ranks: HashMap<(String, String), usize>,
    /// Control/user-defined tokens, longest first, matched literally.
    specials: Vec<(String, u32)>,
    pre: fancy_regex::Regex,
    special: SpecialTokens,
    add_bos: bool,
}

impl BpeTokenizer {
    pub(crate) fn new(
        tokens: Vec<String>,
        token_types: &[i32],
        merges: &[String],
        pre: Option<&str>,
        special: SpecialTokens,
        add_bos: bool,
    ) -> Result<BpeTokenizer> {
        let pre = fancy_regex::Regex::new(pre_pattern(pre)?)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        let vocab: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let ranks = merges
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                m.split_once(' ')
                    .map(|(a, b)| ((a.to_string(), b.to_string()), i))
            })
            .collect();
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                matches!(
                    token_types.get(*i),
                    Some(&GGUF_TOKEN_CONTROL) | Some(&GGUF_TOKEN_USER_DEFINED)
                )
            })
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));
        Ok(BpeTokenizer {
            vocab,
            tokens,
            ranks,
            specials,
            pre,
            special,
            add_bos,
        })
    }

    /// One pre-tokenized piece → token ids via rank-ordered pair merging.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) -> Result<()> {
        let table = byte_to_unicode();
        let mut syms: Vec<String> = piece
            .bytes()
            .map(|b| table[b as usize].to_string())
            .collect();
        loop {
            let best = syms
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    self.ranks
                        .get(&(w[0].clone(), w[1].clone()))
                        .map(|&r| (r, i))
                })
                .min();
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
                Some(&id) => out.push(id),
                None => {
                    // Unmergeable multi-char symbol without a vocab entry can
                    // only happen with an inconsistent vocab/merges pair.
                    return Err(RuntimeError::Tokenizer(format!(
                        "symbol {s:?} not in vocab (inconsistent merges)"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Split text on literal special-token occurrences (longest match wins).
    fn split_specials<'a>(&'a self, text: &'a str) -> Vec<(bool, &'a str, u32)> {
        let mut parts = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            let mut earliest: Option<(usize, &(String, u32))> = None;
            for sp in &self.specials {
                if let Some(pos) = rest.find(&sp.0)
                    && earliest
                        .is_none_or(|(e, cur)| pos < e || (pos == e && sp.0.len() > cur.0.len()))
                {
                    earliest = Some((pos, sp));
                }
            }
            match earliest {
                Some((pos, (tok, id))) => {
                    if pos > 0 {
                        parts.push((false, &rest[..pos], 0));
                    }
                    parts.push((true, tok.as_str(), *id));
                    rest = &rest[pos + tok.len()..];
                }
                None => {
                    parts.push((false, rest, 0));
                    break 'outer;
                }
            }
        }
        parts
    }
}

impl crate::Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        if add_bos && let Some(b) = self.special.bos {
            ids.push(b);
        }
        for (is_special, chunk, id) in self.split_specials(text) {
            if is_special {
                ids.push(id);
                continue;
            }
            let mut at = 0;
            while at < chunk.len() {
                let m = self
                    .pre
                    .find_from_pos(chunk, at)
                    .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
                let Some(m) = m else { break };
                self.encode_piece(&chunk[m.start()..m.end()], &mut ids)?;
                at = m.end();
            }
        }
        Ok(ids)
    }

    fn decode_token(&self, id: u32) -> Vec<u8> {
        let Some(tok) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        if self.specials.iter().any(|(_, sid)| *sid == id) {
            return tok.as_bytes().to_vec(); // specials pass through literally
        }
        bpe_token_to_bytes(tok).unwrap_or_else(|| tok.as_bytes().to_vec())
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
    use inferno_formats::fixtures;
    use inferno_formats::{SpecialTokens, TokenizerKind, TokenizerSpec};

    fn native() -> Box<dyn crate::Tokenizer> {
        let (tokens, merges) = fixtures::tiny_vocab();
        let mut token_types = vec![1i32; 256];
        token_types.extend([3, 3, 1, 1]);
        crate::tokenizer_for(&TokenizerSpec::Embedded {
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

    fn hf() -> Box<dyn crate::Tokenizer> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../inferno-formats/tests/fixtures/mlx/tokenizer.json");
        crate::tokenizer_for(&TokenizerSpec::HfJson { path }).unwrap()
    }

    #[test]
    fn merges_apply_in_rank_order() {
        let t = native();
        assert_eq!(t.encode("the", false).unwrap(), vec![259]);
        assert_eq!(t.encode("th", false).unwrap(), vec![258]);
        // " the": pre-tokenized as one piece "Ġthe"; no merge with Ġ exists
        // → [Ġ, the] after t+h, th+e merges.
        assert_eq!(t.encode(" the", false).unwrap(), vec![u32::from(b' '), 259]);
    }

    #[test]
    fn special_tokens_split_literally() {
        let t = native();
        let ids = t.encode("<|bos|>the", false).unwrap();
        assert_eq!(ids, vec![256, 259]);
    }

    #[test]
    fn add_bos_prepends() {
        let t = native();
        assert_eq!(t.encode("the", true).unwrap(), vec![256, 259]);
    }

    #[test]
    fn decode_roundtrips_bytes() {
        let t = native();
        for text in ["the cat", "héllo\nworld", "  spaces  "] {
            let ids = t.encode(text, false).unwrap();
            let bytes: Vec<u8> = ids.iter().flat_map(|&i| t.decode_token(i)).collect();
            assert_eq!(bytes, text.as_bytes(), "{text:?}");
        }
    }

    #[test]
    fn matches_hf_reference_on_fixture_vocab() {
        let (n, h) = (native(), hf());
        for text in ["the", "th the then", "a\nb", "unrelated words", "  the  "] {
            assert_eq!(
                n.encode(text, false).unwrap(),
                h.encode(text, false).unwrap(),
                "{text:?}"
            );
        }
    }

    #[test]
    fn unsupported_pre_id_is_loud_error() {
        let (tokens, merges) = fixtures::tiny_vocab();
        let r = crate::tokenizer_for(&TokenizerSpec::Embedded {
            kind: TokenizerKind::Bpe,
            tokens,
            scores: vec![],
            token_types: vec![],
            merges,
            pre: Some("some-future-model".into()),
            special: SpecialTokens::default(),
            add_bos: false,
        });
        assert!(matches!(r, Err(crate::RuntimeError::Tokenizer(_))));
    }
}
