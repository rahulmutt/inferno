//! tokenizer.json via the `tokenizers` crate (MLX models). Also the
//! property-test reference for the native BPE implementation.

use std::path::Path;

use crate::tokenizer::bytes::bpe_token_to_bytes;
use crate::{Result, RuntimeError};

pub(crate) struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    bos: Option<u32>,
    eos: Option<u32>,
}

impl HfTokenizer {
    pub(crate) fn load(path: &Path) -> Result<HfTokenizer> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        // Conventional special-token names; absent → None (MLX configs vary,
        // and generation just won't auto-stop — CLI max-tokens still bounds it).
        let find = |names: &[&str]| names.iter().find_map(|n| inner.token_to_id(n));
        Ok(HfTokenizer {
            bos: find(&["<|bos|>", "<s>", "<|begin_of_text|>"]),
            eos: find(&[
                "<|eos|>",
                "</s>",
                "<|end_of_text|>",
                "<|im_end|>",
                "<|endoftext|>",
            ]),
            inner,
        })
    }
}

impl crate::Tokenizer for HfTokenizer {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| RuntimeError::Tokenizer(e.to_string()))?;
        let mut ids = Vec::new();
        if add_bos && let Some(b) = self.bos {
            ids.push(b);
        }
        ids.extend_from_slice(enc.get_ids());
        Ok(ids)
    }

    fn decode_token(&self, id: u32) -> Vec<u8> {
        match self.inner.id_to_token(id) {
            // Byte-level BPE token → exact bytes via the shared table.
            Some(tok) => bpe_token_to_bytes(&tok)
                .unwrap_or_else(|| crate::tokenizer::bytes::spm_token_to_bytes(&tok)),
            None => Vec::new(),
        }
    }

    fn bos(&self) -> Option<u32> {
        self.bos
    }
    fn eos(&self) -> Option<u32> {
        self.eos
    }
    fn default_add_bos(&self) -> bool {
        false // HF post-processors handle BOS when configured; fixture/Qwen don't add one
    }
}

#[cfg(test)]
mod tests {
    use inferno_formats::TokenizerSpec;
    use std::path::Path;

    fn fixture_tokenizer() -> Box<dyn crate::Tokenizer> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../inferno-formats/tests/fixtures/mlx/tokenizer.json");
        crate::tokenizer_for(&TokenizerSpec::HfJson { path }).unwrap()
    }

    #[test]
    fn encodes_with_merges() {
        let t = fixture_tokenizer();
        // "the" merges via "t h"→"th", "th e"→"the" → single token 259.
        assert_eq!(t.encode("the", false).unwrap(), vec![259]);
        // "cat" has no merges → three byte tokens.
        assert_eq!(
            t.encode("cat", false).unwrap(),
            vec![u32::from(b'c'), u32::from(b'a'), u32::from(b't')]
        );
    }

    #[test]
    fn decode_token_returns_bytes() {
        let t = fixture_tokenizer();
        assert_eq!(t.decode_token(259), b"the");
        assert_eq!(t.decode_token(u32::from(b' ')), b" ");
    }
}
