mod bpe;
pub(crate) mod bytes;
mod hf;
mod spm;

use inferno_formats::{TokenizerKind, TokenizerSpec};

use crate::Result;

pub trait Tokenizer: Send {
    fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>>;
    fn decode_token(&self, id: u32) -> Vec<u8>;
    fn bos(&self) -> Option<u32>;
    fn eos(&self) -> Option<u32>;
    fn default_add_bos(&self) -> bool;
}

pub fn tokenizer_for(spec: &TokenizerSpec) -> Result<Box<dyn Tokenizer>> {
    match spec {
        TokenizerSpec::HfJson { path } => Ok(Box::new(hf::HfTokenizer::load(path)?)),
        TokenizerSpec::Embedded {
            kind: TokenizerKind::Bpe,
            tokens,
            token_types,
            merges,
            pre,
            special,
            add_bos,
            ..
        } => Ok(Box::new(bpe::BpeTokenizer::new(
            tokens.clone(),
            token_types,
            merges,
            pre.as_deref(),
            special.clone(),
            *add_bos,
        )?)),
        TokenizerSpec::Embedded {
            kind: TokenizerKind::Spm,
            tokens,
            scores,
            token_types,
            special,
            add_bos,
            ..
        } => Ok(Box::new(spm::SpmTokenizer::new(
            tokens.clone(),
            scores.clone(),
            token_types.clone(),
            special.clone(),
            *add_bos,
        )?)),
    }
}
