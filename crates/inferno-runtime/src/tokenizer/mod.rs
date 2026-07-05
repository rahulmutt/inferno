pub(crate) mod bytes;
mod hf;
// bpe and spm modules join in Tasks 11–12.

use inferno_formats::TokenizerSpec;

use crate::{Result, RuntimeError};

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
        TokenizerSpec::Embedded { .. } => {
            // Native implementations land in Tasks 11–12.
            Err(RuntimeError::Tokenizer(
                "embedded tokenizers not yet wired".into(),
            ))
        }
    }
}
