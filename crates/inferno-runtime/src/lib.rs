//! Tokenizer, sampling, and the generation loop. In M1 the model executes
//! on the inferno-graph scalar interpreter; M3 swaps in compiled entry
//! points without moving this code.

mod error;
pub mod tokenizer;

pub use error::{Result, RuntimeError};
pub use tokenizer::{Tokenizer, tokenizer_for};
