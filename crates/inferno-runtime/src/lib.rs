//! Tokenizer, sampling, and the generation loop. In M1 the model executes
//! on the inferno-graph scalar interpreter; M3 swaps in compiled entry
//! points without moving this code.

mod error;
mod generate;
mod sampler;
pub mod tokenizer;

pub use error::{Result, RuntimeError};
pub use generate::{GenStats, Generator};
pub use sampler::{Greedy, Sampler};
pub use tokenizer::{Tokenizer, tokenizer_for};
