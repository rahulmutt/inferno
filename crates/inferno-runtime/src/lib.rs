//! Tokenizer, sampling, and the generation loop. In M1 the model executes
//! on the inferno-graph scalar interpreter; M3 swaps in compiled entry
//! points without moving this code.

pub mod backend;
mod diff;
mod error;
mod generate;
mod rng;
mod sampler;
pub mod tokenizer;

pub use backend::{Backend, InterpBackend};
pub use diff::{DiffOutcome, Mismatch, teacher_forced};
pub use error::{Result, RuntimeError};
pub use generate::{GenStats, Generator};
pub use sampler::{Greedy, Sampler};
pub use tokenizer::{Tokenizer, tokenizer_for};
