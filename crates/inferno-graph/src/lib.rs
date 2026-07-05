//! Graph IR, Llama-family builder, and the scalar reference interpreter —
//! the correctness oracle every compiled path is measured against.

mod error;
pub mod ir;
pub mod tolerance;

pub use error::{GraphError, Result};
pub use ir::{Dim, Graph, Node, Op, Shape, TensorRef, ValueId};
