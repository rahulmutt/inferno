/// Errors from building or interpreting the graph IR.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("tensor {name}: expected shape {expected:?}, got {got:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<u64>,
        got: Vec<u64>,
    },
    #[error("invalid hyperparameters: {0}")]
    BadHyperParams(String),
    #[error("token id {id} out of range (vocab {vocab})")]
    TokenOutOfRange { id: u32, vocab: usize },
    #[error("sequence length {got} exceeds capacity {max}")]
    SeqTooLong { got: usize, max: usize },
    #[error(transparent)]
    Format(#[from] inferno_formats::FormatError),
}

pub type Result<T> = std::result::Result<T, GraphError>;
