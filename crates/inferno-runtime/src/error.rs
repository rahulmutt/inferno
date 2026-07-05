/// Errors from tokenization, prompt handling, and the generation loop.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("model has no usable tokenizer metadata")]
    NoTokenizer,
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("prompt ({got} tokens) exceeds max sequence length ({max})")]
    PromptTooLong { got: usize, max: usize },
    #[error(transparent)]
    Graph(#[from] inferno_graph::GraphError),
    #[error(transparent)]
    Format(#[from] inferno_formats::FormatError),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, RuntimeError>;
