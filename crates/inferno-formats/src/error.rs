/// Errors from parsing model files. Every variant is a *rejection* of
/// untrusted input, not a panic — parsers must be total over arbitrary bytes.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a recognized model file: {0}")]
    UnknownFormat(String),
    #[error("bad magic bytes (expected {expected})")]
    BadMagic { expected: &'static str },
    #[error("unsupported gguf version {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),
    #[error("malformed {context}: {detail}")]
    Malformed {
        context: &'static str,
        detail: String,
    },
    #[error("{what} ({got}) exceeds limit ({limit})")]
    LimitExceeded {
        what: &'static str,
        got: u64,
        limit: u64,
    },
    #[error("missing required metadata: {0}")]
    MissingKey(String),
    #[error("invalid json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, FormatError>;
