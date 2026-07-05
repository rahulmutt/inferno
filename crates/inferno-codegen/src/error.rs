use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("planning failed: {0}")]
    Plan(#[from] inferno_plan::PlanError),
    #[error("LLVM: {0}")]
    Llvm(String),
    #[error("object emission failed: {0}")]
    Emit(String),
    #[error("linker failed: {0}")]
    Link(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CodegenError>;
