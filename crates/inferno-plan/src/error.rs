use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("no kernel for weight dtype {0:?} on the target ISA")]
    NoKernel(inferno_formats::DType),
    #[error("weight tensor {name} has {rank} dims, expected 2 ([n_out, k])")]
    BadWeightRank { name: String, rank: usize },
    #[error("kernel packing failed: {0}")]
    Pack(#[from] inferno_kernels::KernelError),
    #[error("reading weight bytes failed: {0}")]
    Formats(#[from] inferno_formats::FormatError),
    #[error("plan overflow: {0}")]
    Overflow(&'static str),
}

pub type Result<T> = std::result::Result<T, PlanError>;
