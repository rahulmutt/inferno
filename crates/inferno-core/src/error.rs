use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("format: {0}")]
    Format(#[from] inferno_formats::FormatError),
    #[error("graph: {0}")]
    Graph(#[from] inferno_graph::GraphError),
    #[error("codegen: {0}")]
    Codegen(#[from] inferno_codegen::CodegenError),
    #[error("target: {0}")]
    Target(#[from] inferno_target::TargetError),
    #[error("meta json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("dlopen/symbol: {0}")]
    Load(#[from] libloading::Error),
    #[error("mmap: {0}")]
    Mmap(std::io::Error),
    /// A cached artifact failed integrity verification (weights/model hash or
    /// version mismatch). Signalling value only: `load_or_compile` treats this
    /// as "discard and recompile", never propagating it to the caller.
    #[error("cache verification failed: {0}")]
    Verification(String),
    #[error("pool: {0}")]
    Pool(#[from] inferno_pool::PoolError),
    /// Surfaced by `Engine::profile_matmul_bytes` re-deriving the plan
    /// (pure, no LLVM) to map weight bytes onto profiler slots.
    #[error("plan: {0}")]
    Plan(#[from] inferno_plan::PlanError),
}

pub type Result<T> = std::result::Result<T, CoreError>;
