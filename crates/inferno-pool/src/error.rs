use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PoolError {
    /// The process-global pool is sized once; a mismatched re-init is a
    /// caller bug (spec: error loudly, never silently reconfigure). Use
    /// `set_global_active_threads` to vary per-run parallelism instead.
    #[error(
        "thread pool already initialized with {current} threads (requested {requested}); \
         use set_global_active_threads to change per-dispatch parallelism"
    )]
    AlreadyInitialized { current: usize, requested: usize },
}
