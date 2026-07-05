use thiserror::Error;

#[derive(Debug, Error)]
pub enum TargetError {
    #[error("unknown target profile `{name}` (available: {available})")]
    UnknownProfile { name: String, available: String },
    #[error("malformed target profile `{name}`: {detail}")]
    MalformedProfile { name: String, detail: String },
    #[error("cannot detect hardware on this platform ({detail}); pass a named profile instead")]
    UnsupportedPlatform { detail: String },
    #[error("malformed sysfs data at {path}: {detail}")]
    MalformedSysfs { path: String, detail: String },
}

pub type Result<T> = std::result::Result<T, TargetError>;
