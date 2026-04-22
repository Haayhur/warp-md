use thiserror::Error;
use traj_core::TrajError;

#[derive(Debug, Error)]
pub enum StructureError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("invalid structure data: {0}")]
    Invalid(String),
}

pub type StructureResult<T> = Result<T, StructureError>;

impl From<TrajError> for StructureError {
    fn from(err: TrajError) -> Self {
        match err {
            TrajError::Io(source) => Self::Io(source),
            TrajError::Parse(message) => Self::Parse(message),
            TrajError::Unsupported(message)
            | TrajError::Mismatch(message)
            | TrajError::InvalidSelection(message)
            | TrajError::Invalid(message) => Self::Invalid(message),
        }
    }
}

impl From<StructureError> for TrajError {
    fn from(err: StructureError) -> Self {
        match err {
            StructureError::Io(source) => Self::Io(source),
            StructureError::Parse(message) => Self::Parse(message),
            StructureError::Invalid(message) => Self::Invalid(message),
        }
    }
}
