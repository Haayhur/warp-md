use thiserror::Error;
use traj_core::TrajError;

#[derive(Debug, Error)]
pub enum PackError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("invalid config: {0}")]
    Invalid(String),
    #[error("placement failed: {0}")]
    Placement(String),
}

pub type PackResult<T> = Result<T, PackError>;

impl From<TrajError> for PackError {
    fn from(err: TrajError) -> Self {
        match err {
            TrajError::Io(e) => PackError::Io(e),
            TrajError::Parse(msg) => PackError::Parse(msg),
            TrajError::Unsupported(msg) => PackError::Invalid(msg),
            TrajError::Mismatch(msg) => PackError::Invalid(msg),
            TrajError::InvalidSelection(msg) => PackError::Invalid(msg),
            TrajError::Invalid(msg) => PackError::Invalid(msg),
        }
    }
}
