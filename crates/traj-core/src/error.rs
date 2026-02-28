use std::fmt;

#[derive(Debug)]
pub enum TrajError {
    Io(std::io::Error),
    Parse(String),
    Unsupported(String),
    Mismatch(String),
    InvalidSelection(String),
    Invalid(String),
}

impl fmt::Display for TrajError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrajError::Io(err) => write!(f, "io error: {err}"),
            TrajError::Parse(msg) => write!(f, "parse error: {msg}"),
            TrajError::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            TrajError::Mismatch(msg) => write!(f, "mismatch: {msg}"),
            TrajError::InvalidSelection(msg) => write!(f, "invalid selection: {msg}"),
            TrajError::Invalid(msg) => write!(f, "invalid: {msg}"),
        }
    }
}

impl std::error::Error for TrajError {}

impl From<std::io::Error> for TrajError {
    fn from(err: std::io::Error) -> Self {
        TrajError::Io(err)
    }
}

pub type TrajResult<T> = Result<T, TrajError>;
