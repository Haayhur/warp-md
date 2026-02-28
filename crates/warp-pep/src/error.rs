//! Structured error type for warp-pep.

use std::fmt;

/// Warp-pep error categories.
#[derive(Debug)]
pub enum PepError {
    /// Invalid amino acid code or residue name.
    InvalidResidue(String),
    /// Missing atom needed for a calculation or build step.
    MissingAtom(String),
    /// Position out of range.
    OutOfRange(String),
    /// I/O failure (file read/write).
    Io(String),
    /// Invalid JSON input.
    Json(String),
    /// Invalid angle or geometry parameter.
    InvalidGeometry(String),
    /// Unsupported format or feature.
    Unsupported(String),
    /// Selection parse error.
    Selection(String),
    /// Validation failure (bond length, clash, etc.).
    Validation(String),
}

impl fmt::Display for PepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResidue(s) => write!(f, "invalid residue: {s}"),
            Self::MissingAtom(s) => write!(f, "missing atom: {s}"),
            Self::OutOfRange(s) => write!(f, "out of range: {s}"),
            Self::Io(s) => write!(f, "I/O error: {s}"),
            Self::Json(s) => write!(f, "JSON error: {s}"),
            Self::InvalidGeometry(s) => write!(f, "invalid geometry: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported: {s}"),
            Self::Selection(s) => write!(f, "selection error: {s}"),
            Self::Validation(s) => write!(f, "validation: {s}"),
        }
    }
}

impl std::error::Error for PepError {}

impl From<PepError> for String {
    fn from(e: PepError) -> String {
        e.to_string()
    }
}
