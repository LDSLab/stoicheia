use thiserror::Error;

#[derive(Error, Debug)]
pub enum StoiError {
    #[error("SQLite storage error")]
    SQLiteError(#[from] rusqlite::Error),
    #[error("Bincode serialization error")]
    BincodeError(#[from] bincode::Error),
    #[error("Json serialization error")]
    JsonError(#[from] serde_json::Error),
    #[error("no record found for the {0} {1}")]
    NotFound(&'static str, String),
    #[error("resource request is too large: {0}")]
    TooLarge(&'static str),
    #[error("invalid value: {0}")]
    InvalidValue(&'static str),
    #[error("invalid value (expected {expected:?}, found {found:?})")]
    InvalidHeader { expected: String, found: String },
    #[error("unknown data store error")]
    Unknown,
}

pub type Fallible<T> = Result<T, StoiError>;
