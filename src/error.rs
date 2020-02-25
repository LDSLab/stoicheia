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
    #[error("misaligned axes: {0}")]
    MisalignedAxes(String),
    #[error("runtime error: {0}")]
    RuntimeError(&'static str),
    #[error("impossible error to handle infallible conversions")]
    ImpossibleError(#[from] std::convert::Infallible),
}

pub type Fallible<T> = Result<T, StoiError>;
