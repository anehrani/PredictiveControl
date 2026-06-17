use thiserror::Error as ThisError;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("dimension mismatch: {context} expected {expected}, got {got}")]
    DimensionMismatch {
        context: &'static str,
        expected: String,
        got: String,
    },

    #[error("matrix is not positive definite: {0}")]
    NotPositiveDefinite(&'static str),

    #[error("probability must be in (0, 1), got {0}")]
    InvalidProbability(f64),

    #[error("parameter must be positive: {name} = {value}")]
    NonPositiveParameter { name: &'static str, value: f64 },

    #[error("empty input: {0}")]
    Empty(&'static str),

    #[error("linear solve failed: {0}")]
    LinearSolve(&'static str),

    #[error("unsupported polynomial family for this operation: {0}")]
    UnsupportedPolynomialFamily(&'static str),
}

pub(crate) fn dim_error(
    context: &'static str,
    expected: impl Into<String>,
    got: impl Into<String>,
) -> Error {
    Error::DimensionMismatch {
        context,
        expected: expected.into(),
        got: got.into(),
    }
}
