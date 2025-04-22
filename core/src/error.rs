/*
    Appellation: error <module>
    Contrib: @FL03
*/

/// a type alias for a [Result] with a [Error]
pub(crate) type Result<T = ()> = core::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Dimension Error: {0}")]
    DimensionalError(String),
    #[error("Infinity Error")]
    InfinityError,
    #[error("Invalid Batch Size")]
    InvalidBatchSize,
    #[error("Invalid Input Shape")]
    InvalidInputShape,
    #[error("Invalid Output Shape")]
    InvalidOutputShape,
    #[error("Invalid Shape: {0}")]
    InvalidShape(String),
    #[error("Invalid Shape Mismatch: {0:?} != {1:?}")]
    ShapeMismatch(Vec<usize>, Vec<usize>),
    #[error("NaN Error")]
    NaNError,
    #[error("Parameter Error")]
    ParameterError,
    #[error("Training Failed")]
    TrainingFailed(String),
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
    #[cfg(feature = "anyhow")]
    #[error(transparent)]
    Other(#[from] anyhow::Error),
    #[error("Unknown Error: {0}")]
    Unknown(String),
}
