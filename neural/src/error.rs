/*
    Appellation: error <module>
    Contrib: @FL03
*/

/// a type alias for a [Result] with a [NeuralError]
pub type NeuralResult<T = ()> = core::result::Result<T, NeuralError>;

#[derive(Debug, thiserror::Error)]
pub enum NeuralError {
    #[error("Invalid Batch Size")]
    InvalidBatchSize,
    #[error("Invalid Input Shape")]
    InvalidInputShape,
    #[error("Invalid Output Shape")]
    InvalidOutputShape,
    #[error("Parameter Error")]
    ParameterError,
    #[error("Training Failed")]
    TrainingFailed(String),
    #[error(transparent)]
    CoreError(#[from] concision_core::error::Error),
}
