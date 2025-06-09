/*
    Appellation: error <module>
    Contrib: @FL03
*/

/// a type alias for a [Result] with a [NeuralError]
pub type NeuralResult<T = ()> = core::result::Result<T, NeuralError>;

#[derive(Debug, scsys_derive::VariantConstructors, thiserror::Error)]
pub enum NeuralError {
    #[error("Invalid Batch Size")]
    InvalidBatchSize,
    #[error("Invalid Input Shape")]
    InvalidInputShape,
    #[error("Invalid Output Shape")]
    InvalidOutputShape,
    #[error("Parameter Error")]
    ParameterError(String),
    #[error("Training Failed")]
    TrainingFailed,
    #[error(transparent)]
    TrainingError(#[from] TrainingError),
    #[cfg(feature = "anyhow")]
    #[error(transparent)]
    AnyError(#[from] anyhow::Error),
    #[error(transparent)]
    BoxError(#[from] Box<dyn core::error::Error + Send + Sync>),
    #[error(transparent)]
    FmtError(#[from] core::fmt::Error),
    #[error(transparent)]
    CoreError(#[from] concision_core::error::Error),
}

#[derive(Debug, scsys_derive::VariantConstructors, thiserror::Error)]
pub enum TrainingError {
    #[error("Invalid Training Data")]
    InvalidTrainingData,
    #[error("Training Failed")]
    TrainingFailed,
    #[error(transparent)]
    CoreError(#[from] concision_core::error::Error),
}
