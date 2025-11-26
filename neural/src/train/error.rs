#[allow(dead_code)]
/// a type alias for a [`Result`](core::result::Result) with an error type of
/// [`TrainingError`].
pub(crate) type TrainingResult<T> = Result<T, TrainingError>;

/// The [`TrainingError`] type enumerates the various errors that can occur during the
/// training process.
#[derive(Debug, thiserror::Error, variants::VariantConstructors)]
#[non_exhaustive]
pub enum TrainingError {
    #[error("Invalid Training Data")]
    InvalidTrainingData,
    #[error("Training Failed")]
    TrainingFailed,
}
