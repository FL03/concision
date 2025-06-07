

pub(crate) type ModelResult<T> = core::result::Result<T, ModelError>;

#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum ModelError {
    /// The model is not initialized
    #[error("The model is not initialized")]
    NotInitialized,
    #[error("The model is not trained")]
    /// The model is not trained
    NotTrained,
    #[error("Invalid model configuration")]
    /// The model is not valid
    InvalidModelConfig,
    #[error("Unsupported model")]
    /// The model is not supported
    UnsupportedModel,
    #[error("The model is not supported for the given input")]
    /// The model is not compatible with the given input
    IncompatibleInput,
    /// The model is not compatible with the given output
    IncompatibleOutput,
}