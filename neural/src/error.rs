/*
    Appellation: error <module>
    Contrib: @FL03
*/
//! this module defines the [`ModelError`] type, used to define the various errors encountered
//! by the different components of a neural network. Additionally, the [`ModelResult`] alias
//! is defined for convenience, allowing for a more ergonomic way to handle results that may
//! fail.

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, string::String};

#[allow(deprecated)]
#[deprecated(since = "0.2.8", note = "use `NeuralResult` instead")]
pub type ModelResult<T> = core::result::Result<T, ModelError>;
#[deprecated(since = "0.2.8", note = "use `NeuralError` instead")]
pub type ModelError = NeuralError;

/// a type alias for a [Result](core::result::Result) configured to use the [`ModelError`]
/// implementation as its error type.
pub type NeuralResult<T> = core::result::Result<T, NeuralError>;

/// The [`ModelError`] type is used to define the various errors encountered by the different
/// components of a neural network. It is designed to be comprehensive, covering a wide range of
/// potential issues that may arise during the operation of neural network components, such as
/// invalid configurations, training failures, and other runtime errors. This error type is
/// intended to provide a clear and consistent way to handle errors across the neural network
/// components, making it easier to debug and resolve issues that may occur during the development
/// and execution of neural network models.
#[derive(Debug, variants::VariantConstructors, thiserror::Error)]
#[non_exhaustive]
pub enum NeuralError {
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
    #[error("An unsupported operation was attempted")]
    UnsupportedOperation,
    #[error("Invalid Batch Size")]
    InvalidBatchSize,
    #[error("Invalid Input Shape")]
    InvalidInputShape,
    #[error("Invalid Output Shape")]
    InvalidOutputShape,
    #[error(transparent)]
    TrainingError(#[from] crate::train::TrainingError),
    #[error(transparent)]
    CoreError(#[from] concision_core::error::Error),
    #[cfg(feature = "alloc")]
    #[error("Parameter Error")]
    ParameterError(String),
}

impl From<NeuralError> for concision_core::error::Error {
    fn from(err: NeuralError) -> Self {
        match err {
            NeuralError::CoreError(e) => e,
            _ => concision_core::error::Error::box_error(err),
        }
    }
}

#[cfg(feature = "alloc")]
impl From<Box<dyn core::error::Error + Send + Sync>> for NeuralError {
    fn from(err: Box<dyn core::error::Error + Send + Sync>) -> Self {
        cnc::Error::BoxError(err).into()
    }
}
#[cfg(feature = "alloc")]
impl From<String> for NeuralError {
    fn from(err: String) -> Self {
        cnc::Error::unknown(err).into()
    }
}

#[cfg(feature = "alloc")]
impl From<&str> for NeuralError {
    fn from(err: &str) -> Self {
        cnc::Error::unknown(err).into()
    }
}
