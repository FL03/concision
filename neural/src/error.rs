/*
    Appellation: error <module>
    Contrib: @FL03
*/
//! this module defines the [`NeuralError`] type, used to define the various errors encountered
//! by the different components of a neural network. Additionally, the [`NeuralResult`] alias 
//! is defined for convenience, allowing for a more ergonomic way to handle results that may 
//! fail.
  
#[cfg(feature = "alloc")]
use alloc::{boxed::Box, string::{String, ToString}};

/// a type alias for a [Result](core::result::Result) configured to use the [`NeuralError`] 
/// implementation as its error type.
pub type NeuralResult<T> = core::result::Result<T, NeuralError>;

/// The [`NeuralError`] type is used to define the various errors encountered by the different
/// components of a neural network. It is designed to be comprehensive, covering a wide range of
/// potential issues that may arise during the operation of neural network components, such as
/// invalid configurations, training failures, and other runtime errors. This error type is
/// intended to provide a clear and consistent way to handle errors across the neural network
/// components, making it easier to debug and resolve issues that may occur during the development
/// and execution of neural network models.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NeuralError {
    #[error("Invalid Batch Size: {0}")]
    InvalidBatchSize(usize),
    #[error("Invalid Input Shape")]
    InvalidInputShape,
    #[error("Invalid Output Shape")]
    InvalidOutputShape,
    #[error(transparent)]
    TrainingError(#[from] TrainingError),
    #[error(transparent)]
    CoreError(#[from] concision_core::error::Error),
    #[cfg(feature = "alloc")]
    #[error("Parameter Error")]
    ParameterError(String),
}

#[derive(Debug, scsys_derive::VariantConstructors, thiserror::Error)]
pub enum TrainingError {
    #[error("Invalid Training Data")]
    InvalidTrainingData,
    #[error("Training Failed")]
    TrainingFailed,
}

impl From<NeuralError> for concision_core::error::Error {
    fn from(err: NeuralError) -> Self {
        match err {
            NeuralError::CoreError(e) => e,
            NeuralError::TrainingError(e) => e.into(),
            _ => concision_core::error::Error::box_error(err),
        }
    }
}

#[cfg(feature = "alloc")]
impl From<NeuralError> for Box<dyn core::error::Error + Send + Sync> {
    fn from(err: NeuralError) -> Self {
        Box::new(err)
    }
}

#[cfg(feature = "alloc")]
impl From<Box<dyn core::error::Error + Send + Sync>> for NeuralError {
    fn from(err: Box<dyn core::error::Error + Send + Sync>) -> Self {
        cnc::Error::BoxError(err)
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