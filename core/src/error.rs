/*
    Appellation: error <module>
    Contrib: @FL03
*/
//! This module implements the core [`Error`] type for the framework and provides a [`Result`]
//! type alias for convenience.
#[cfg(feature = "alloc")]
use alloc::{
    boxed::Box,
    string::{String, ToString},
};

#[allow(dead_code)]
/// a type alias for a [Result](core::result::Result) configured with an [`Error`] as its error
/// type.
pub type Result<T> = core::result::Result<T, Error>;

/// The [`Error`] type enumerates various errors that can occur within the framework.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid Shape")]
    InvalidShape,
    #[error(transparent)]
    PadError(#[from] crate::utils::pad::PadError),
    #[error(transparent)]
    TraitError(#[from] concision_traits::Error),
    #[error(transparent)]
    ParamError(#[from] concision_params::ParamsError),
    #[error(transparent)]
    InitError(#[from] concision_init::InitError),
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
    #[cfg(feature = "alloc")]
    #[error(transparent)]
    BoxError(#[from] Box<dyn core::error::Error + Send + Sync>),
    #[cfg(feature = "serde")]
    #[error(transparent)]
    DeserializeError(#[from] serde::de::value::Error),
    #[error(transparent)]
    FmtError(#[from] core::fmt::Error),
    #[cfg(feature = "serde_json")]
    #[error(transparent)]
    JsonError(#[from] serde_json::Error),
    #[cfg(feature = "std")]
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[error(transparent)]
    #[cfg(feature = "rand")]
    UniformError(#[from] rand_distr::uniform::Error),
    #[cfg(feature = "alloc")]
    #[error("Unknown Error: {0}")]
    Unknown(String),
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
    #[cfg(feature = "alloc")]
    #[error("Parameter Error")]
    ParameterError(String),
}

#[cfg(feature = "alloc")]
impl Error {
    /// create a new [`BoxError`](Error::BoxError) variant using the given error
    pub fn box_error<E>(error: E) -> Self
    where
        E: 'static + Send + Sync + core::error::Error,
    {
        Self::BoxError(Box::new(error))
    }
    /// create a new [`Unknown`](Error::Unknown) variant using the given string
    pub fn unknown<S>(error: S) -> Self
    where
        S: ToString,
    {
        Self::Unknown(error.to_string())
    }
}

#[cfg(feature = "alloc")]
impl From<String> for Error {
    fn from(value: String) -> Self {
        Self::Unknown(value)
    }
}

#[cfg(feature = "alloc")]
impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Self::Unknown(String::from(value))
    }
}
