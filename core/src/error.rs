/*
    Appellation: error <module>
    Contrib: @FL03
*/
//! This module implements the core [`Error`] type for the framework and provides a [`Result`]
//! type alias for convenience.

/// a type alias for a [`Result`](core::result::Result) defined to use the custom [`Error`] as its error type.
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, string::String};
/// The [`Error`] type enumerates various errors that can occur within the framework.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[cfg(feature = "alloc")]
    #[error(transparent)]
    BoxError(#[from] Box<dyn core::error::Error + Send + Sync>),

    #[error("Invalid model configuration")]
    InvalidModelConfig,
    #[error("The model is not supported for the given input")]
    IncompatibleInput,
    #[error("Mismatched Dimension: expected {expected}, found {found}")]
    MismatchedDimension { expected: usize, found: usize },
    #[error("An invalid batch size was provided: {0}")]
    InvalidBatchSize(usize),
    #[error("Input is incompatible with the model: found {0} and expected {1}")]
    InvalidInputFeatures(usize, usize),
    #[error("The provided dataset has invalid target features: found {0} and expected {1}")]
    InvalidTargetFeatures(usize, usize),
    #[error("An uninitialized object was used")]
    Uninitialized,
    #[error("The model is not trained")]
    Untrained,
    #[cfg(feature = "alloc")]
    #[error("Unsupported model {0}")]
    UnsupportedModel(String),
    #[cfg(feature = "alloc")]
    #[cfg(feature = "alloc")]
    #[error("An unsupported operation was attempted: {0}")]
    UnsupportedOperation(String),
    #[error("Unknown Error: {0}")]
    Unknown(String),
    #[error("Parameter Error")]
    ParameterError(String),
    #[error(transparent)]
    AnyError(#[from] anyhow::Error),
    #[error(transparent)]
    PadError(#[from] crate::utils::pad::PadError),
    #[error(transparent)]
    ParamError(#[from] concision_params::ParamsError),
    #[error(transparent)]
    InitError(#[from] concision_init::InitError),
    #[error(transparent)]
    TraitError(#[from] concision_traits::Error),
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
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
}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Self::Unknown(value)
    }
}

impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Self::Unknown(String::from(value))
    }
}
