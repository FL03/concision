/*
    Appellation: error <module>
    Contrib: @FL03
*/
//! This module implements the core [`Error`] type for the framework and provides a [`Result`]
//! type alias for convenience.

/// a type alias for a [`Result`](core::result::Result) defined to use the custom [`Error`] as its error type.
pub type Result<T> = core::result::Result<T, Error>;

/// The [`Error`] type enumerates various errors that can occur within the framework.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[cfg(feature = "alloc")]
    #[error(transparent)]
    AllocError(#[from] alloc_err::AllocError),
    #[error(transparent)]
    ExtError(#[from] CommonError),
    #[error("The model is not trained")]
    NotTrained,
    #[error("Invalid model configuration")]
    InvalidModelConfig,
    #[error("The model is not supported for the given input")]
    IncompatibleInput,
    #[error("An unsupported operation was attempted")]
    UnsupportedOperation,
    #[error("Invalid Batch Size")]
    InvalidBatchSize,
    #[error("Invalid Input Shape")]
    InvalidInputShape,
    #[error("Invalid Output Shape")]
    InvalidOutputShape,
    #[error("Uninitialized")]
    Uninitialized,
    #[error("Unsupported model {0}")]
    UnsupportedModel(&'static str),
    #[error("Parameter Error")]
    ParameterError(&'static str),
}

/// The [`CommonError`] type enumerates external errors handled by the framework
#[derive(Debug, thiserror::Error)]
pub enum CommonError {
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
}

#[cfg(feature = "alloc")]
mod alloc_err {
    use alloc::{boxed::Box, string::String};

    #[derive(Debug, thiserror::Error)]
    pub enum AllocError {
        #[error(transparent)]
        BoxError(#[from] Box<dyn core::error::Error + Send + Sync>),
        #[error("Unknown Error: {0}")]
        Unknown(String),
    }

    impl From<String> for AllocError {
        fn from(value: String) -> Self {
            Self::Unknown(value)
        }
    }

    impl From<&str> for AllocError {
        fn from(value: &str) -> Self {
            Self::Unknown(String::from(value))
        }
    }
}
