/*
    Appellation: error <module>
    Contrib: @FL03
*/
//! This module implements the core [`TensorError`] type for the framework and provides a
//! [`Result`] type alias for convenience.
#[cfg(feature = "alloc")]
use alloc::{
    boxed::Box,
    string::{String, ToString},
};
#[allow(dead_code)]
/// a type alias for a [`Result`](core::result::Result) configured with the [`TensorError`] as
/// its error type.
pub(crate) type Result<T> = core::result::Result<T, TensorError>;

/// The [`Error`] type enumerates various errors that can occur within the framework.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
    #[error(transparent)]
    #[cfg(feature = "cnc_init")]
    InitError(#[from] concision_init::error::InitError),
    #[cfg(feature = "anyhow")]
    #[error(transparent)]
    AnyError(#[from] anyhow::Error),
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
}

#[cfg(feature = "alloc")]
impl TensorError {
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
impl From<String> for TensorError {
    fn from(value: String) -> Self {
        Self::Unknown(value)
    }
}

#[cfg(feature = "alloc")]
impl From<&str> for TensorError {
    fn from(value: &str) -> Self {
        Self::Unknown(String::from(value))
    }
}
