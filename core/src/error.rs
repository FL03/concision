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

/// a type alias for a [Result] with a [Error]
pub type Result<T = ()> = core::result::Result<T, Error>;

/// The [`Error`] type enumerates various errors that can occur within the framework.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid Shape")]
    InvalidShape,
    #[error(transparent)]
    PadError(#[from] crate::ops::pad::error::PadError),
    #[error(transparent)]
    ParamError(#[from] crate::params::error::ParamsError),
    #[error(transparent)]
    #[cfg(feature = "cnc_init")]
    InitError(#[from] concision_init::error::InitError),
    #[error(transparent)]
    #[cfg(feature = "cnc_utils")]
    UtilityError(#[from] concision_utils::error::UtilityError),
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
    ShapeError(#[from] ndarray::ShapeError),
    #[error(transparent)]
    #[cfg(feature = "rand")]
    UniformError(#[from] rand_distr::uniform::Error),
    #[cfg(feature = "alloc")]
    #[error("Unknown Error: {0}")]
    Unknown(String),
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
