/*
    Appellation: error <module>
    Contrib: @FL03
*/

/// a type alias for a [Result] with a [Error]
pub type Result<T = ()> = core::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid Shape: {0}")]
    InvalidShape(&'static str),
    #[error("Unknown Error: {0}")]
    Unknown(&'static str),
    #[error(transparent)]
    MathError(#[from] concision_utils::UtilityError),
    #[error(transparent)]
    PadError(#[from] crate::ops::pad::error::PadError),
    #[error(transparent)]
    ParamError(#[from] crate::params::error::ParamsError),
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
    #[cfg(feature = "alloc")]
    #[error(transparent)]
    BoxError(#[from] alloc::boxed::Box<dyn core::error::Error + Send + Sync + 'static>),
    #[cfg(feature = "anyhow")]
    #[error(transparent)]
    Other(#[from] anyhow::Error),
    #[cfg(feature = "serde")]
    #[error(transparent)]
    DeserializeError(#[from] serde::de::value::Error),
    #[cfg(feature = "std")]
    #[error(transparent)]
    IoError(#[from] std::io::Error),
    #[cfg(feature = "rand")]
    #[error(transparent)]
    UniformError(#[from] rand_distr::uniform::Error),
}

#[cfg(feature = "alloc")]
impl From<alloc::string::String> for Error {
    fn from(value: alloc::string::String) -> Self {
        Self::Unknown(alloc::boxed::Box::leak(value.into_boxed_str()))
    }
}

impl From<&'static str> for Error {
    fn from(value: &'static str) -> Self {
        Self::Unknown(value)
    }
}
