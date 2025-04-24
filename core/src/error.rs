/*
    Appellation: error <module>
    Contrib: @FL03
*/

/// a type alias for a [Result] with a [Error]
pub type Result<T = ()> = core::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    MathError(#[from] concision_math::MathematicalError),
    #[error(transparent)]
    PadError(#[from] crate::ops::pad::error::PadError),
    #[error(transparent)]
    ParamError(#[from] crate::params::error::ParamsError),
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
    #[cfg(feature = "anyhow")]
    #[error(transparent)]
    Other(#[from] anyhow::Error),
    #[error("Unknown Error: {0}")]
    Unknown(String),
}
