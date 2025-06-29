/*
    appellation: error <module>
    authors: @FL03
*/
/// this module defines the error type, [`InitError`], used for various initialization errors
/// that one may encounter within the library.
#[cfg(feature = "alloc")]
use alloc::string::String;
use rand_distr::NormalError;
use rand_distr::uniform::Error as UniformError;
#[allow(dead_code)]
/// a private type alias for a [`Result`](core::result::Result) type that is used throughout
/// the library using an [`InitError`](InitError) as the error type.
pub(crate) type Result<T> = core::result::Result<T, InitError>;

#[derive(Debug, thiserror::Error)]
pub enum InitError {
    #[cfg(feature = "alloc")]
    #[error("Failed to initialize with the given distribution: {0}")]
    DistributionError(String),
    #[cfg(feature = "rand")]
    #[error("[NormalError] {0}")]
    NormalError(NormalError),
    #[error(transparent)]
    #[cfg(feature = "rand")]
    UniformError(#[from] UniformError),
}

#[cfg(feature = "rand")]
impl From<NormalError> for InitError {
    fn from(err: NormalError) -> Self {
        InitError::NormalError(err)
    }
}
