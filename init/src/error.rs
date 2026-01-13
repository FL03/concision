/*
    appellation: error <module>
    authors: @FL03
*/
//! Initialization related errors and other useful primitives
#[cfg(feature = "alloc")]
use alloc::string::String;

/// a type alias for a [`Result`](core::result::Result) type that is used throughout
/// the library using an [`InitError`] as the error type.
pub type Result<T> = core::result::Result<T, InitError>;

/// The [`InitError`] type enumerates various initialization errors while integrating with the
/// external errors largely focused on randomization.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InitError {
    #[cfg(feature = "alloc")]
    #[error("Failed to initialize with the given distribution: {0}")]
    DistributionError(String),
    #[cfg(feature = "rng")]
    #[error(transparent)]
    RngError(#[from] getrandom::Error),
    #[cfg(feature = "rand")]
    #[error("[NormalError]: {0}")]
    NormalError(rand_distr::NormalError),
    #[error(transparent)]
    #[cfg(feature = "rand")]
    UniformError(#[from] rand_distr::uniform::Error),
    #[error("[WeibullError]: {0}")]
    #[cfg(feature = "rand")]
    WeibullError(rand_distr::WeibullError),
}

#[cfg(feature = "rand")]
mod rand_err {
    use super::InitError;
    use rand_distr::{NormalError, WeibullError};

    impl From<NormalError> for InitError {
        fn from(err: rand_distr::NormalError) -> Self {
            InitError::NormalError(err)
        }
    }

    impl From<WeibullError> for InitError {
        fn from(err: rand_distr::WeibullError) -> Self {
            InitError::WeibullError(err)
        }
    }
}
