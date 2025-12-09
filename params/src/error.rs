/*
    Appellation: error <module>
    Contrib: @FL03
*/
//! This module defines error types and handling mechanisms for the `params` crate.
//!

/// A type alias for a [`Result`](core::result::Result) which uses the [`ParamsError`] type
pub type Result<T = ()> = core::result::Result<T, ParamsError>;

/// the [`ParamsError`] enumerates various errors that can occur within the parameters of a
/// neural network.
#[derive(Debug, thiserror::Error)]
pub enum ParamsError {
    #[error("Invalid biases")]
    InvalidBiases,
    #[error("Invalid weights")]
    InvalidWeights,
    #[error(
        "Unable to complete the operation due to a mismatch between shapes: expected {expected:?}, found {found:?}"
    )]
    MismatchedShapes {
        expected: &'static [usize],
        found: &'static [usize],
    },
    #[error("An invalid tensor of length {0} was provided")]
    InvalidLength(usize),
    #[error("Invalid output shape")]
    InvalidOutputShape,
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Invalid parameter type")]
    InvalidParameterType,
    #[error("Invalid parameter value")]
    InvalidParameterValue,
    #[error("Must be non-empty")]
    EmptyInput,
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
}
