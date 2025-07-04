/*
    Appellation: error <module>
    Contrib: @FL03
*/

/// the [`ParamsError`] enumerates various errors that can occur within the parameters of a
/// neural network.
#[derive(Debug, thiserror::Error)]
pub enum ParamsError {
    #[error("Dimension Error: {0}")]
    DimensionalError(String),
    #[error("Invalid biases")]
    InvalidBiases,
    #[error("Invalid weights")]
    InvalidWeights,
    #[error("Invalid input shape")]
    InvalidInputShape,
    #[error("Invalid output shape")]
    InvalidOutputShape,
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Invalid parameter type")]
    InvalidParameterType,
    #[error("Invalid parameter value")]
    InvalidParameterValue,
}
