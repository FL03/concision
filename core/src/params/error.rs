/*
    Appellation: error <module>
    Contrib: @FL03
*/

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
}
