/*
    Appellation: error <module>
    Contrib: @FL03
*/

#[derive(Debug, thiserror::Error)]
pub enum MathematicalError {
    #[error("Dimension Error: {0}")]
    DimensionalError(String),
    #[error("Infinity Error")]
    InfinityError,
    #[error("NaN Error")]
    NaNError,
}
