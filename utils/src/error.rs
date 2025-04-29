/*
    Appellation: error <module>
    Contrib: @FL03
*/

#[allow(dead_code)]
pub(crate) type UtilityResult<T = ()> = Result<T, UtilityError>;

#[derive(Debug, thiserror::Error)]
pub enum UtilityError {
    #[error("Dimension Error: {0}")]
    DimensionalError(&'static str),
    #[error("Infinity Error")]
    InfinityError,
    #[error("NaN Error")]
    NaNError,
}
