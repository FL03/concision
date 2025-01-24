/*
   Appellation: kinds <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, thiserror::Error)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum PredictError {
    #[error("Arithmetic Error: {0}")]
    ArithmeticError(String),
    #[error("Propagation Error: {0}")]
    PropagationError(String),

    #[error("Type Error: {0}")]
    TypeError(String),
}
