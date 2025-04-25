/*
    Appellation: error <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub type PadResult<T = ()> = Result<T, PadError>;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, thiserror::Error)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
pub enum PadError {
    #[error("Inconsistent Dimensions")]
    InconsistentDimensions,
}
