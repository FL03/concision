/*
   Appellation: pad <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
mod impl_pad;
mod impl_pad_mode;
mod impl_padding;

pub type PadResult<T = ()> = Result<T, PadError>;

/// The [`Pad`] trait defines a padding operation for tensors.
pub trait Pad<T> {
    type Output;

    fn pad(&self, mode: PadMode<T>, pad: &[[usize; 2]]) -> Self::Output;
}

pub struct Padding<T> {
    pub(crate) action: PadAction,
    pub(crate) mode: PadMode<T>,
    pub(crate) pad: Vec<[usize; 2]>,
    pub(crate) padding: usize,
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    variants::VariantConstructors,
    strum::AsRefStr,
    strum::Display,
    strum::EnumCount,
    strum::EnumIs,
    strum::EnumIter,
    strum::EnumString,
    strum::VariantArray,
    strum::VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case", untagged)
)]
#[strum(serialize_all = "snake_case")]
pub enum PadAction {
    Clipping,
    Lane,
    Reflecting,
    #[default]
    StopAfterCopy,
    Wrapping,
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    strum::AsRefStr,
    strum::Display,
    strum::EnumCount,
    strum::EnumIs,
    strum::VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
#[strum(serialize_all = "snake_case")]
#[repr(C)]
#[derive(Default)]
pub enum PadMode<T = f64> {
    Constant(T),
    Edge,
    Maximum,
    Mean,
    Median,
    Minimum,
    Mode,
    Reflect,
    Symmetric,
    #[default]
    Wrap,
}

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
