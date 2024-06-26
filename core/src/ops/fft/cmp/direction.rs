/*
    Appellation: direction <fft>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use scsys::VariantConstructors;
use strum::{
    AsRefStr, Display, EnumCount, EnumIs, EnumIter, EnumString, VariantArray, VariantNames,
};

///
#[derive(
    AsRefStr,
    Clone,
    Copy,
    Debug,
    Default,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    EnumString,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    VariantArray,
    VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "lowercase", untagged)
)]
#[repr(usize)]
#[strum(serialize_all = "lowercase")]
pub enum FftDirection {
    #[default]
    Forward = 0,
    Inverse = 1,
}

impl FftDirection {
    pub fn forward() -> Self {
        Self::Forward
    }

    pub fn inverse() -> Self {
        Self::Inverse
    }
}

impl From<usize> for FftDirection {
    fn from(direction: usize) -> Self {
        match direction % Self::COUNT {
            0 => Self::Forward,
            _ => Self::Inverse,
        }
    }
}
impl From<FftDirection> for usize {
    fn from(direction: FftDirection) -> Self {
        direction as usize
    }
}

///
#[derive(
    AsRefStr,
    Clone,
    Copy,
    Debug,
    Default,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    EnumString,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    VariantArray,
    VariantConstructors,
    VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "lowercase", untagged)
)]
#[repr(usize)]
#[strum(serialize_all = "lowercase")]
pub enum FftMode {
    #[default]
    Complex,
    Real,
}
