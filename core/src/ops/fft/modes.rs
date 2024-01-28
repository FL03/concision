/*
   Appellation: modes <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use strum::{Display, EnumCount, EnumIs, EnumIter, EnumString, VariantArray, VariantNames};

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
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
    Serialize,
    VariantArray,
    VariantNames,
)]
#[repr(usize)]
#[serde(rename_all = "lowercase")]
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

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
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
    Serialize,
    VariantArray,
    VariantNames,
)]
#[repr(usize)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum FftMode {
    #[default]
    Complex,
    Real,
}
