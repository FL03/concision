/*
   Appellation: mode <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use strum::{Display, EnumCount, EnumIs, EnumIter, EnumString, VariantNames};

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
    VariantNames,
)]
#[repr(usize)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum TensorKind {
    #[default]
    Standard = 0,
    Variable = 1,
}

impl TensorKind {
    /// A functional alias for [TensorKind::Standard].
    pub fn standard() -> Self {
        Self::Standard
    }
    /// A functional alias for [TensorKind::Variable].
    pub fn variable() -> Self {
        Self::Variable
    }
}

impl From<bool> for TensorKind {
    fn from(var: bool) -> Self {
        if var {
            Self::Variable
        } else {
            Self::Standard
        }
    }
}

impl From<TensorKind> for bool {
    fn from(var: TensorKind) -> Self {
        match var {
            TensorKind::Standard => false,
            TensorKind::Variable => true,
        }
    }
}

impl From<TensorKind> for usize {
    fn from(var: TensorKind) -> Self {
        var as usize
    }
}

impl From<usize> for TensorKind {
    fn from(index: usize) -> Self {
        match index % Self::COUNT {
            0 => Self::Standard,
            _ => Self::Variable,
        }
    }
}
