/*
   Appellation: mode <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use strum::{Display, EnumCount, EnumIs, EnumIter, EnumString, EnumVariantNames};

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
    EnumVariantNames,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
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