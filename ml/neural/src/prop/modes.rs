/*
    Appellation: kinds <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIs, EnumIter, EnumString, EnumVariantNames};

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Display,
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
pub enum PropagationMode {
    #[default]
    Backward = 0,
    Forward = 1,
}

impl PropagationMode {
    pub fn backward() -> Self {
        Self::Backward
    }

    pub fn forward() -> Self {
        Self::Forward
    }
}

impl From<usize> for PropagationMode {
    fn from(value: usize) -> Self {
        match value % 2 {
            1 => Self::Forward,
            _ => Self::Backward,
        }
    }
}

impl From<PropagationMode> for usize {
    fn from(value: PropagationMode) -> Self {
        value as usize
    }
}
