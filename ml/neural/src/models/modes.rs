/*
    Appellation: modes <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIs, EnumIter, EnumString, VariantNames};

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
pub enum Mode {
    Predict,
    #[default]
    Test,
    Train,
}

impl Mode {
    pub fn predict() -> Self {
        Self::Predict
    }

    pub fn test() -> Self {
        Self::Test
    }

    pub fn train() -> Self {
        Self::Train
    }
}

impl From<usize> for Mode {
    fn from(mode: usize) -> Self {
        match mode % Mode::VARIANTS.len() {
            0 => Self::Predict,
            1 => Self::Test,
            _ => Self::Train,
        }
    }
}

impl From<Mode> for usize {
    fn from(mode: Mode) -> Self {
        mode as usize
    }
}
