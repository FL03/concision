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
pub enum LayerType {
    #[default]
    Input = 0,
    Hidden(usize),
    Output,
}

pub struct Position {
    pub index: usize,
    pub kind: LayerType,
}

impl Position {
    pub fn new(index: usize, kind: LayerType) -> Self {
        Self { index, kind }
    }
    pub fn new_input() -> Self {
        Self {
            index: 0,
            kind: LayerType::Input,
        }
    }
}
