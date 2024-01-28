/*
    Appellation: direction <mod>
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
pub enum Direction {
    Backward = 0,
    #[default]
    Forward = 1,
}

impl Direction {
    /// A functional alias for [Direction::Backward].
    pub fn backward() -> Self {
        Self::Backward
    }
    /// A functional alias for [Direction::Forward].
    pub fn forward() -> Self {
        Self::Forward
    }
}

impl From<Direction> for usize {
    fn from(direction: Direction) -> Self {
        direction as usize
    }
}

impl From<usize> for Direction {
    fn from(index: usize) -> Self {
        match index % Self::COUNT {
            0 => Self::Backward,
            _ => Self::Forward,
        }
    }
}
