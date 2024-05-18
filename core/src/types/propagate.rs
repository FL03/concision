/*
    Appellation: direction <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use strum::{AsRefStr, Display, EnumCount, EnumIs, EnumIter, EnumString, VariantNames};

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
    VariantNames,
)]
#[repr(usize)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "lowercase")
)]
#[strum(serialize_all = "lowercase")]
pub enum Propagate {
    Backward = 0,
    #[default]
    Forward = 1,
}

impl Propagate {
    /// A functional alias for [Direction::Backward].
    pub fn backward() -> Self {
        Self::Backward
    }
    /// A functional alias for [Direction::Forward].
    pub fn forward() -> Self {
        Self::Forward
    }
}

impl From<Propagate> for usize {
    fn from(direction: Propagate) -> Self {
        direction as usize
    }
}

impl From<usize> for Propagate {
    fn from(index: usize) -> Self {
        match index % Self::COUNT {
            0 => Self::Backward,
            _ => Self::Forward,
        }
    }
}
