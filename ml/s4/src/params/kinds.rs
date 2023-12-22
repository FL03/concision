/*
    Appellation: kinds <mod>
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
pub enum SSMParams {
    #[default]
    A = 0,
    B = 1,
    C = 2,
    D = 3,
}

impl SSMParams {
    pub fn a() -> Self {
        Self::A
    }

    pub fn b() -> Self {
        Self::B
    }

    pub fn c() -> Self {
        Self::C
    }

    pub fn d() -> Self {
        Self::D
    }
}

impl From<usize> for SSMParams {
    fn from(i: usize) -> Self {
        match i % SSMParams::COUNT {
            0 => Self::A,
            1 => Self::B,
            2 => Self::C,
            _ => Self::D,
        }
    }
}
