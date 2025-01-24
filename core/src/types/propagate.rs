/*
    Appellation: direction <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    scsys::VariantConstructors,
    strum::AsRefStr,
    strum::Display,
    strum::EnumCount,
    strum::EnumIs,
    strum::EnumIter,
    strum::EnumString,
    strum::VariantNames,
)]
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

impl From<Propagate> for usize {
    fn from(direction: Propagate) -> Self {
        direction as usize
    }
}

impl From<usize> for Propagate {
    fn from(index: usize) -> Self {
        use strum::EnumCount;
        match index % Self::COUNT {
            0 => Self::Backward,
            _ => Self::Forward,
        }
    }
}
