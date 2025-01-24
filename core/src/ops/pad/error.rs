/*
    Appellation: error <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use strum::{
    AsRefStr, Display, EnumCount, EnumIs, EnumIter, EnumMessage, EnumString, VariantArray,
    VariantNames,
};

pub type PadResult<T = ()> = Result<T, PadError>;

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
    EnumMessage,
    EnumString,
    Eq,
    Ord,
    PartialEq,
    PartialOrd,
    VariantArray,
    VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(feature = "std", derive(std::hash::Hash))]
#[strum(serialize_all = "snake_case")]
#[repr(u8)]
pub enum PadError {
    #[default]
    InconsistentDimensions,
}

impl_err!(PadError);
