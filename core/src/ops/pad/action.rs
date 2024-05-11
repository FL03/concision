/*
    Appellation: action <module>
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
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case", untagged)
)]
#[repr(u8)]
#[strum(serialize_all = "snake_case")]
pub enum PadAction {
    Clipping,
    Lane,
    Reflecting,
    #[default]
    StopAfterCopy,
    Wrapping,
}
