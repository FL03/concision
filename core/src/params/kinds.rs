/*
    Appellation: kinds <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIs, EnumIter, EnumString, EnumVariantNames};

#[derive(
    Clone,
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
#[non_exhaustive]
#[repr(usize)]
#[serde(rename_all = "lowercase", tag = "kind")]
#[strum(serialize_all = "lowercase")]
pub enum ParamKind {
    #[default]
    Bias,
    Weight,
    Other(String),
}
