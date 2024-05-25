/*
    Appellation: mode <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use scsys::VariantConstructors;
use strum::{
    AsRefStr, Display, EnumCount, EnumIs, EnumIter, EnumString, VariantArray, VariantNames,
};

toggle!(enum C, R);

///
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
    VariantArray,
    VariantConstructors,
    VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "lowercase", untagged)
)]
#[repr(usize)]
#[strum(serialize_all = "lowercase")]
pub enum FftMode {
    #[default]
    Complex,
    Real,
}
