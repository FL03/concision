/*
    Appellation: kinds <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use strum::{AsRefStr, EnumCount, EnumIs, EnumIter, EnumString, VariantNames};


#[derive(
    AsRefStr,
    Clone,
    Debug,
    Default,
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
#[non_exhaustive]
#[repr(usize)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "lowercase", tag = "kind")
)]
#[strum(serialize_all = "lowercase")]
pub enum ParamKind {
    #[default]
    Bias,
    Weight,
    Other(String),
}

impl ParamKind {
    pub fn bias() -> Self {
        Self::Bias
    }

    pub fn weight() -> Self {
        Self::Weight
    }

    pub fn other(name: impl ToString) -> Self {
        Self::Other(name.to_string())
    }
}

impl core::fmt::Display for ParamKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let content = match self {
            ParamKind::Bias => "bias",
            ParamKind::Weight => "weight",
            ParamKind::Other(name) => name,
        };
        write!(f, "{}", content)
    }
}
