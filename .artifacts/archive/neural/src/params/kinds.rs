/*
    Appellation: kinds <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use strum::{AsRefStr, EnumCount, EnumIs, EnumIter, EnumString, VariantNames};

pub trait ParamType: ToString {
    fn kind(&self) -> String;
}

impl<T> ParamType for T
where
    T: ToString,
{
    fn kind(&self) -> String {
        self.to_string()
    }
}

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

impl std::fmt::Display for ParamKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content = match self {
            ParamKind::Bias => "bias",
            ParamKind::Weight => "weight",
            ParamKind::Other(name) => name,
        };
        write!(f, "{}", content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_param_kind_map() {
        let name = "test";
        let other = ParamKind::other(name);

        let data = [
            (ParamKind::Bias, 0),
            (ParamKind::Weight, 1),
            (other.clone(), 2),
            (ParamKind::other("mask"), 3),
        ];
        let store = HashMap::<ParamKind, usize>::from_iter(data);
        assert_eq!(store.get(&ParamKind::Bias), Some(&0));
        assert_eq!(store.get(&other), Some(&2));
    }
}
