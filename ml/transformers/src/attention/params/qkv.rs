/*
   Appellation: qkv <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use strum::{Display, EnumIs, EnumIter, EnumString, EnumVariantNames};

#[derive(
    Clone,
    Copy,
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
#[repr(usize)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum QKV {
    #[serde(alias = "k")]
    #[strum(serialize = "k", serialize = "key")]
    Key,
    #[default]
    #[serde(alias = "q")]
    #[strum(serialize = "q", serialize = "query")]
    Query,
    #[serde(alias = "v")]
    #[strum(serialize = "v", serialize = "val", serialize = "value")]
    Value,
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_qkv() {
        use QKV::Key;
        let w = Key;
        assert_eq!(w.to_string(), "key");
        assert_eq!(Key, QKV::from_str("key").unwrap());
        assert_eq!(Key, QKV::from_str("k").unwrap());
    }
}
