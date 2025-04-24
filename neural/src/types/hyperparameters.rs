/*
    Appellation: hyperparameters <module>
    Contrib: @FL03
*/
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct KeyValue<K = String, V = f64> {
    pub key: K,
    pub value: V,
}

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
    scsys_derive::VariantConstructors,
    strum::AsRefStr,
    strum::Display,
    strum::EnumCount,
    strum::EnumIs,
    strum::VariantArray,
    strum::VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case", untagged)
)]
#[strum(serialize_all = "snake_case")]
pub enum Hyperparameters {
    Decay,
    #[default]
    LearningRate,
    Momentum,
    BatchSize,
}

impl core::str::FromStr for Hyperparameters {
    type Err = strum::ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "decay" => Ok(Hyperparameters::Decay),
            "learning_rate" => Ok(Hyperparameters::LearningRate),
            "momentum" => Ok(Hyperparameters::Momentum),
            _ => Err(strum::ParseError::VariantNotFound),
        }
    }
}
