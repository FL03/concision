/*
    Appellation: hyperparameters <module>
    Contrib: @FL03
*/

#[doc(hidden)]
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
    strum::EnumIter,
    strum::EnumString,
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
    Dropout,
    #[default]
    LearningRate,
    Momentum,
    Temperature,
    WeightDecay,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    #[test]
    fn test_hyper() {
        use strum::IntoEnumIterator;

        assert_eq!(
            Hyperparameters::from_str("learning_rate"),
            Ok(Hyperparameters::LearningRate)
        );

        for variant in Hyperparameters::iter() {
            let name = variant.as_ref();
            let parsed = Hyperparameters::from_str(name);
            assert_eq!(parsed, Ok(variant));
        }
    }
}
