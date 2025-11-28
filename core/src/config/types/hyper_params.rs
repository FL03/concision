/*
    Appellation: hyperparameters <module>
    Contrib: @FL03
*/
use super::KeyValue;

#[derive(
    Clone,
    Debug,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    strum::EnumCount,
    strum::EnumIs,
    strum::EnumDiscriminants,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case", untagged),
    strum_discriminants(derive(serde::Deserialize, serde::Serialize))
)]
#[strum_discriminants(
    name(Hyperparameters),
    derive(
        Hash,
        Ord,
        PartialOrd,
        strum::AsRefStr,
        strum::Display,
        strum::EnumCount,
        strum::EnumIs,
        strum::EnumIter,
        strum::EnumString,
        strum::VariantArray,
        strum::VariantNames,
        variants::VariantConstructors,
    ),
    strum(serialize_all = "snake_case")
)]
#[strum(serialize_all = "snake_case")]
pub enum HyperParams<T = f64> {
    Decay(T),
    Dropout(T),
    LearningRate(T),
    Momentum(T),
    Temperature(T),
    WeightDecay(T),
    Unknown(KeyValue<String, T>),
}

impl<T> From<KeyValue<String, T>> for HyperParams<T> {
    fn from(kv: KeyValue<String, T>) -> Self {
        match kv.key.as_str() {
            "decay" => HyperParams::Decay(kv.value),
            "dropout" => HyperParams::Dropout(kv.value),
            "learning_rate" => HyperParams::LearningRate(kv.value),
            "momentum" => HyperParams::Momentum(kv.value),
            "temperature" => HyperParams::Temperature(kv.value),
            "weight_decay" => HyperParams::WeightDecay(kv.value),
            _ => HyperParams::Unknown(kv),
        }
    }
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
