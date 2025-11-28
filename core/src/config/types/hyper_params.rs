/*
    Appellation: hyperparameters <module>
    Contrib: @FL03
*/
use super::KeyValue;

#[cfg(feature = "alloc")]
use alloc::string::String;

/// An enumeration of common hyperparameters used in neural network configurations.
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
    Custom { key: String, value: T },
}

impl<K, V> From<KeyValue<K, V>> for HyperParams<V>
where
    K: AsRef<str>,
{
    fn from(KeyValue { key, value }: KeyValue<K, V>) -> Self {
        match key.as_ref() {
            "decay" => HyperParams::Decay(value),
            "dropout" => HyperParams::Dropout(value),
            "learning_rate" => HyperParams::LearningRate(value),
            "momentum" => HyperParams::Momentum(value),
            "temperature" => HyperParams::Temperature(value),
            "weight_decay" => HyperParams::WeightDecay(value),
            k => HyperParams::Custom {
                key: String::from(k),
                value,
            },
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
