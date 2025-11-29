/*
    Appellation: hyper_params <module>
    Contrib: @FL03
*/
#[cfg(feature = "alloc")]
use alloc::string::{String, ToString};

/// An enumeration of common HyperParams used in neural network configurations.
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
    strum::EnumIter,
    strum::VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case", untagged)
)]
pub enum HyperParams {
    Decay,
    Dropout,
    LearningRate,
    Momentum,
    Temperature,
    WeightDecay,
    Beta1,
    Beta2,
    Epsilon,
    #[cfg(feature = "alloc")]
    Custom(String),
}

impl HyperParams {
    #[cfg(feature = "alloc")]
    /// creates a custom hyperparameter variant
    pub fn custom<T: ToString>(name: T) -> Self {
        HyperParams::Custom(name.to_string())
    }
    /// returns a list of variants as strings
    pub const fn variants() -> &'static [&'static str] {
        use strum::VariantNames;
        HyperParams::VARIANTS
    }
}

impl core::fmt::Display for HyperParams {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_ref())
    }
}

impl AsRef<str> for HyperParams {
    fn as_ref(&self) -> &str {
        match self {
            HyperParams::Decay => "decay",
            HyperParams::Dropout => "dropout",
            HyperParams::LearningRate => "learning_rate",
            HyperParams::Momentum => "momentum",
            HyperParams::Temperature => "temperature",
            HyperParams::WeightDecay => "weight_decay",
            HyperParams::Beta1 => "beta1",
            HyperParams::Beta2 => "beta2",
            HyperParams::Epsilon => "epsilon",
            HyperParams::Custom(s) => s.as_ref(),
        }
    }
}

impl core::borrow::Borrow<str> for HyperParams {
    fn borrow(&self) -> &str {
        self.as_ref()
    }
}

#[cfg(feature = "alloc")]
impl core::convert::From<String> for HyperParams {
    fn from(s: String) -> Self {
        core::str::FromStr::from_str(&s).expect("Failed to convert String to HyperParams")
    }
}

impl From<&str> for HyperParams {
    fn from(s: &str) -> Self {
        core::str::FromStr::from_str(s).expect("Failed to convert &str to HyperParams")
    }
}

impl core::str::FromStr for HyperParams {
    type Err = core::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "decay" => Ok(HyperParams::Decay),
            "dropout" => Ok(HyperParams::Dropout),
            "learning_rate" => Ok(HyperParams::LearningRate),
            "momentum" => Ok(HyperParams::Momentum),
            "temperature" => Ok(HyperParams::Temperature),
            "weight_decay" => Ok(HyperParams::WeightDecay),
            #[cfg(feature = "alloc")]
            other => Ok(HyperParams::Custom(other.to_string())),
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
            HyperParams::from_str("learning_rate"),
            Ok(HyperParams::LearningRate)
        );

        for variant in HyperParams::iter() {
            let name = variant.as_ref();
            let parsed = HyperParams::from_str(name);
            assert_eq!(parsed, Ok(variant));
        }
    }
}
