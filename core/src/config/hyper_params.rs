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
pub enum HyperParam {
    Decay,
    #[serde(alias = "drop_out", alias = "p")]
    Dropout,
    #[serde(alias = "lr", alias = "gamma")]
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

impl HyperParam {
    #[cfg(feature = "alloc")]
    /// creates a custom hyperparameter variant
    pub fn custom<T: ToString>(name: T) -> Self {
        HyperParam::Custom(name.to_string())
    }
    /// returns a list of variants as strings
    pub const fn variants() -> &'static [&'static str] {
        use strum::VariantNames;
        HyperParam::VARIANTS
    }
}

impl core::fmt::Display for HyperParam {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_ref())
    }
}

impl AsRef<str> for HyperParam {
    fn as_ref(&self) -> &str {
        match self {
            HyperParam::Decay => "decay",
            HyperParam::Dropout => "dropout",
            HyperParam::LearningRate => "learning_rate",
            HyperParam::Momentum => "momentum",
            HyperParam::Temperature => "temperature",
            HyperParam::WeightDecay => "weight_decay",
            HyperParam::Beta1 => "beta1",
            HyperParam::Beta2 => "beta2",
            HyperParam::Epsilon => "epsilon",
            HyperParam::Custom(s) => s.as_ref(),
        }
    }
}

impl core::borrow::Borrow<str> for HyperParam {
    fn borrow(&self) -> &str {
        self.as_ref()
    }
}

#[cfg(feature = "alloc")]
impl core::convert::From<String> for HyperParam {
    fn from(s: String) -> Self {
        core::str::FromStr::from_str(&s).expect("Failed to convert String to HyperParams")
    }
}

impl From<&str> for HyperParam {
    fn from(s: &str) -> Self {
        core::str::FromStr::from_str(s).expect("Failed to convert &str to HyperParams")
    }
}

impl core::str::FromStr for HyperParam {
    type Err = core::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "decay" => Ok(HyperParam::Decay),
            "dropout" => Ok(HyperParam::Dropout),
            "learning_rate" => Ok(HyperParam::LearningRate),
            "momentum" => Ok(HyperParam::Momentum),
            "temperature" => Ok(HyperParam::Temperature),
            "weight_decay" => Ok(HyperParam::WeightDecay),
            #[cfg(feature = "alloc")]
            other => Ok(HyperParam::Custom(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HyperParam;

    #[test]
    fn test_hyper_params() {
        use HyperParam::*;

        assert_eq!(HyperParam::from("learning_rate"), LearningRate);

        assert_eq!(HyperParam::from("weight_decay"), WeightDecay);

        for v in ["something", "another_param", "custom_hyperparam"] {
            let param = HyperParam::from(v);
            assert!(param.is_custom());
            match param {
                HyperParam::Custom(s) => assert_eq!(s, v),
                _ => panic!("Expected Custom variant"),
            }
        }
    }
}
