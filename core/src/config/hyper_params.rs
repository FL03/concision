/*
    Appellation: hyper_params <module>
    Contrib: @FL03
*/
/// An enumeration of common HyperParams used in neural network configurations.
#[derive(
    Clone,
    Debug,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    strum::AsRefStr,
    strum::Display,
    strum::EnumCount,
    strum::EnumIs,
    strum::EnumIter,
    strum::EnumString,
    strum::VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case", untagged)
)]
#[strum(serialize_all = "snake_case")]
#[non_exhaustive]
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
}

impl HyperParam {
    /// returns a list of variants as strings
    pub const fn variants() -> &'static [&'static str] {
        use strum::VariantNames;
        HyperParam::VARIANTS
    }
}

impl core::borrow::Borrow<str> for HyperParam {
    fn borrow(&self) -> &str {
        self.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::HyperParam;

    #[test]
    fn test_hyper_params() -> anyhow::Result<()> {
        use HyperParam::*;

        assert_eq!(HyperParam::try_from("learning_rate")?, LearningRate);

        assert_eq!(HyperParam::try_from("weight_decay")?, WeightDecay);

        for v in ["something", "another_param", "custom_hyperparam"] {
            assert!(HyperParam::try_from(v).is_err());
        }

        Ok(())
    }
}
