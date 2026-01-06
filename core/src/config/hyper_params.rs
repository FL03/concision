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
    #[cfg_attr(feature = "serde", serde(alias = "drop_out", alias = "p"))]
    Dropout,
    #[cfg_attr(feature = "serde", serde(alias = "lr", alias = "gamma"))]
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
    use crate::Parameter;

    use super::HyperParam;

    #[test]
    fn test_hyper_params() {
        use HyperParam::*;

        let tests = [
            ("decay", Decay),
            ("dropout", Dropout),
            ("momentum", Momentum),
            ("temperature", Temperature),
            ("beta1", Beta1),
            ("beta2", Beta2),
            ("epsilon", Epsilon),
        ];
        for (s, param) in tests {
            assert!(HyperParam::try_from(s).ok(), Some(param));
        }
    }
}
