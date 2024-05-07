/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use smart_default::SmartDefault;
use strum::{AsRefStr, Display, EnumCount, EnumIs, EnumIter, EnumString, VariantNames};

#[derive(
    AsRefStr,
    Clone,
    Debug,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    EnumString,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    SmartDefault,
    VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case", untagged)
)]
#[strum(serialize_all = "snake_case")]
pub enum PredictError {
    #[default]
    ArithmeticError,
    ShapeMismatch,
    TypeError,
}

impl PredictError {
    variant_constructor!(
        ArithmeticError.arithmetic_error,
        ShapeMismatch.shape_mismatch,
        TypeError.type_error
    );
}
