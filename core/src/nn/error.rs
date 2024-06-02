/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::error::PredictError;
use strum::{AsRefStr, Display, EnumCount, EnumIs, VariantNames};

#[derive(
    AsRefStr,
    Clone,
    Copy,
    Debug,
    Display,
    EnumCount,
    EnumIs,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case", untagged)
)]
#[strum(serialize_all = "snake_case")]
pub enum ModelError {
    Predict(PredictError),
}

impl ModelError {
    pub fn from_predict(err: PredictError) -> Self {
        ModelError::Predict(err)
    }

    pub fn predict(&self) -> Option<PredictError> {
        match *self {
            ModelError::Predict(err) => Some(err),
        }
    }
}

impl From<PredictError> for ModelError {
    fn from(err: PredictError) -> Self {
        ModelError::from_predict(err)
    }
}
