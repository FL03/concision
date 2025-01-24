/*
   Appellation: kinds <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::external::*;

mod external;

use crate::nn::ModelError;
use strum::{AsRefStr, Display, EnumCount, EnumIs, VariantNames};

err! {
    PredictError {
        ArithmeticError,
        ShapeMismatch,
        TypeError,
    }
}

err! {
    ShapeError {
        IncompatibleLayout,
        IncompatibleRank,
        ShapeMismatch,
        SizeMismatch,
    }
}

#[derive(
    AsRefStr,
    Clone,
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
    serde(rename_all = "lowercase", tag = "kind")
)]
#[strum(serialize_all = "lowercase")]
pub enum Errors {
    IO,
    External(ExternalError),
    Model(ModelError),
    Shape(String),
}

/*
 ************* Implementations *************
*/
from_err!(Errors:
    Errors::External(ExternalError),
    Errors::Model(ModelError),
);

impl From<&str> for Errors {
    fn from(err: &str) -> Self {
        Errors::External(err.into())
    }
}

impl From<PredictError> for Errors {
    fn from(err: PredictError) -> Self {
        Errors::Model(ModelError::Predict(err))
    }
}
