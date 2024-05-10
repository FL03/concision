/*
   Appellation: kinds <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{external::*, predict::*};

mod external;
mod predict;

use crate::nn::ModelError;
use strum::{AsRefStr, Display, EnumCount, EnumIs, VariantNames};

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
pub enum ErrorKind {
    IO,
    External(ExternalError),
    Predict(PredictError),
    Model(ModelError),
    Shape(String),
}

macro_rules! from_err {
    ($S:ty: $($($p:ident)::*($err:ty)),* $(,)*) => {
        $(
            from_err!(@impl $S: $($p)::*($err));
        )*
    };
    (@impl $S:ty: $($p:ident)::*($err:ty)) => {
        impl From<$err> for $S {
            fn from(err: $err) -> Self {
                $($p)::*(err)
            }
        }
    };
}

from_err!(ErrorKind:
    ErrorKind::External(ExternalError),
    ErrorKind::Model(ModelError),
    ErrorKind::Predict(PredictError),
);
