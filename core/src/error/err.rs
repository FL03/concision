/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::kinds::*;
use crate::models::ModelError;
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
pub enum Error {
    IO(String),
    External(ExternalError),
    Predict(PredictError),
    Model(ModelError),
    Shape(String),
}

// impl core::fmt::Display for Error {
//     fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
//         let msg = match self {
//             Error::IO(ref err) => err.to_string(),
//             Error::Error(ref err) => err.to_string(),
//             Error::Shape(ref err) => err.to_string(),
//         };
//         write!(f, "{}", msg)
//     }
// }

#[cfg(feature = "std")]
impl_from_error!(Error::IO<std::io::Error>);

macro_rules! from_err {
    ($($variant:ident<$err:ty>),* $(,)*) => {
        $(
            from_err!(@impl $variant<$err>);
        )*
    };
    (@impl $variant:ident<$err:ty>) => {
        impl From<$err> for Error {
            fn from(err: $err) -> Self {
                Error::$variant(err)
            }
        }
    };
}

from_err!(
    External<ExternalError>,
    Model<ModelError>,
    Predict<PredictError>,
);
