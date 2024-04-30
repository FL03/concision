/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::kinds::*;

pub mod kinds;

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
    serde(rename_all = "lowercase", untagged)
)]
#[strum(serialize_all = "lowercase")]
pub enum Error {
    IO(String),
    #[default]
    External(ExternalError),
    Shape(String),
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

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


impl_from_error!(Error::IO<std::io::Error>);
