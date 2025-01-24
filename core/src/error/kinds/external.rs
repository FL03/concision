/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::error::ErrorKind;
#[cfg(any(feature = "alloc", feature = "std"))]
use crate::rust::String;
use smart_default::SmartDefault;
use strum::{AsRefStr, EnumCount, EnumIs, VariantNames};

#[derive(
    AsRefStr,
    Clone,
    Debug,
    EnumCount,
    EnumIs,
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
pub enum ExternalError<E = String> {
    Error(E),
    #[default]
    Unknown,
}

impl<E> ExternalError<E> {
    pub fn new(err: Option<E>) -> Self {
        if let Some(err) = err {
            Self::error(err)
        } else {
            Self::unknown()
        }
    }

    pub fn error(err: impl Into<E>) -> Self {
        ExternalError::Error(err.into())
    }

    pub fn unknown() -> Self {
        ExternalError::Unknown
    }
}

impl<E> ErrorKind for ExternalError<E> where E: Clone + ToString {}

impl<E> core::fmt::Display for ExternalError<E>
where
    E: ToString,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let msg = match self {
            ExternalError::Error(err) => err.to_string(),
            ExternalError::Unknown => "Unknown error".to_string(),
        };
        write!(f, "{}", msg)
    }
}

#[cfg(feature = "std")]
impl From<Box<dyn std::error::Error>> for ExternalError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        ExternalError::Error(err.to_string())
    }
}

from_variant! {
    ExternalError::Error {
        <&str>.to_string()
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
from_variant! {
    ExternalError::Error {
        <String>.to_string(),
    }
}
