/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use smart_default::SmartDefault;
use strum::{AsRefStr, EnumCount, EnumIs, EnumIter, VariantNames};

#[derive(
    AsRefStr,
    Clone,
    Debug,
    EnumCount,
    EnumIs,
    EnumIter,
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
pub enum ExternalError {
    Error(String),
    #[default]
    Unknown,
}

impl ExternalError {
    pub fn new(err: impl ToString) -> Self {
        let err = err.to_string();
        if err.is_empty() {
            return Self::unknown();
        }
        Self::error(err)
    }

    pub fn error(err: impl ToString) -> Self {
        ExternalError::Error(err.to_string())
    }

    pub fn unknown() -> Self {
        ExternalError::Unknown
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ExternalError {}

impl core::fmt::Display for ExternalError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let msg = match self {
            ExternalError::Error(ref err) => err.to_string(),
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

impl_from_error!(ExternalError::Error<&str, String>);
