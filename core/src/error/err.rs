/*
    Appellation: err <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Errors;
use crate::uuid;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize,))]
pub struct Error {
    id: String,
    kind: Errors,
    message: String,
}

impl Error {
    pub fn new(kind: Errors, message: impl ToString) -> Self {
        Self {
            id: uuid().to_string(),
            kind,
            message: message.to_string(),
        }
    }

    pub fn from_kind<K>(kind: K) -> Self
    where
        K: Into<Errors>,
    {
        Self::new(kind.into(), "")
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn kind(&self) -> &Errors {
        &self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn with_message(mut self, message: impl ToString) -> Self {
        self.message = message.to_string();
        self
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}
#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl From<Errors> for Error {
    fn from(kind: Errors) -> Self {
        Self::from_kind(kind)
    }
}

impl<'a, K> From<&'a K> for Error
where
    K: Clone + Into<Errors>,
{
    fn from(kind: &'a K) -> Self {
        Self::from_kind(kind.clone())
    }
}
