/*
    Appellation: err <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::ErrorKind;
use uuid::Uuid;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize,))]
pub struct Error {
    id: Uuid,
    kind: ErrorKind,
    message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl ToString) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind,
            message: message.to_string(),
        }
    }

    pub fn from_kind<K>(kind: K) -> Self
    where
        K: Into<ErrorKind>,
    {
        Self::new(kind.into(), "")
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn kind(&self) -> &ErrorKind {
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

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Self {
        Self::from_kind(kind)
    }
}

impl<'a, K> From<&'a K> for Error
where
    K: Clone + Into<ErrorKind>,
{
    fn from(kind: &'a K) -> Self {
        Self::from_kind(kind.clone())
    }
}
