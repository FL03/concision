/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Errors;

#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Error {
    pub kind: Errors,
    pub message: String,
    pub ts: u128,
}

impl Error {
    pub fn new(kind: Errors, message: String) -> Self {
        let ts = crate::prelude::now();
        Self { kind, message, ts }
    }

    pub fn kind(&self) -> &Errors {
        &self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn ts(&self) -> u128 {
        self.ts
    }

    pub fn set_kind(&mut self, kind: Errors) {
        self.kind = kind;
        self.on_update();
    }

    pub fn set_message(&mut self, message: String) {
        self.message = message;
        self.on_update();
    }

    pub fn with_kind(mut self, kind: Errors) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_message(mut self, message: String) -> Self {
        self.message = message;
        self
    }

    fn on_update(&mut self) {
        self.ts = crate::prelude::now();
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Error: {}", self.message())
    }
}

impl std::error::Error for Error {}

impl From<Box<dyn std::error::Error>> for Error {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        Self::new(Errors::Unknown, err.to_string())
    }
}

impl From<Errors> for Error {
    fn from(err: Errors) -> Self {
        Self::new(err, String::new())
    }
}

impl From<&str> for Error {
    fn from(err: &str) -> Self {
        Self::new(Errors::Unknown, err.to_string())
    }
}

impl From<String> for Error {
    fn from(err: String) -> Self {
        Self::new(Errors::Unknown, err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::new(Errors::IO, err.to_string())
    }
}

impl From<std::num::ParseFloatError> for Error {
    fn from(err: std::num::ParseFloatError) -> Self {
        Self::new(Errors::Parse, err.to_string())
    }
}

impl From<std::num::ParseIntError> for Error {
    fn from(err: std::num::ParseIntError) -> Self {
        Self::new(Errors::Parse, err.to_string())
    }
}

impl From<anyhow::Error> for Error {
    fn from(err: anyhow::Error) -> Self {
        Self::new(Errors::Unknown, err.to_string())
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(err: ndarray::ShapeError) -> Self {
        Self::new(Errors::Dimension, err.to_string())
    }
}

#[cfg(feature = "serde")]
impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::new(Errors::Syntax, err.to_string())
    }
}

mod std_impl {}
