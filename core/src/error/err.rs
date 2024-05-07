/*
    Appellation: err <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::ErrorKind;
use uuid::Uuid;

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

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}
