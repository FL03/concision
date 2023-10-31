/*
   Appellation: state <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct State {
    pub kind: String,
    pub message: String,
    pub ts: u128,
}

impl State {
    pub fn new(kind: impl ToString, message: String) -> Self {
        let ts = crate::now();
        Self {
            kind: kind.to_string(),
            message,
            ts,
        }
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn ts(&self) -> u128 {
        self.ts
    }

    pub fn set_kind(&mut self, kind: impl ToString) {
        self.kind = kind.to_string();
        self.on_update();
    }

    pub fn set_message(&mut self, message: String) {
        self.message = message;
        self.on_update();
    }

    pub fn with_kind(mut self, kind: impl ToString) -> Self {
        self.kind = kind.to_string();
        self
    }

    pub fn with_message(mut self, message: String) -> Self {
        self.message = message;
        self
    }

    fn on_update(&mut self) {
        self.ts = crate::now();
    }
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}
