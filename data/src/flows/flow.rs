/*
   Appellation: flow <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct Flow {
    data: Vec<String>,
}

impl Flow {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl std::fmt::Display for Flow {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}
