/*
   Appellation: head <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct AttentionHead {
    keys: Vec<f32>,
    queries: Vec<f32>,
    values: Vec<f32>,
    pos: usize,
}

impl AttentionHead {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            queries: Vec::new(),
            values: Vec::new(),
            pos: 0,
        }
    }
}

impl std::fmt::Display for AttentionHead {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}
