/*
   Appellation: head <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize)]
pub struct Weights {
    key: Vec<f64>,
    query: Vec<f64>,
    value: Vec<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct AttentionHead {
    data: Vec<f64>,
    dim: usize,
}

impl AttentionHead {
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
        }
    }
}

impl std::fmt::Display for AttentionHead {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}
