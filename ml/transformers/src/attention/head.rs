/*
   Appellation: head <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};



pub struct AttentionDim {
    pub attention: usize, // The dimension of the key, query, and value vectors
    pub batch: usize,     // The batch size
    pub heads: usize,     // The number of attention heads
    pub model: usize,     // The dimension of the model (embedding size)
}

impl AttentionDim {
    pub fn new(attention: usize, batch: usize, heads: usize, model: usize) -> Self {
        Self {
            attention,
            batch,
            heads,
            model,
        }
    }

    pub fn linear(batch: usize, model: usize, heads: usize) -> Self {
        Self {
            attention: model / heads,
            batch,
            heads,
            model,
        }
    }
}

pub struct AttentionParams {
    pub dim: AttentionDim,
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
