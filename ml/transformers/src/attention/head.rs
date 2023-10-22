/*
   Appellation: head <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::prelude::activate::{Activator, Softmax};
use ndarray::{s, Array3, Array2};
use serde::{Deserialize, Serialize};

fn _attention(qkv: &Array3<f64>) -> Array2<f64> {
    let query = qkv.slice(s![0, .., ..]).to_owned();
    let key = qkv.slice(s![1, .., ..]).to_owned();
    let value = qkv.slice(s![2, .., ..]).to_owned();
    let dk = qkv.shape()[1] as f64;

    let inner = (query * key.t()) / dk.sqrt();
    Softmax::rho(inner) * value
}


pub struct AttentionDim {
    pub attention: usize, // The dimension of the key, query, and value vectors
    pub heads: usize,     // The number of attention heads
    pub model: usize,     // The dimension of the model (embedding size)
}

impl AttentionDim {
    pub fn new(attention: usize, heads: usize, model: usize) -> Self {
        Self {
            attention,
            heads,
            model,
        }
    }

    pub fn linear(model: usize, heads: usize) -> Self {
        Self {
            attention: model / heads,
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
