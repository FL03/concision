/*
   Appellation: head <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::params::AttentionDim;
use super::Context;
use crate::neural::neurons::activate::{Activator, Softmax};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct AttentionHead {
    context: Context,
    dim: AttentionDim,
}

impl AttentionHead {
    pub fn new(dim: AttentionDim) -> Self {
        Self {
            context: Context::new(dim.head_dim()),
            dim,
        }
    }

    pub fn attention(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.context *= data;

        Softmax::rho(self.query().dot(&self.key().t()) * self.scale()) * self.value().clone()
    }

    pub fn scale(&self) -> f64 {
        1.0 / (self.dim.query_size() as f64).sqrt()
    }

    pub fn query(&self) -> &Array2<f64> {
        &self.context.query
    }

    pub fn key(&self) -> &Array2<f64> {
        &self.context.key
    }

    pub fn value(&self) -> &Array2<f64> {
        &self.context.value
    }
}

impl std::fmt::Display for AttentionHead {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}
