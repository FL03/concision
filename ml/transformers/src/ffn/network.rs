/*
   Appellation: network <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::neurons::activate::{Activator, ReLU};
use ndarray::{Array1, Array2};

/// All vectors have a dimension of (nodes, elem)
pub fn ffn(data: Array2<f64>, bias: Array2<f64>, weights: Array2<f64>) -> Array2<f64> {
    let a = data.dot(&weights) + bias.clone();
    ReLU::rho(a).dot(&weights) + bias
}

pub struct FFN {
    bias: Array2<f64>,
    weights: Array2<f64>,
}

impl FFN {
    pub fn new(bias: Array2<f64>, weights: Array2<f64>) -> Self {
        Self { bias, weights }
    }

    pub fn forward(&self, data: Array2<f64>) -> Array2<f64> {
        ffn(data, self.bias.clone(), self.weights.clone())
    }
}
