/*
   Appellation: network <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neural::neurons::activate::{Activator, ReLU};
use ndarray::Array2;

/// All vectors have a dimension of (nodes, elem)
pub fn ffn(data: Array2<f64>, bias: Array2<f64>, weights: Array2<f64>) -> Array2<f64> {
    let a = data * weights.row(0) + bias.row(0);
    ReLU::rho(a) * weights.row(1) + bias.row(1)
}

pub struct FFN {}
