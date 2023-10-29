/*
   Appellation: network <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::FFNParams;
use crate::data::linear::LinearLayer;
use crate::neural::neurons::activate::{Activator, ReLU};
use ndarray::prelude::Array2;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct FFN {
    input: LinearLayer,
    output: LinearLayer,
    pub params: FFNParams,
}

impl FFN {
    pub fn new(model: usize, network: Option<usize>) -> Self {
        let params = FFNParams::new(model, network.unwrap_or(crate::NETWORK_SIZE));
        let layer = LinearLayer::new(params.model, params.network);
        Self {
            input: layer.clone(),
            output: layer,
            params,
        }
    }

    pub fn forward(&self, data: &Array2<f64>) -> Array2<f64> {
        self.output.linear(&ReLU::rho(self.input.linear(data)))
    }
}
