/*
   Appellation: network <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::FFNParams;
use crate::neural::func::activate::{Activate, ReLU};
use crate::neural::layers::linear::LinearLayer;
use crate::neural::prelude::Forward;
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
}

impl Forward<Array2<f64>> for FFN {
    type Output = Array2<f64>;

    fn forward(&self, data: &Array2<f64>) -> Self::Output {
        self.output
            .forward(&Activate::activate(&ReLU, self.input.forward(data)))
    }
}
