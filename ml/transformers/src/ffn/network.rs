/*
   Appellation: network <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::FFNParams;
use crate::neural::func::activate::{Activate, ReLU};
use crate::neural::prelude::{Forward, Layer, LayerShape};
use ndarray::prelude::Array2;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FFN {
    input: Layer,
    output: Layer,
    pub params: FFNParams,
}

impl FFN {
    pub fn new(model: usize, network: Option<usize>) -> Self {
        let params = FFNParams::new(model, network.unwrap_or(crate::NETWORK_SIZE));
        let features = LayerShape::new(model, params.network_size());

        Self {
            input: Layer::input(features),
            output: Layer::output(features, 1),
            params,
        }
    }
}

impl Forward<Array2<f64>> for FFN {
    type Output = Array2<f64>;

    fn forward(&self, data: &Array2<f64>) -> Self::Output {
        self.output
            .forward(&ReLU::default().activate(&self.input.forward(data)))
    }
}
