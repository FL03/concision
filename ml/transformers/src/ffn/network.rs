/*
   Appellation: network <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::FFNParams;
use crate::neural::func::activate::{Activate, ReLU};
use crate::neural::prelude::{Forward, Layer};
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
        let network = network.unwrap_or(crate::NETWORK_SIZE);
        let features = network / model;
        Self {
            input: Layer::input((model, features).into()),
            output: Layer::output((features, model).into(), 1),
            params: FFNParams::new(model, network),
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
