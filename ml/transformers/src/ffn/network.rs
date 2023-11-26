/*
   Appellation: network <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::FFNParams;
use crate::neural::prelude::{Forward, Layer, ReLU};
use crate::prelude::{MODEL, NETWORK};
use ndarray::prelude::{Array2, NdFloat};
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FFN<T = f64>
where
    T: Float,
{
    input: Layer<T>,
    output: Layer<T>,
    pub params: FFNParams,
}

impl<T> FFN<T>
where
    T: Float,
{
    pub fn new(model: usize, network: usize) -> Self {
        Self {
            input: Layer::input((model, network).into()),
            output: Layer::output((network, model).into(), 1),
            params: FFNParams::new(model, network),
        }
    }

    pub fn input(&self) -> &Layer<T> {
        &self.input
    }

    pub fn output(&self) -> &Layer<T> {
        &self.output
    }
}

impl<T> Default for FFN<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(MODEL, NETWORK)
    }
}

impl<T> Forward<Array2<T>> for FFN<T>
where
    T: NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, data: &Array2<T>) -> Self::Output {
        self.output.forward(&ReLU(&self.input.forward(data)))
    }
}
