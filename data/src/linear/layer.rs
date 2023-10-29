/*
   Appellation: layer <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Features;
use ndarray::prelude::{Array1, Array2};
use num::Float;
use serde::{Deserialize, Serialize};



#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LinearLayer<T = f64> {
    bias: Array1<T>,
    pub params: Features,
    weights: Array2<T>,
}

impl<T> LinearLayer<T>
where
    T: Float,
{
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let params = Features::new(inputs, outputs);
        let weights = Array2::ones((params.input, params.output));
        let bias = Array1::ones(params.output);
        Self {
            bias,
            params,
            weights,
        }
    }

    pub fn bias(&self) -> &Array1<T> {
        &self.bias
    }

    pub fn linear(&self, data: &Array2<T>) -> Array2<T>
    where
        T: 'static,
    {
        data.dot(&self.weights.t()) + &self.bias
    }

    pub fn weights(&self) -> &Array2<T> {
        &self.weights
    }
}
