/*
   Appellation: layer <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::layers::Features;
use crate::prelude::{Bias, Forward};
use ndarray::prelude::Array2;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LinearLayer<T: Float = f64> {
    bias: Bias<T>,
    pub params: Features,
    weights: Array2<T>,
}

impl<T> LinearLayer<T>
where
    T: Float,
{
    pub fn bias(&self) -> &Bias<T> {
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

impl<T> LinearLayer<T>
where
    T: Float + SampleUniform,
{
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let params = Features::new(inputs, outputs);
        let weights = Array2::ones((inputs, outputs));
        let bias = Bias::biased(outputs);
        Self {
            bias,
            params,
            weights,
        }
    }
}

impl<T: Float + 'static> Forward<Array2<T>> for LinearLayer<T> {
    type Output = Array2<T>;

    fn forward(&self, data: &Array2<T>) -> Self::Output {
        data.dot(&self.weights().t()) + self.bias()
    }
}
