/*
   Appellation: layer <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::layers::Features;
use crate::prelude::{Bias, Forward, GenerateRandom};
use ndarray::prelude::{Array1, Array2};
use ndarray::ScalarOperand;
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

    pub fn bias_mut(&mut self) -> &mut Bias<T> {
        &mut self.bias
    }

    pub fn params(&self) -> &Features {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut Features {
        &mut self.params
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

    pub fn weights_mut(&mut self) -> &mut Array2<T> {
        &mut self.weights
    }

    pub fn update_weights(&mut self, weights: Array2<T>) {
        self.weights = weights;
    }

    pub fn with_params(mut self, params: Features) -> Self {
        self.params = params;
        self
    }
}

impl<T> LinearLayer<T>
where
    T: Float + SampleUniform,
{
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let params = Features::new(inputs, outputs);
        let weights = Array2::uniform(1, (outputs, inputs));
        let bias = Bias::biased(outputs);
        Self {
            bias,
            params,
            weights,
        }
    }
}

impl<T> LinearLayer<T>
where
    T: Float + ScalarOperand,
{
    pub fn update_with_gradient(&mut self, gradient: &Array2<T>, lr: T) {
        self.weights = self.weights() + gradient * lr;
    }
}

impl<T> Forward<Array1<T>> for LinearLayer<T>
where
    T: Float + ScalarOperand,
{
    type Output = Array1<T>;

    fn forward(&self, data: &Array1<T>) -> Self::Output {
        data.dot(&self.weights().t()) + self.bias()
    }
}

impl<T> Forward<Array2<T>> for LinearLayer<T>
where
    T: Float + ScalarOperand,
{
    type Output = Array2<T>;

    fn forward(&self, data: &Array2<T>) -> Self::Output {
        data.dot(&self.weights().t()) + self.bias()
    }
}
