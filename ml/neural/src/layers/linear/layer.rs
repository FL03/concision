/*
   Appellation: layer <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::GenerateRandom;
use crate::layers::Features;
use crate::prelude::{Bias, Forward};

use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array2};
use ndarray::{Dimension, ScalarOperand};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LinearLayer<T: Float = f64> {
    bias: Bias<T>,
    pub features: Features,
    weights: Array2<T>,
}

impl<T> LinearLayer<T>
where
    T: Float + ScalarOperand,
{
    pub fn bias(&self) -> &Bias<T> {
        &self.bias
    }

    pub fn bias_mut(&mut self) -> &mut Bias<T> {
        &mut self.bias
    }

    pub fn features(&self) -> &Features {
        &self.features
    }

    pub fn features_mut(&mut self) -> &mut Features {
        &mut self.features
    }

    pub fn fit(&mut self, data: &Array2<T>) -> Array2<T>
    where
        T: 'static,
    {
        self.linear(data)
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
        self.features = params;
        self
    }
}

impl<T> LinearLayer<T>
where
    T: Float + SampleUniform,
{
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let features = Features::new(inputs, outputs);
        let weights = Array2::uniform(1, (outputs, inputs));
        let bias = Bias::biased(outputs);
        Self {
            bias,
            features,
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

impl<T, D> Forward<Array<T, D>> for LinearLayer<T>
where
    D: Dimension,
    T: Float + ScalarOperand,
    Array<T, D>: Add<Bias<T>, Output = Array<T, D>> + Dot<Array2<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn forward(&self, data: &Array<T, D>) -> Self::Output {
        data.dot(&self.weights().t().to_owned()) + self.bias().clone()
    }
}

impl<T> Forward<T> for LinearLayer<T>
where
    T: Float + ScalarOperand + Mul<T, Output = T>,
{
    type Output = Array2<T>;

    fn forward(&self, data: &T) -> Self::Output {
        self.weights().t().to_owned() * data.clone() + self.bias().clone()
    }
}
