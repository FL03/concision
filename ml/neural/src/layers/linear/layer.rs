/*
   Appellation: layer <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::GenerateRandom;
use crate::prelude::{Features, Forward};

use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1, Array2, NdFloat};
use ndarray::Dimension;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};
use std::ops::Add;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LinearLayer<T: Float = f64> {
    bias: Array1<T>,
    pub features: Features,
    weights: Array2<T>,
}

impl<T> LinearLayer<T>
where
    T: Float,
{
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            bias: Array1::zeros(outputs),
            features: Features::new(inputs, outputs),
            weights: Array2::zeros((inputs, outputs)),
        }
    }
    pub fn from_features(features: Features) -> Self {
        Self {
            bias: Array1::zeros(features.outputs()),
            features,
            weights: Array2::zeros(features.out_by_in()),
        }
    }

    pub fn bias(&self) -> &Array1<T> {
        &self.bias
    }

    pub fn bias_mut(&mut self) -> &mut Array1<T> {
        &mut self.bias
    }

    pub fn features(&self) -> &Features {
        &self.features
    }

    pub fn features_mut(&mut self) -> &mut Features {
        &mut self.features
    }

    pub fn weights(&self) -> &Array2<T> {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut Array2<T> {
        &mut self.weights
    }

    pub fn set_bias(&mut self, bias: Array1<T>) {
        self.bias = bias;
    }

    pub fn set_features(&mut self, features: Features) {
        self.features = features;
    }

    pub fn set_weights(&mut self, weights: Array2<T>) {
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
    pub fn init_bias(mut self) -> Self {
        let dk = (T::one() / T::from(self.features().inputs()).unwrap()).sqrt();
        self.bias = ndarray::Array1::uniform_between(dk, self.features().outputs());
        self
    }

    pub fn init_weight(mut self) -> Self {
        let dk = (T::one() / T::from(self.features().inputs()).unwrap()).sqrt();
        self.weights = Array2::uniform_between(dk, self.features().out_by_in());
        self
    }
}

impl<T> LinearLayer<T>
where
    T: NdFloat,
{
    pub fn fit(&mut self, data: &Array2<T>) -> Array2<T>
    where
        T: 'static,
    {
        self.forward(data)
    }

    pub fn update_with_gradient(&mut self, gradient: &Array2<T>, lr: T) {
        self.weights = &self.weights - gradient * lr;
    }

    pub fn apply_gradient(&mut self, gradient: &Array1<T>, gamma: T) {
        for (ws, g) in self.weights_mut().iter_mut().zip(gradient.iter()) {
            *ws -= *g * gamma;
        }
    }
}

impl<S, T, D> Forward<Array<T, D>> for LinearLayer<T>
where
    D: Dimension,
    S: Dimension,
    T: NdFloat,
    Array<T, D>: Add<Array1<T>, Output = Array<T, S>> + Dot<Array2<T>, Output = Array<T, S>>,
    Array<T, S>: Add<Array1<T>, Output = Array<T, S>>,
{
    type Output = Array<T, S>;

    fn forward(&self, data: &Array<T, D>) -> Self::Output {
        data.dot(&self.weights().t().to_owned()) + self.bias().clone()
    }
}

impl<T> Forward<T> for LinearLayer<T>
where
    T: NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, data: &T) -> Self::Output {
        &self.weights().t().to_owned() * data.clone() + self.bias().clone()
    }
}
