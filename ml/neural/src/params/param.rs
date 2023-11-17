/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::GenerateRandom;
use crate::layers::Features;
use ndarray::prelude::{Array1, Array2};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};

pub trait ParamFeatures {
    fn inputs(&self) -> usize;

    fn outputs(&self) -> usize;

    fn in_by_out(&self) -> (usize, usize) {
        (self.inputs(), self.outputs())
    }

    fn out_by_in(&self) -> (usize, usize) {
        (self.outputs(), self.inputs())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ParamGroup<T = f64>
where
    T: Float,
{
    bias: Array1<T>,
    pub features: Features,
    weights: Array2<T>,
}

impl<T> ParamGroup<T>
where
    T: Float,
{
    pub fn new(features: Features) -> Self {
        Self {
            bias: Array1::zeros(features.outputs()),
            features,
            weights: Array2::zeros(features.out_by_in()),
        }
    }

    pub fn reset(&mut self) {
        self.bias = Array1::zeros(self.features.outputs());
        self.weights = Array2::zeros(self.features.out_by_in());
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

    pub fn set_weights(&mut self, weights: Array2<T>) {
        self.weights = weights;
    }

    pub fn with_bias(mut self, bias: Array1<T>) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_weights(mut self, weights: Array2<T>) -> Self {
        self.weights = weights;
        self
    }
}

impl<T> ParamGroup<T>
where
    T: Float + SampleUniform,
{
    pub fn init(mut self, biased: bool) -> Self {
        if biased {
            self = self.init_bias();
        }
        self.init_weight()
    }

    pub fn init_bias(mut self) -> Self {
        let dk = (T::one() / T::from(self.features().inputs()).unwrap()).sqrt();
        self.bias = Array1::uniform_between(dk, self.features().outputs());
        self
    }

    pub fn init_weight(mut self) -> Self {
        let dk = (T::one() / T::from(self.features().inputs()).unwrap()).sqrt();
        self.weights = Array2::uniform_between(dk, self.features().out_by_in());
        self
    }
}
