/*
    Appellation: neuron <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::activate::{Activate, ActivationFn};
use crate::core::GenerateRandom;
use crate::prelude::Forward;
use ndarray::prelude::{Array1, Array2};

/// Artificial Neuron
#[derive(Clone, Debug, PartialEq)]
pub struct Neuron {
    activation: ActivationFn<Array1<f64>>,
    bias: f64,
    features: usize,
    weights: Array1<f64>,
}

impl Neuron {
    pub fn new(features: usize) -> Self {
        Self {
            activation: |x| x,
            bias: 0.0,
            features,
            weights: Array1::zeros(features),
        }
    }

    pub fn with_rho(mut self, rho: ActivationFn<Array1<f64>>) -> Self {
        self.activation = rho;
        self
    }

    pub fn init_weights(mut self) -> Self {
        self.weights = Array1::uniform(0, self.features);
        self
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn process(&self, args: &Array2<f64>) -> Array1<f64> {
        self.rho()(args.dot(&self.weights.t()) + self.bias())
    }

    pub fn rho(&self) -> ActivationFn<Array1<f64>> {
        self.activation
    }

    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }

    pub fn set_weights(&mut self, weights: Array1<f64>) {
        self.weights = weights;
    }

    pub fn with_bias(mut self, bias: f64) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_weights(mut self, weights: Array1<f64>) -> Self {
        self.weights = weights;
        self
    }

    pub fn apply_weight_gradient(&mut self, gamma: f64, gradient: &Array1<f64>) {
        self.weights = &self.weights - gamma * gradient;
    }
}

// impl Forward<Array1<f64>> for Neuron {
//     type Output = f64;

//     fn forward(&self, args: &Array1<f64>) -> Self::Output {
//         self.rho().activate(args.dot(&self.weights().t().to_owned()) + self.bias)
//     }

// }

impl Forward<Array2<f64>> for Neuron {
    type Output = Array1<f64>;

    fn forward(&self, args: &Array2<f64>) -> Self::Output {
        let linstep = args.dot(&self.weights().t().to_owned()) + self.bias;
        self.rho().activate(linstep)
    }
}
