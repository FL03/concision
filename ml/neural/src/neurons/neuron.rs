/*
    Appellation: neuron <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::activate::ActivationFn;
use ndarray::prelude::Array1;

/// Artificial Neuron
#[derive(Clone, Debug, PartialEq)]
pub struct Neuron {
    activation: ActivationFn<f64>,
    bias: f64,
    weights: Array1<f64>,
}

impl Neuron {
    pub fn new(activation: ActivationFn<f64>, bias: f64, weights: Array1<f64>) -> Self {
        Self {
            activation,
            bias,
            weights,
        }
    }

    pub fn bias(&self) -> &f64 {
        &self.bias
    }

    pub fn process(&self, args: &Array1<f64>) -> f64 {
        self.rho()(args.dot(&self.weights.t()) + self.bias())
    }

    pub fn rho(&self) -> ActivationFn<f64> {
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
}
