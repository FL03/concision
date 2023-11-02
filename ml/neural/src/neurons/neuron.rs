/*
    Appellation: neuron <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::activate::ActivationFn;
use ndarray::prelude::Array1;

/// Artificial Neuron
#[derive(Clone, Debug, PartialEq)]
pub struct Neuron {
    activation: ActivationFn<Array1<f64>>,
    bias: Array1<f64>,
    weights: Array1<f64>,
}

impl Neuron {
    pub fn new(
        activation: ActivationFn<Array1<f64>>,
        bias: Array1<f64>,
        weights: Array1<f64>,
    ) -> Self {
        Self {
            activation,
            bias,
            weights,
        }
    }

    pub fn bias(&self) -> &Array1<f64> {
        &self.bias
    }

    pub fn compute(&self, args: &Array1<f64>) -> Array1<f64> {
        self.rho()(args.dot(&self.weights) - self.bias())
    }

    pub fn rho(&self) -> ActivationFn<Array1<f64>> {
        self.activation
    }

    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    pub fn set_bias(&mut self, bias: Array1<f64>) {
        self.bias = bias;
    }

    pub fn set_weights(&mut self, weights: Array1<f64>) {
        self.weights = weights;
    }
}
