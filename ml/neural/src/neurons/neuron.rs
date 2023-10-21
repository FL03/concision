/*
    Appellation: neuron <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::activate::{ActivationFn, Activator};
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

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn compute(&self, args: &Array1<f64>) -> f64 {
        self.rho()(args.dot(&self.weights) - self.bias())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neurons::activate::{Activator, Heavyside};
    use ndarray::array;

    fn _artificial(
        args: &Array1<f64>,
        bias: Option<f64>,
        rho: impl Activator<f64>,
        weights: &Array1<f64>,
    ) -> f64 {
        rho.activate(args.dot(weights) - bias.unwrap_or_default())
    }

    #[test]
    fn test_neuron() {
        let bias = 0.0;

        let a_data = array![10.0, 10.0, 6.0, 1.0, 8.0];
        let a_weights = array![2.0, 1.0, 10.0, 1.0, 7.0];
        let a = Neuron::new(Heavyside::rho, bias, a_weights.clone());

        let exp = _artificial(&a_data, Some(bias), Heavyside, &a_weights);
        assert_eq!(a.compute(&a_data), exp);

        let b_data = array![0.0, 9.0, 3.0, 5.0, 3.0];
        let b_weights = array![2.0, 8.0, 8.0, 0.0, 3.0];

        let b = Neuron::new(Heavyside::rho, bias, b_weights.clone());

        let exp = _artificial(&b_data, Some(bias), Heavyside, &b_weights);
        assert_eq!(b.compute(&b_data), exp);

        // assert_eq!(a.dot() + b.dot(), 252.0);
    }
}
