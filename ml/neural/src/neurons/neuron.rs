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

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn compute(&self, args: &Array1<f64>) -> f64 {
        let dot = args.dot(&self.weights);
        self.rho()(dot - self.bias())
    }

    pub fn process(&self, args: impl AsRef<[f64]>) -> f64 {
        let data = Array1::from(args.as_ref().to_vec());
        let dot = data.dot(&self.weights);
        self.rho()(dot - self.bias())
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
    use crate::neurons::activate::heavyside;
    use ndarray::array;

    fn _artificial(
        args: &[f64],
        bias: Option<f64>,
        rho: ActivationFn<f64>,
        weights: &Array1<f64>,
    ) -> f64 {
        let data = Array1::from(args.to_vec());
        rho(data.dot(weights) - bias.unwrap_or_default())
    }

    #[test]
    fn test_neuron() {
        let bias = 0.0;

        let a_data = [10.0, 10.0, 6.0, 1.0, 8.0];
        let a_weights = array![2.0, 1.0, 10.0, 1.0, 7.0];
        let a = Neuron::new(heavyside, bias, a_weights.clone());

        let exp = _artificial(&a_data, Some(bias), heavyside, &a_weights);
        assert_eq!(a.process(&a_data), exp);

        let b_data = [0.0, 9.0, 3.0, 5.0, 3.0];
        let b_weights = array![2.0, 8.0, 8.0, 0.0, 3.0];

        let b = Neuron::new(heavyside, bias, b_weights.clone());

        let exp = _artificial(&b_data, Some(bias), heavyside, &b_weights);
        assert_eq!(b.process(&b_data), exp);

        // assert_eq!(a.dot() + b.dot(), 252.0);
    }
}
