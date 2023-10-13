/*
    Appellation: neuron <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::activate::ActivationFn;
use ndarray::prelude::Array1;

fn _heavyside(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}
pub struct Neuron {
    activation: ActivationFn<f64>,
    bias: f64,
    data: Array1<f64>,
    weights: Array1<f64>,
}

impl Neuron {
    pub fn new(activation: ActivationFn<f64>, bias: f64, weights: Array1<f64>) -> Self {
        Self {
            activation,
            bias,
            data: Array1::default(weights.len()),
            weights,
        }
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn dot(&self) -> f64 {
        self.data.dot(&self.weights)
    }

    pub fn compute(&self) -> f64 {
        self.rho()(self.dot() - self.bias())
    }

    pub fn rho(&self) -> ActivationFn<f64> {
        self.activation
    }

    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias;
    }

    pub fn set_data(&mut self, data: Array1<f64>) {
        self.data = data;
    }

    pub fn set_weights(&mut self, weights: Array1<f64>) {
        self.weights = weights;
    }

    pub fn with_data(mut self, data: Array1<f64>) -> Self {
        self.data = data;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn _artificial(args: &Array1<f64>, bias: Option<f64>, rho: ActivationFn<f64>, weights: &Array1<f64>) -> f64 {
        rho(args.dot(weights) - bias.unwrap_or_default())
    }

    #[test]
    fn test_neuron() {
        let bias = 0.0;

        let a_data = array![10.0, 10.0, 6.0, 1.0, 8.0];
        let a_weights = array![2.0, 1.0, 10.0, 1.0, 7.0];
        let a = Neuron::new(_heavyside, bias, a_weights.clone()).with_data(a_data.clone());

        let exp = _artificial(&a_data, Some(bias), _heavyside, &a_weights);
        assert_eq!(a.compute(), exp);

        let b_data = array![0.0, 9.0, 3.0, 5.0, 3.0];
        let b_weights = array![2.0, 8.0, 8.0, 0.0, 3.0];

        let b = Neuron::new(_heavyside, bias, b_weights.clone()).with_data(b_data.clone());

        let exp = _artificial(&b_data, Some(bias), _heavyside, &b_weights);
        assert_eq!(b.compute(), exp);

        assert_eq!(a.dot() + b.dot(), 252.0);
    }
}