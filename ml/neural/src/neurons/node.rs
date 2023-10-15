/*
    Appellation: node <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Neuron;

use ndarray::prelude::Array1;

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    data: Array1<f64>,
    neuron: Neuron,
}

impl Node {
    pub fn new(neuron: Neuron) -> Self {
        let shape = neuron.weights().shape();
        Self { data: Array1::default([shape[0]]), neuron }
    }

    pub fn data(&self) -> &Array1<f64> {
        &self.data
    }

    pub fn dot(&self) -> f64 {
        self.data.dot(self.neuron.weights())
    }

    pub fn neuron(&self) -> &Neuron {
        &self.neuron
    }

    pub fn process(&self) -> f64 {
        self.neuron.compute(&self.data)
    }

    pub fn set_data(&mut self, data: Array1<f64>) {
        self.data = data;
    }

    pub fn with_data(mut self, data: Array1<f64>) -> Self {
        self.data = data;
        self
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::neurons::activate::{heavyside, ActivationFn};
    use ndarray::array;

    fn _artificial(args: &[f64], bias: Option<f64>, rho: ActivationFn<f64>, weights: &Array1<f64>) -> f64 {
        let data = Array1::from(args.to_vec());
        rho(data.dot(weights) - bias.unwrap_or_default())
    }

    #[test]
    fn test_node() {
        let bias = 0.0;

        let a_data = [10.0, 10.0, 6.0, 1.0, 8.0];
        let a_weights = array![2.0, 1.0, 10.0, 1.0, 7.0];
        let a = Neuron::new(heavyside, bias, a_weights.clone());
        let node_a = Node::new(a.clone()).with_data(Array1::from(a_data.to_vec()));

        let exp = _artificial(&a_data, Some(bias), heavyside, &a_weights);
        assert_eq!(node_a.process(), exp);

        let b_data = [0.0, 9.0, 3.0, 5.0, 3.0];
        let b_weights = array![2.0, 8.0, 8.0, 0.0, 3.0];

        let b = Neuron::new(heavyside, bias, b_weights.clone());
        let node_b = Node::new(b.clone()).with_data(Array1::from(b_data.to_vec()));
        let exp = _artificial(&b_data, Some(bias), heavyside, &b_weights);
        assert_eq!(node_b.process(), exp);

        assert_eq!(node_a.dot() + node_b.dot(), 252.0);
    }
}