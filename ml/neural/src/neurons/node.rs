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
        Self {
            data: Array1::default([shape[0]]),
            neuron,
        }
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

    pub fn set_data(&mut self, data: Array1<f64>) {
        self.data = data;
    }

    pub fn with_data(mut self, data: Array1<f64>) -> Self {
        self.data = data;
        self
    }
}
