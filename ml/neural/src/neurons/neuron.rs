/*
    Appellation: neuron <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::activate::{Activate, ActivationFn, LinearActivation};
use crate::prelude::Forward;
use crate::{core::GenerateRandom, layers::L};
use ndarray::prelude::{Array1, Array2};

pub trait ArtificialNeuron<T> {
    type Rho: Activate<T>;
}

/// Artificial Neuron
#[derive(Clone, Debug, PartialEq)]
pub struct Neuron<Rho = LinearActivation>
where
    Rho: Activate<Array1<f64>>,
{
    activation: Rho,
    bias: f64,
    features: usize,
    weights: Array1<f64>,
}

impl<Rho> Neuron<Rho>
where
    Rho: Activate<Array1<f64>>,
{
    pub fn with_rho(mut self, rho: Rho) -> Self {
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
        self.rho()
            .activate(args.dot(&self.weights.t()) + self.bias())
    }

    pub fn rho(&self) -> &Rho {
        &self.activation
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

impl<Rho> Neuron<Rho>
where
    Rho: Activate<Array1<f64>> + Default,
{
    pub fn new(features: usize) -> Self {
        Self {
            activation: Rho::default(),
            bias: 0.0,
            features,
            weights: Array1::zeros(features),
        }
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
