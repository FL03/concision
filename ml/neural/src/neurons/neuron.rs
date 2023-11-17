/*
    Appellation: neuron <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::activate::{Activate, LinearActivation};
use crate::core::GenerateRandom;
use crate::prelude::Forward;
use ndarray::prelude::{Array1, Array2, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use rand::Rng;

/// Artificial Neuron
#[derive(Clone, Debug, PartialEq)]
pub struct Neuron<T = f64, A = LinearActivation>
where
    A: Activate<Array1<T>>,
{
    activation: A,
    bias: T,
    features: usize,
    weights: Array1<T>,
}

impl<T, A> Neuron<T, A>
where
    A: Activate<Array1<T>>,
    T: Float,
{
    pub fn bias(&self) -> &T {
        &self.bias
    }

    pub fn rho(&self) -> &A {
        &self.activation
    }

    pub fn weights(&self) -> &Array1<T> {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut Array1<T> {
        &mut self.weights
    }

    pub fn set_bias(&mut self, bias: T) {
        self.bias = bias;
    }

    pub fn set_weights(&mut self, weights: Array1<T>) {
        self.weights = weights;
    }

    pub fn with_bias(mut self, bias: T) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_rho(mut self, rho: A) -> Self {
        self.activation = rho;
        self
    }

    pub fn with_weights(mut self, weights: Array1<T>) -> Self {
        self.weights = weights;
        self
    }
}

impl<T, A> Neuron<T, A>
where
    T: NdFloat,
    A: Activate<Array1<T>> + Default,
{
    pub fn new(features: usize) -> Self {
        Self {
            activation: A::default(),
            bias: T::zero(),
            features,
            weights: Array1::zeros(features),
        }
    }
}

impl<T, A> Neuron<T, A>
where
    T: NdFloat,
    A: Activate<Array1<T>>,
{
    pub fn apply_gradient<G>(&mut self, gamma: T, gradient: G)
    where
        G: Fn(&Array1<T>) -> Array1<T>,
    {
        let grad = gradient(&self.weights);
        self.weights_mut().scaled_add(-gamma, &grad);
    }
}

impl<T, A> Neuron<T, A>
where
    T: Float + SampleUniform,
    A: Activate<Array1<T>>,
{
    pub fn init(mut self, biased: bool) -> Self {
        if biased {
            self = self.init_bias();
        }
        self.init_weight()
    }

    pub fn init_bias(mut self) -> Self {
        let dk = (T::one() / T::from(self.features).unwrap()).sqrt();
        self.bias = rand::thread_rng().gen_range(-dk..dk);
        self
    }

    pub fn init_weight(mut self) -> Self {
        let features = self.features;
        let dk = (T::one() / T::from(features).unwrap()).sqrt();
        self.weights = Array1::uniform_between(dk, features);
        self
    }
}

// impl Forward<Array1<f64>> for Neuron {
//     type Output = f64;

//     fn forward(&self, args: &Array1<f64>) -> Self::Output {
//         self.rho().activate(args.dot(&self.weights().t().to_owned()) + self.bias)
//     }

// }

impl<T, A> Forward<Array2<T>> for Neuron<T, A>
where
    T: NdFloat,
    A: Activate<Array1<T>>,
{
    type Output = Array1<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        let linstep = args.dot(&self.weights().t()) + self.bias;
        self.rho().activate(linstep)
    }
}
