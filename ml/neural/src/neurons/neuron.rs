/*
    Appellation: neuron <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::GenerateRandom;
use crate::func::activate::{Activate, Linear};
use crate::prelude::{Biased, Forward, Weighted};
use ndarray::prelude::{Array0, Array1, Array2, Ix1, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;

/// Artificial Neuron
#[derive(Clone, Debug, PartialEq)]
pub struct Neuron<T = f64, A = Linear>
where
    A: Activate<T, Ix1>,
    T: Float,
{
    activation: A,
    bias: Array0<T>,
    features: usize,
    weights: Array1<T>,
}

impl<T, A> Neuron<T, A>
where
    A: Activate<T, Ix1>,
    T: Float,
{
    pub fn rho(&self) -> &A {
        &self.activation
    }

    pub fn with_bias(mut self, bias: Array0<T>) -> Self {
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
    A: Activate<T, Ix1> + Default,
{
    pub fn new(features: usize) -> Self {
        Self {
            activation: A::default(),
            bias: Array0::zeros(()),
            features,
            weights: Array1::zeros(features),
        }
    }
}

impl<T, A> Neuron<T, A>
where
    T: NdFloat,
    A: Activate<T, Ix1>,
{
    pub fn apply_gradient<G>(&mut self, gamma: T, gradient: G)
    where
        G: Fn(&Array1<T>) -> Array1<T>,
    {
        let grad = gradient(&self.weights);
        self.update_with_gradient(gamma, &grad);
    }

    pub fn update_with_gradient(&mut self, gamma: T, grad: &Array1<T>) {
        self.weights_mut().scaled_add(-gamma, grad);
    }
}

impl<T, A> Neuron<T, A>
where
    T: Float + SampleUniform,
    A: Activate<T, Ix1>,
{
    pub fn init(mut self, biased: bool) -> Self {
        if biased {
            self = self.init_bias();
        }
        self.init_weight()
    }

    pub fn init_bias(mut self) -> Self {
        let dk = (T::one() / T::from(self.features).unwrap()).sqrt();
        self.bias = Array0::uniform_between(dk, ());
        self
    }

    pub fn init_weight(mut self) -> Self {
        let features = self.features;
        let dk = (T::one() / T::from(features).unwrap()).sqrt();
        self.weights = Array1::uniform_between(dk, features);
        self
    }
}

impl<T, A> Biased<T, Ix1> for Neuron<T, A>
where
    T: NdFloat,
    A: Activate<T, Ix1>,
{
    fn bias(&self) -> &Array0<T> {
        &self.bias
    }

    fn bias_mut(&mut self) -> &mut Array0<T> {
        &mut self.bias
    }

    fn set_bias(&mut self, bias: Array0<T>) {
        self.bias = bias;
    }
}

impl<T, A> Weighted<T, Ix1> for Neuron<T, A>
where
    T: NdFloat,
    A: Activate<T, Ix1>,
{
    fn weights(&self) -> &Array1<T> {
        &self.weights
    }

    fn weights_mut(&mut self) -> &mut Array1<T> {
        &mut self.weights
    }

    fn set_weights(&mut self, weights: Array1<T>) {
        self.weights = weights;
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
    A: Activate<T, Ix1>,
{
    type Output = Array1<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        let linstep = args.dot(&self.weights().t()) + self.bias();
        self.rho().activate(&linstep)
    }
}
