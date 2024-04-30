/*
    Appellation: node <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::GenerateRandom;

use crate::prelude::Forward;
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array0, Array1, Array2, Dimension};
use ndarray::{LinalgScalar, RemoveAxis, ScalarOperand};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num::{Float, Num};
use std::ops::{self, Neg};

#[derive(Clone, Debug, PartialEq)]
pub struct Node<T = f64> {
    bias: Option<Array0<T>>,
    features: usize,
    weights: Array1<T>,
}

impl<T> Node<T>
where
    T: Default,
{
    pub fn create(biased: bool, features: usize) -> Self {
        let bias = if biased {
            Some(Array0::default(()))
        } else {
            None
        };
        Self {
            bias,
            features,
            weights: Array1::default(features),
        }
    }

    pub fn biased(features: usize) -> Self {
        Self::create(true, features)
    }

    pub fn new(features: usize) -> Self {
        Self::create(false, features)
    }
}
impl<T> Node<T> {
    pub fn bias(&self) -> Option<&Array0<T>> {
        self.bias.as_ref()
    }

    pub fn bias_mut(&mut self) -> Option<&mut Array0<T>> {
        self.bias.as_mut()
    }

    pub fn features(&self) -> usize {
        self.features
    }

    pub fn is_biased(&self) -> bool {
        self.bias.is_some()
    }

    pub fn set_bias(&mut self, bias: Option<Array0<T>>) {
        self.bias = bias;
    }

    pub fn set_features(&mut self, features: usize) {
        self.features = features;
    }

    pub fn set_weights(&mut self, weights: Array1<T>) {
        self.weights = weights;
    }

    pub fn weights(&self) -> &Array1<T> {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut Array1<T> {
        &mut self.weights
    }

    pub fn with_bias(mut self, bias: Option<Array0<T>>) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_features(mut self, features: usize) -> Self {
        self.features = features;
        self
    }

    pub fn with_weights(mut self, weights: Array1<T>) -> Self {
        self.weights = weights;
        self
    }
}

impl<T> Node<T>
where
    T: Num + ScalarOperand + 'static,
    Array2<T>: Dot<Array1<T>, Output = Array1<T>>,
{
    pub fn linear(&self, data: &Array2<T>) -> Array1<T> {
        let w = self.weights().t().to_owned();
        if let Some(bias) = self.bias() {
            data.dot(&w) + bias
        } else {
            data.dot(&w)
        }
    }
}
impl<T> Node<T>
where
    T: Float + SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn init(mut self, biased: bool) -> Self {
        if biased {
            self = self.init_bias();
        }
        self.init_weight()
    }

    pub fn init_bias(mut self) -> Self {
        let dk = (T::one() / T::from(self.features).unwrap()).sqrt();
        self.bias = Some(Array0::uniform_between(dk, ()));
        self
    }

    pub fn init_weight(mut self) -> Self {
        let features = self.features;
        let dk = (T::one() / T::from(features).unwrap()).sqrt();
        self.weights = Array1::uniform_between(dk, features);
        self
    }
}

impl<T> Node<T>
where
    T: LinalgScalar + Neg<Output = T>,
{
    pub fn apply_gradient<G>(&mut self, gamma: T, gradient: G)
    where
        G: Fn(&Array1<T>) -> Array1<T>,
    {
        let grad = gradient(self.weights());
        self.weights_mut().scaled_add(-gamma, &grad);
    }

    pub fn activate<A>(&self, data: &Array2<T>, activator: A) -> Array1<T>
    where
        A: Fn(&Array1<T>) -> Array1<T>,
    {
        activator(&self.forward(data))
    }
}

impl<T, D> Forward<Array<T, D>> for Node<T>
where
    D: Dimension + RemoveAxis,
    T: Clone,
    Array<T, D>: Dot<Array1<T>, Output = Array<T, D::Smaller>>,
    Array<T, D::Smaller>: ops::Add<Array0<T>, Output = Array<T, D::Smaller>>,
{
    type Output = Array<T, D::Smaller>;

    fn forward(&self, data: &Array<T, D>) -> Self::Output {
        let w = self.weights().t().to_owned();
        if let Some(bias) = self.bias() {
            return data.dot(&w) + bias.clone();
        }
        data.dot(&w)
    }
}

impl<T> FromIterator<T> for Node<T>
where
    T: Float,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let weights = Array1::<T>::from_iter(iter);
        Self {
            bias: None,
            features: weights.len(),
            weights,
        }
    }
}

impl<T> From<(Array1<T>, Array0<T>)> for Node<T> {
    fn from((weights, bias): (Array1<T>, Array0<T>)) -> Self {
        Self {
            bias: Some(bias),
            features: weights.len(),
            weights,
        }
    }
}

impl<T> From<(Array1<T>, T)> for Node<T>
where
    T: Num + ScalarOperand,
{
    fn from((weights, bias): (Array1<T>, T)) -> Self {
        Self {
            bias: Some(Array0::ones(()) * bias),
            features: weights.len(),
            weights,
        }
    }
}

impl<T> From<(Array1<T>, Option<T>)> for Node<T>
where
    T: Num + ScalarOperand,
{
    fn from((weights, bias): (Array1<T>, Option<T>)) -> Self {
        let bias = if let Some(b) = bias {
            Some(Array0::ones(()) * b)
        } else {
            None
        };
        Self {
            bias,
            features: weights.len(),
            weights,
        }
    }
}

impl<T> From<(Array1<T>, Option<Array0<T>>)> for Node<T> {
    fn from((weights, bias): (Array1<T>, Option<Array0<T>>)) -> Self {
        Self {
            bias,
            features: weights.len(),
            weights,
        }
    }
}

impl<T> From<Node<T>> for (Array1<T>, Option<Array0<T>>) {
    fn from(node: Node<T>) -> Self {
        (node.weights, node.bias)
    }
}
