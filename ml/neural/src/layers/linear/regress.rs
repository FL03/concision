/*
   Appellation: regress <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::GenerateRandom;
use crate::prelude::Forward;
use ndarray::prelude::{Array1, Array2, Axis, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_stats::CorrelationExt;
use num::{Float, FromPrimitive};
use rand::Rng;

pub enum Params {
    Layer {
        bias: Array1<f64>,    // (outputs,)
        weights: Array2<f64>, // (outputs, inputs)
    },
    Node {
        bias: f64,
        weights: Array1<f64>, // (inputs,)
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Linear<T = f64>
where
    T: Float,
{
    bias: T,
    pub features: usize,
    weights: Array1<T>,
}

impl<T> Linear<T>
where
    T: Float,
{
    pub fn new(features: usize) -> Self {
        Self {
            bias: T::zero(),
            features,
            weights: Array1::zeros(features),
        }
    }

    pub fn bias(&self) -> &T {
        &self.bias
    }

    pub fn bias_mut(&mut self) -> &mut T {
        &mut self.bias
    }

    pub fn features(&self) -> usize {
        self.features
    }

    pub fn features_mut(&mut self) -> &mut usize {
        &mut self.features
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

    pub fn set_features(&mut self, features: usize) {
        self.features = features;
    }

    pub fn set_weights(&mut self, weights: Array1<T>) {
        self.weights = weights;
    }

    pub fn with_bias(mut self, bias: T) -> Self {
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

impl<T> Linear<T>
where
    T: Float + SampleUniform,
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

impl<T> Linear<T>
where
    T: FromPrimitive + NdFloat,
{
    pub fn fit(&mut self, data: &Array2<T>, _targets: &Array1<T>) -> anyhow::Result<()> {
        let _m = data.cov(T::zero())? / data.var_axis(Axis(0), T::zero());
        // let covar = covariance(0.0, x, y);
        // self.bias = targets.mean().unwrap_or_default() - m * data.mean().unwrap_or_default();
        // self.weights -= m;
        Ok(())
    }

    pub fn predict(&self, data: &Array2<T>) -> Array1<T> {
        data.dot(&self.weights().t()) + *self.bias()
    }

    pub fn apply_gradient<G>(&mut self, gamma: T, gradient: G)
    where
        G: Fn(&Array1<T>) -> Array1<T>,
    {
        let grad = gradient(self.weights());
        self.weights_mut().scaled_add(-gamma, &grad);
    }

    pub fn update_with_gradient(&mut self, gamma: T, gradient: &Array1<T>) {
        self.weights = &self.weights - gradient * gamma;
    }
}

impl<T> Forward<Array2<T>> for Linear<T>
where
    T: NdFloat,
{
    type Output = Array1<T>;

    fn forward(&self, data: &Array2<T>) -> Self::Output {
        data.dot(&self.weights().t().to_owned()) + *self.bias()
    }
}

impl<T> Forward<Array1<T>> for Linear<T>
where
    T: NdFloat,
{
    type Output = T;

    fn forward(&self, data: &Array1<T>) -> Self::Output {
        data.dot(&self.weights().t().to_owned()) + *self.bias()
    }
}

// impl<D> Forward<Array<f64, D>> for Linear
// where
//     D: Dimension,
//     Array<f64, D>: Dot<Array<f64, D>, Output = Array<f64, D>>,
// {
//     type Output = Array1<f64>;

//     fn forward(&self, data: &Array<f64, D>) -> Self::Output {
//         data.dot(&self.weights().t().to_owned()) + self.bias().clone()
//     }
// }
