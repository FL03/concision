/*
    Appellation: group <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Biased, Weighted};
use crate::core::prelude::GenerateRandom;
use crate::prelude::Forward;
use ndarray::prelude::{Array, Axis, Dimension, Ix1, Ix2};
use ndarray::{IntoDimension, RemoveAxis};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParamGroup<T = f64, D = Ix2>
where
    T: Float,
    D: Dimension,
{
    bias: Array<T, D::Smaller>,
    features: D,
    weights: Array<T, D>,
}

impl<T, D> ParamGroup<T, D>
where
    T: Float,
    D: Dimension + RemoveAxis,
{
    pub fn new(dim: impl IntoDimension<Dim = D>) -> Self {
        let dim = dim.into_dimension();
        Self {
            bias: Array::zeros(dim.remove_axis(Axis(dim.ndim() - 1))),
            features: dim.clone(),
            weights: Array::zeros(dim),
        }
    }
}

impl<T, D> ParamGroup<T, D>
where
    T: Float,
    D: Dimension,
{
    pub fn features(&self) -> &D {
        &self.features
    }

    pub fn inputs(&self) -> usize {
        self.weights.shape().last().unwrap().clone()
    }

    pub fn outputs(&self) -> usize {
        if self.features.ndim() == 1 {
            return 1;
        }
        self.weights.shape().first().unwrap().clone()
    }
}

impl<T, D> ParamGroup<T, D>
where
    T: Float + SampleUniform,
    D: Dimension + RemoveAxis,
{
    pub fn init(mut self, biased: bool) -> Self {
        if biased {
            self = self.init_bias();
        }
        self.init_weight()
    }

    pub fn init_bias(mut self) -> Self {
        let dk = (T::one() / T::from(self.inputs()).unwrap()).sqrt();
        self.bias = Array::uniform_between(
            dk,
            self.features()
                .remove_axis(Axis(self.features().ndim() - 1))
                .clone(),
        );
        self
    }

    pub fn init_weight(mut self) -> Self {
        let dk = (T::one() / T::from(self.inputs()).unwrap()).sqrt();
        self.weights = Array::uniform_between(dk, self.features().clone());
        self
    }
}

impl<T, D> Biased<T, D> for ParamGroup<T, D>
where
    T: Float,
    D: Dimension + RemoveAxis,
{
    fn bias(&self) -> &Array<T, D::Smaller> {
        &self.bias
    }

    fn bias_mut(&mut self) -> &mut Array<T, D::Smaller> {
        &mut self.bias
    }

    fn set_bias(&mut self, bias: Array<T, D::Smaller>) {
        self.bias = bias;
    }
}

impl<T, D> Weighted<T, D> for ParamGroup<T, D>
where
    T: Float,
    D: Dimension,
    <D as Dimension>::Smaller: Dimension,
{
    fn weights(&self) -> &Array<T, D> {
        &self.weights
    }

    fn weights_mut(&mut self) -> &mut Array<T, D> {
        &mut self.weights
    }

    fn set_weights(&mut self, weights: Array<T, D>) {
        self.weights = weights;
    }
}

impl<T> Forward<Array<T, Ix2>> for ParamGroup<T, Ix1>
where
    T: Float + 'static,
{
    type Output = Array<T, Ix1>;

    fn forward(&self, data: &Array<T, Ix2>) -> Self::Output {
        data.dot(self.weights()) + self.bias()
    }
}

impl<T> Forward<Array<T, Ix2>> for ParamGroup<T, Ix2>
where
    T: Float + 'static,
{
    type Output = Array<T, Ix2>;

    fn forward(&self, data: &Array<T, Ix2>) -> Self::Output {
        data.dot(self.weights()) + self.bias()
    }
}
