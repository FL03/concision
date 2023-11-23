/*
   Appellation: weight <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::GenerateRandom;
use ndarray::prelude::Array2;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};

pub enum WeightShape {}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Weight<T = f64> {
    weights: Array2<T>,
}

impl<T> Weight<T>
where
    T: Float,
{
    pub fn new(inputs: usize, outputs: Option<usize>) -> Self {
        Self {
            weights: Array2::zeros((outputs.unwrap_or(1), inputs)),
        }
    }
    /// Returns the shape of the weights. (outputs, inputs)
    pub fn shape(&self) -> (usize, usize) {
        self.weights.dim()
    }

    pub fn inputs(&self) -> usize {
        self.shape().1
    }

    pub fn outputs(&self) -> usize {
        self.shape().0
    }
}

impl<T> Weight<T>
where
    T: Float + SampleUniform,
{
    pub fn init(mut self) -> Self {
        let dk = (T::one() / T::from(self.inputs()).unwrap()).sqrt();
        self.weights = Array2::uniform_between(dk, self.shape());
        self
    }
}

impl<T> AsRef<Array2<T>> for Weight<T> {
    fn as_ref(&self) -> &Array2<T> {
        &self.weights
    }
}

impl<T> AsMut<Array2<T>> for Weight<T> {
    fn as_mut(&mut self) -> &mut Array2<T> {
        &mut self.weights
    }
}
