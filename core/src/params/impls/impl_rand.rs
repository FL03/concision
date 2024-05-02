/*
    Appellation: impl_rand <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

use crate::params::Parameter;
use crate::rand::GenerateRandom;
use ndarray::{Array, Dimension};
use ndrand::rand_distr::uniform::SampleUniform;
use ndrand::rand_distr::{Distribution, StandardNormal};
use num::Float;


impl<T, D> Parameter<T, D>
where
    D: Dimension,
    T: Float + SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn init_uniform(mut self, dk: T) -> Self {
        let dim = self.value.dim();
        self.value = Array::uniform_between(dk, dim);
        self
    }
}
