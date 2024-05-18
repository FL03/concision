/*
    Appellation: impl_rand <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::Parameter;
use concision::InitializeExt;
use ndarray::{Array, Dimension};
use concision::rand::rand_distr::uniform::SampleUniform;
use concision::rand::rand_distr::{Distribution, StandardNormal};
use num::Float;

impl<T, D> Parameter<T, D>
where
    D: Dimension,
    T: Float + SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn init_uniform(mut self, dk: T) -> Self {
        let dim = self.value.dim();
        self.value = Array::uniform(dim, dk);
        self
    }
}
