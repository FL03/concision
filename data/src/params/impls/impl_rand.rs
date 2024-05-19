/*
    Appellation: impl_rand <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::Parameter;
use concision::init::rand_distr::uniform::SampleUniform;
use concision::init::rand_distr::{Distribution, StandardNormal};
use concision::InitializeExt;
use ndarray::{Array, Dimension};
use num::Float;

impl<T, D> Parameter<T, D>
where
    D: Dimension,
    T: Float,
    StandardNormal: Distribution<T>,
{
    pub fn init_uniform(mut self, dk: T) -> Self
    where
        T: SampleUniform,
        <T as SampleUniform>::Sampler: Clone,
    {
        let dim = self.value.dim();
        self.value = Array::uniform(dim, dk);
        self
    }
}
