/*
    Appellation: init <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

use crate::model::Linear;
use nd::*;
use ndrand::rand_distr::{uniform, Distribution, StandardNormal};
use num::Float;

impl<T, D> Linear<T, D>
where
    D: RemoveAxis,
    T: Float + uniform::SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn uniform(self) -> Self {
        Self {
            params: self.params.init_uniform(self.config.biased),
            ..self
        }
    }
}
