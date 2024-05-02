/*
    Appellation: rand <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

use crate::params::LinearParams;
use concision::prelude::GenerateRandom;
use nd::*;
use ndrand::rand_distr::{uniform, Distribution, StandardNormal};
use num::Float;

impl<T, D> LinearParams<T, D>
where
    D: RemoveAxis,
    T: Float + uniform::SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn init_uniform(mut self, biased: bool) -> Self {
        if biased {
            self = self.init_bias();
        }
        self.init_weight()
    }

    pub fn init_bias(mut self) -> Self {
        let dk = (T::one() / T::from(self.inputs()).unwrap()).sqrt();
        let dim = self
            .features()
            .remove_axis(Axis(self.features().ndim() - 1));
        self.bias = Some(Array::uniform_between(dk, dim));
        self
    }

    pub fn init_weight(mut self) -> Self {
        let dk = (T::one() / T::from(self.inputs()).unwrap()).sqrt();
        self.weights = Array::uniform_between(dk, self.features().clone());
        self
    }
}
