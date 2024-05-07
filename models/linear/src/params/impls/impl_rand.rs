/*
    Appellation: rand <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

use crate::bias_dim;
use crate::params::ParamsBase;
use concision::prelude::GenerateRandom;
use nd::*;
use ndrand::rand_distr::{uniform, Distribution, StandardNormal};
use num::Float;

impl<A, D> ParamsBase<OwnedRepr<A>, D>
where
    A: Float + uniform::SampleUniform,
    D: RemoveAxis,
    StandardNormal: Distribution<A>,
{
    pub fn init_uniform(mut self, biased: bool) -> Self {
        if biased {
            self = self.init_bias();
        }
        self.init_weight()
    }

    pub fn init_bias(mut self) -> Self {
        let dk = (A::one() / A::from(self.inputs()).unwrap()).sqrt();
        let dim = bias_dim(self.raw_dim());
        self.bias = Some(Array::uniform_between(dk, dim));
        self
    }

    pub fn init_weight(mut self) -> Self {
        let dk = (A::one() / A::from(self.inputs()).unwrap()).sqrt();
        self.weights = Array::uniform_between(dk, self.raw_dim());
        self
    }

    pub fn uniform(self) -> Self {
        let dk = (A::one() / A::from(self.inputs()).unwrap()).sqrt();
        let bias = if self.is_biased() {
            let dim = bias_dim(self.raw_dim());
            Some(Array::uniform_between(dk, dim))
        } else {
            None
        };
        let weights = Array::uniform_between(dk, self.raw_dim());
        Self { bias, weights }
    }
}
