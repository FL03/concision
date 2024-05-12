/*
    Appellation: impl_rand <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

use crate::params::ParamsBase;
use crate::{bias_dim, Linear};
use concision::prelude::GenerateRandom;
use concision::rand::rand_distr::{uniform, Distribution, StandardNormal};
use nd::*;
use num::Float;

impl<A, D, K> Linear<A, K, D>
where
    A: Float + uniform::SampleUniform,
    D: RemoveAxis,
    StandardNormal: Distribution<A>,
{
    pub fn uniform(self) -> Self
    where
        K: 'static,
    {
        let biased = self.is_biased();
        Self {
            params: self.params.init_uniform(biased),
            ..self
        }
    }
}

impl<A, D, K> ParamsBase<OwnedRepr<A>, D, K>
where
    A: Float + uniform::SampleUniform,
    D: RemoveAxis,
    StandardNormal: Distribution<A>,
{
    pub(crate) fn dk(&self) -> A {
        A::from(self.in_features()).unwrap().recip().sqrt()
    }

    pub fn init_uniform(mut self, biased: bool) -> Self {
        if biased {
            self = self.init_bias();
        }
        self.init_weight()
    }

    pub fn init_bias(mut self) -> Self {
        let dim = bias_dim(self.raw_dim());
        self.bias = Some(Array::uniform_between(self.dk(), dim));
        self
    }

    pub fn init_weight(mut self) -> Self {
        self.weights = Array::uniform_between(self.dk(), self.raw_dim());
        self
    }

    pub fn uniform(self) -> Self {
        let dk = self.dk();
        let bias = if self.is_biased() {
            let dim = bias_dim(self.raw_dim());
            Some(Array::uniform_between(dk, dim))
        } else {
            None
        };
        let weights = Array::uniform_between(dk, self.raw_dim());
        Self {
            bias,
            weights,
            _mode: self._mode,
        }
    }
}
