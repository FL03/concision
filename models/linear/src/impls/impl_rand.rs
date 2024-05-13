/*
    Appellation: impl_rand <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

use crate::params::ParamsBase;
use crate::{bias_dim, Linear};
use concision::prelude::InitializeExt;
use concision::rand::rand_distr::{uniform, Distribution, StandardNormal};
use nd::*;
use num::Float;

impl<A, D, K> Linear<A, K, D>
where
    A: Float + uniform::SampleUniform,
    D: RemoveAxis,
    K: 'static,
    StandardNormal: Distribution<A>,
{
    pub fn uniform(self) -> Self {
        Self {
            params: self.params.uniform(),
            ..self
        }
    }

    pub fn uniform_between(self, low: A, high: A) -> Self {
        Self {
            params: self.params.uniform_between(low, high),
            ..self
        }
    }
}

impl<A, D, K> ParamsBase<OwnedRepr<A>, D, K>
where
    A: Float + uniform::SampleUniform,
    D: RemoveAxis,
    K: 'static,
    StandardNormal: Distribution<A>,
{
    /// Computes the reciprocal of the input features.
    pub(crate) fn dk(&self) -> A {
        A::from(self.in_features()).unwrap().recip()
    }
    /// Computes the square root of the reciprical of the input features.
    pub(crate) fn dk_sqrt(&self) -> A {
        self.dk().sqrt()
    }

    pub fn uniform(self) -> Self {
        let dk = self.dk_sqrt();
        self.uniform_between(-dk, dk)
    }

    pub fn uniform_between(self, low: A, high: A) -> Self {
        if self.is_biased() && !self.bias.is_some() {
            let b_dim = bias_dim(self.raw_dim());
            Self {
                bias: Some(Array::uniform_between(b_dim, low, high)),
                weights: Array::uniform_between(self.raw_dim(), low, high),
                _mode: self._mode,
            }
        } else if !self.is_biased() && self.bias.is_some() {
            Self {
                bias: None,
                weights: Array::uniform_between(self.raw_dim(), low, high),
                _mode: self._mode,
            }
        } else {
            Self {
                bias: self
                    .bias
                    .as_ref()
                    .map(|b| Array::uniform_between(b.raw_dim(), low, high)),
                weights: Array::uniform_between(self.raw_dim(), low, high),
                _mode: self._mode,
            }
        }
    }
}
