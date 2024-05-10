/*
    Appellation: init <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

use crate::model::{Linear, ParamMode};
use nd::RemoveAxis;
use ndrand::rand_distr::{uniform, Distribution, StandardNormal};
use num::Float;

impl<A, D, K> Linear<A, D, K>
where
    A: Float + uniform::SampleUniform,
    D: RemoveAxis,
    K: ParamMode,
    StandardNormal: Distribution<A>,
{
    pub fn uniform(self) -> Self {
        let biased = self.is_biased();
        Self {
            params: self.params.init_uniform(biased),
            ..self
        }
    }
}
