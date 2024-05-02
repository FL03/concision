/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

use crate::params::LinearParams;
use crate::{Biased, Weighted};
use concision::prelude::{GenerateRandom, Predict, PredictError};
use core::ops::Add;
use nd::linalg::Dot;
use nd::*;
use num::Float;

use concision::rand::GenerateRandom;
use ndrand::rand_distr::{uniform, Distribution, StandardNormal};
use ndrand::RandomExt;

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

    pub fn uniform(&mut self) -> Self
    where
        T: Clone + Default,
    {
        let dim = dim.into_dimension();
        let bias = build_bias(biased, dim.clone(), |dim| Array::default(dim));
        Self {
            bias,
            features: dim.clone(),
            weights: Array::default(dim),
        }
    }
}
