/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]

use crate::models::Linear;
use crate::{Biased, Weighted};
use concision::prelude::{GenerateRandom, Predict, PredictError};
use core::ops::Add;
use nd::linalg::Dot;
use nd::*;
use num::Float;

use concision::rand::GenerateRandom;
use ndrand::rand_distr::{uniform, Distribution, StandardNormal};
use ndrand::RandomExt;

impl<T, D> Linear<T, D>
where
    D: RemoveAxis,
    T: Float + uniform::SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn init_uniform(self) -> Self {
        Self {
            params: self.params.init_uniform(self.is_biased()),
            ..self
        }
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
