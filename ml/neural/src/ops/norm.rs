/*
   Appellation: norm <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::Forward;
use ndarray::prelude::{Array, Array1, NdFloat};
use ndarray::Dimension;
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LayerNorm<T = f64>
where
    T: Float,
{
    alpha: Array1<T>,
    beta: Array1<T>,
}

impl<T> LayerNorm<T>
where
    T: Float,
{
    pub fn new(features: usize) -> Self {
        Self {
            alpha: Array1::ones(features),
            beta: Array1::zeros(features),
        }
    }

    pub fn alpha(&self) -> &Array1<T> {
        &self.alpha
    }

    pub fn alpha_mut(&mut self) -> &mut Array1<T> {
        &mut self.alpha
    }

    pub fn beta(&self) -> &Array1<T> {
        &self.beta
    }

    pub fn beta_mut(&mut self) -> &mut Array1<T> {
        &mut self.beta
    }
}

impl<T> LayerNorm<T>
where
    T: FromPrimitive + NdFloat,
{
    pub fn norm_and_scale<D>(&self, x: &Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        Array<T, D>: Add<Array1<T>, Output = Array<T, D>> + Mul<Array1<T>, Output = Array<T, D>>,
    {
        let epsilon = T::from(1e-6).unwrap();
        // Calculate the mean and standard deviation of the activations along the feature axis.
        let mean = x.mean().unwrap_or_else(T::zero);
        // Normalize the activations.
        let norm = (x - mean) / (x.std(T::one()) + epsilon);

        // Scale and shift the normalized activations with learnable parameters alpha and beta.
        norm * self.alpha().clone() + self.beta().clone()
    }
}

impl<T, D> Forward<Array<T, D>> for LayerNorm<T>
where
    D: Dimension,
    T: FromPrimitive + NdFloat,
    Array<T, D>: Add<Array1<T>, Output = Array<T, D>> + Mul<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn forward(&self, x: &Array<T, D>) -> Self::Output {
        let epsilon = T::from(1e-6).unwrap();
        // Calculate the mean and standard deviation of the activations along the feature axis.
        let mean = x.mean().unwrap_or_else(T::zero);
        // Normalize the activations.
        let norm = (x - mean) / (x.std(T::one()) + epsilon);

        // Scale and shift the normalized activations with learnable parameters alpha and beta.
        norm * self.alpha().clone() + self.beta().clone()
    }
}
