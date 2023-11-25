/*
   Appellation: norm <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::MatrixOps;
use crate::prelude::Forward;
use ndarray::prelude::{Array, Axis, Dimension, Ix2, NdFloat};
use ndarray::{IntoDimension, RemoveAxis};
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};

pub fn norm<T, D>(x: &Array<T, D>, axis: usize) -> Array<T, D>
where
    D: Dimension + RemoveAxis,
    T: FromPrimitive + NdFloat,
    Array<T, D>: MatrixOps<T, D::Smaller, D>,
{
    let axis = Axis(axis);
    let epsilon = T::from(1e-6).unwrap();
    // Calculate the mean and standard deviation of the activations along the feature axis.
    let mean = x.mean_axis(axis.clone()).expect("mean_axis failed");

    let std = x.std_axis(axis, T::one());
    (x.clone() - mean) / (std + epsilon)
}

pub fn norm_and_scale<T, D>(
    x: &Array<T, D>,
    alpha: &Array<T, D>,
    beta: &Array<T, D::Smaller>,
) -> Array<T, D>
where
    D: Dimension,
    T: FromPrimitive + NdFloat,
    Array<T, D>:
        Add<Array<T, D::Smaller>, Output = Array<T, D>> + Mul<Array<T, D>, Output = Array<T, D>>,
{
    let epsilon = T::from(1e-6).unwrap();
    // Calculate the mean and standard deviation of the activations along the feature axis.
    let mean = x.mean().unwrap_or_else(T::zero);
    // Normalize the activations.
    let norm = (x - mean) / (x.std(T::one()) + epsilon);

    // Scale and shift the normalized activations with learnable parameters alpha and beta.
    norm * alpha.clone() + beta.clone()
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LayerNorm<T = f64, D = Ix2>
where
    T: Float,
    D: Dimension,
{
    alpha: Array<T, D>,
    beta: Array<T, D>,
}

impl<T, D> LayerNorm<T, D>
where
    T: Float,
    D: Dimension + RemoveAxis,
{
    pub fn new(dim: impl IntoDimension<Dim = D>) -> Self {
        let dim = dim.into_dimension();
        Self {
            alpha: Array::ones(dim.clone()),
            beta: Array::zeros(dim),
        }
    }

    pub fn alpha(&self) -> &Array<T, D> {
        &self.alpha
    }

    pub fn alpha_mut(&mut self) -> &mut Array<T, D> {
        &mut self.alpha
    }

    pub fn beta(&self) -> &Array<T, D> {
        &self.beta
    }

    pub fn beta_mut(&mut self) -> &mut Array<T, D> {
        &mut self.beta
    }
}

impl<T, D> Forward<Array<T, D>> for LayerNorm<T, D>
where
    D: Dimension,
    T: FromPrimitive + NdFloat,
    Array<T, D>: Add<Array<T, D>, Output = Array<T, D>> + Mul<Array<T, D>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn forward(&self, data: &Array<T, D>) -> Self::Output {
        let epsilon = T::from(1e-6).unwrap();
        // Calculate the mean and standard deviation of the activations along the feature axis.
        let mean = data.mean().unwrap_or_else(T::zero);
        // Normalize the activations.
        let norm = (data - mean) / (data.std(T::one()) + epsilon);

        // Scale and shift the normalized activations with learnable parameters alpha and beta.
        norm * self.alpha.clone() + self.beta.clone()
    }
}
