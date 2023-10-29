/*
   Appellation: norm <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::{Axis, ScalarOperand};
use ndarray::prelude::{Array1, Array2};
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};

pub fn layer_normalize<T>(x: &Array2<T>) -> Array2<T>
where
    T: Float + FromPrimitive + ScalarOperand,
{
    // Calculate the mean and variance of the activations along the feature axis.
    let mean = x.mean_axis(Axis(1)).unwrap();
    let var = x.var_axis(Axis(1), T::one());

    // Normalize the activations.
    let epsilon = T::from(1e-6).unwrap();
    let scale = (var + epsilon).mapv(|i| T::one() / i.sqrt());
    let normalized_x = (x - &mean) * &scale;

    // Scale and shift the normalized activations with learnable parameters gamma and beta.
    let gamma = Array1::ones(mean.dim());
    let beta = Array1::zeros(mean.dim());
    let scaled_shifted_x = &normalized_x * &gamma + &beta;

    scaled_shifted_x
}


#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Normalize<T: Float = f64> {
    alpha: Array1<T>,
    beta: Array1<T>,
}

impl<T: Float> Normalize<T> {
    pub fn new(features: usize) -> Self {
        Self {
            alpha: Array1::ones(features),
            beta: Array1::zeros(features),
        }
    }

    pub fn forward(&self, x: &Array2<T>) -> Array2<T> where T: FromPrimitive + ScalarOperand {
        // Calculate the mean and variance of the activations along the feature axis.
        let mean = x.mean_axis(Axis(1)).unwrap();
        let var = x.var_axis(Axis(1), T::one());

        // Normalize the activations.
        let epsilon = T::from(1e-6).unwrap();
        let scale = (var + epsilon).mapv(|i| T::one() / i.sqrt());
        let normalized_x = (x - &mean) * &scale;

        // Scale and shift the normalized activations with learnable parameters gamma and beta.
        let scaled_shifted_x = &normalized_x * &self.alpha + &self.beta;

        scaled_shifted_x
    }
}