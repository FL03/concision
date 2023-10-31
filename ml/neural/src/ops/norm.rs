/*
   Appellation: norm <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array1, Array2};
use ndarray::{Axis, ScalarOperand};
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LayerNorm<T: Float = f64> {
    alpha: Array1<T>,
    beta: Array1<T>,
}

impl<T: Float> LayerNorm<T> {
    pub fn new(features: usize) -> Self {
        Self {
            alpha: Array1::ones(features),
            beta: Array1::zeros(features),
        }
    }

    pub fn forward(&self, x: &Array2<T>) -> Array2<T>
    where
        T: FromPrimitive + ScalarOperand,
    {
        let epsilon = T::from(1e-6).unwrap();
        // Calculate the mean and standard deviation of the activations along the feature axis.
        let mean = x.mean_axis(Axis(1)).unwrap();
        let std = x.std_axis(Axis(1), T::one());
        // Normalize the activations.
        let norm = (x - &mean) / (&std + epsilon);

        // Scale and shift the normalized activations with learnable parameters alpha and beta.
        &norm * &self.alpha + &self.beta
    }
}
