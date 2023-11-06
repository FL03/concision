/*
    Appellation: loss <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Loss Functions
//!
//! Loss functions consider the differences between predicted and target outputs.
//! Overall, neural network models aim to minimize the average loss by adjusting certain hyperparameters,
//! the weights and biases.

pub use self::{kinds::*, utils::*};

pub(crate) mod kinds;

pub mod regress;

pub trait Loss {
    fn loss(&self, pred: &[f64], target: &[f64]) -> f64;
}

pub(crate) mod utils {
    use ndarray::prelude::{Array, Array1};
    use ndarray::{Dimension, ScalarOperand};
    use num::{Float, FromPrimitive};
    use std::ops;

    pub fn mae<'a, T, D>(pred: &Array<T, D>, target: &Array1<T>) -> Option<T>
    where
        T: Float + FromPrimitive + ScalarOperand,
        D: Dimension,
        Array1<T>: ops::Sub<Array<T, D>, Output = Array<T, D>>,
    {
        (target.clone() - pred.clone()).mapv(|x| x.abs()).mean()
    }

    pub fn mse<T, D>(pred: &Array<T, D>, target: &Array1<T>) -> Option<T>
    where
        T: Float + FromPrimitive + ScalarOperand,
        D: Dimension,
        Array1<T>: ops::Sub<Array<T, D>, Output = Array<T, D>>,
    {
        (target.clone() - pred.clone()).mapv(|x| x.powi(2)).mean()
    }
}
