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
    use ndarray::prelude::Array;
    use ndarray::{Dimension, ScalarOperand};
    use num::{Float, FromPrimitive};
    use std::ops;

    pub fn mae<T, D>(pred: &Array<T, D>, target: &Array<T, D>) -> Option<T>
    where
        T: Float + FromPrimitive + ScalarOperand + ops::DivAssign,
        D: Dimension,
    {
        (target - pred).mapv(|x| x.abs()).mean()
    }

    pub fn mse<T, D>(pred: &Array<T, D>, target: &Array<T, D>) -> Option<T>
    where
        T: Float + FromPrimitive + ScalarOperand + ops::DivAssign,
        D: Dimension,
    {
        (target - pred).mapv(|x| x.powi(2)).mean()
    }
}
