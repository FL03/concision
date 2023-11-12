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

use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1};
use ndarray::{Dimension, ScalarOperand};
use num::{Float, FromPrimitive};
use std::ops;

pub trait Loss<T: Float = f64> {
    fn loss<D: Dimension>(&self, pred: &Array<T, D>, target: &Array1<T>) -> T;
}

// pub type LinearWeightGradient<T = f64> = fn()

pub struct MSE;

impl MSE {
    pub fn partial_slope<T, D>(
        data: &Array<T, D>,
        bias: &Array1<T>,
        slope: &Array<T, D>,
        target: &Array1<T>,
    ) -> (T, T)
    where
        D: Dimension,
        T: Float + FromPrimitive + ScalarOperand,
        Array<T, D>: Dot<Array<T, D>>,
        <Array<T, D> as Dot<Array<T, D>>>::Output: ops::Add<Array1<T>, Output = Array<T, D>>
            + ops::Sub<Array1<T>, Output = Array<T, D>>
            + ops::Mul<T, Output = Array<T, D>>,
        Array1<T>: ops::Sub<Array<T, D>, Output = Array<T, D>>,
    {
        let predicted = data.dot(&slope.t().to_owned()) + bias.clone();
        let w = data.dot(&(target.clone() - predicted.clone())) * (-T::from(2).unwrap());
        let b = (target.clone() - predicted) * (-T::from(2).unwrap());
        (w.mean().unwrap(), b.mean().unwrap())
    }
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
