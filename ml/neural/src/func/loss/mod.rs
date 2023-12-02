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
use ndarray::prelude::{Array, Array1, Array2, Dimension, NdFloat};
use num::{Float, FromPrimitive};
use std::ops;

pub trait Loss<T = f64> {
    type Output;

    fn loss(&self, pred: &T, target: &T) -> Self::Output;
}

pub trait LossPartial<T = f64> {
    type Output;

    fn partial(&self, pred: &T, target: &T) -> Self::Output;
}

pub struct MSE;

impl MSE {
    pub fn partial_slope<T>(
        data: &Array2<T>,
        target: &Array1<T>,
        bias: &Array1<T>,
        weights: &Array2<T>,
    ) -> T
    where
        T: FromPrimitive + NdFloat,
    {
        let pred = data.dot(&weights.t().to_owned()) + bias.clone();
        let error = target - &pred;
        let w = data.t().dot(&error) * (-T::from(2).unwrap());
        w.mean().unwrap()
    }
    pub fn partial<T, D>(
        data: &Array<T, D>,
        bias: &Array1<T>,
        slope: &Array<T, D>,
        target: &Array1<T>,
    ) -> (T, T)
    where
        D: Dimension,
        T: FromPrimitive + NdFloat,
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

#[cfg(test)]
mod tests {
    use super::regress::*;
    use super::*;
    use crate::core::GenerateRandom;
    use computare::prelude::RoundTo;
    use ndarray::prelude::{Array, Ix2};
    use ndarray_stats::DeviationExt;

    #[test]
    fn test_mse() {
        let (m, n) = (3, 3);
        let shape = (m, n);

        let ns = m * n;

        let targets: Array<f64, Ix2> = Array::linspace(0.0, ns as f64, ns)
            .into_shape(shape)
            .unwrap();
        let pred = Array::<f64, Ix2>::uniform_between(3.0, shape);

        let loss = MeanSquaredError::new().loss(&pred, &targets).round_to(4);

        let exp = targets.mean_sq_err(&pred).unwrap().round_to(4);

        assert_eq!(&loss, &exp);
    }
}
