/*
    Appellation: mse <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::loss::Loss;
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1, Array2, Dimension, NdFloat};
use num::FromPrimitive;
use serde::{Deserialize, Serialize};
use std::ops;

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct MeanSquaredError;

impl MeanSquaredError {
    pub fn new() -> Self {
        Self
    }

    pub fn mse<T, D>(&self, pred: &Array<T, D>, target: &Array<T, D>) -> T
    where
        D: Dimension,
        T: FromPrimitive + NdFloat,
    {
        (pred - target).mapv(|x| x.powi(2)).mean().unwrap()
    }

    pub fn wg<T, D>(
        data: &Array2<T>,
        target: &Array<T, D>,
        bias: &Array<T, D::Smaller>,
        weights: &Array<T, D>,
    ) -> T
    where
        D: Dimension,
        T: FromPrimitive + NdFloat,
        Array2<T>: Dot<Array<T, D>, Output = Array<T, D>>,
    {
        let pred = data.dot(&weights.t().to_owned()) + bias.clone();
        let error = target - &pred;
        let dw = data.t().to_owned().dot(&error) * (-T::from(2).unwrap());
        dw.mean().unwrap()
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

impl<T, D> Loss<Array<T, D>> for MeanSquaredError
where
    D: Dimension,
    T: FromPrimitive + NdFloat,
{
    type Output = T;

    fn loss(&self, pred: &Array<T, D>, target: &Array<T, D>) -> Self::Output {
        (pred - target).mapv(|x| x.powi(2)).mean().unwrap()
    }
}
