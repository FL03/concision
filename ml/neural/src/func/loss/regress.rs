/*
    Appellation: regress <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Loss;
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1, Array2, Dimension, NdFloat};
use num::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::ops;

pub enum RegressiveLoss {
    Huber(HuberLoss),
    MeanAbsoluteError,
    MeanSquaredError,
    Other(String),
}

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct HuberLoss<T = f64>
where
    T: Float,
{
    delta: T,
}

impl<T> HuberLoss<T>
where
    T: Float,
{
    pub fn new(delta: T) -> Self {
        Self { delta }
    }

    pub fn delta(&self) -> T {
        self.delta
    }

    pub fn set_delta(&mut self, delta: T) {
        self.delta = delta;
    }
}

impl<T, D> Loss<Array<T, D>> for HuberLoss<T>
where
    D: Dimension,
    T: NdFloat,
{
    type Output = T;

    fn loss(&self, pred: &Array<T, D>, target: &Array<T, D>) -> Self::Output {
        let half = T::from(0.5).unwrap();
        let mut loss = T::zero();
        for (x, y) in pred.iter().cloned().zip(target.iter().cloned()) {
            let diff = x - y;
            if diff.abs() <= self.delta() {
                // If the difference is sufficiently small, use the squared error.
                loss += half * diff.powi(2);
            } else {
                // Otherwise, use a variant of the absolute error.
                loss += self.delta * (diff.abs() - half * self.delta);
            }
        }
        loss / T::from(pred.len()).unwrap()
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct MeanAbsoluteError;

impl<T, D> Loss<Array<T, D>> for MeanAbsoluteError
where
    D: Dimension,
    T: Float + ops::AddAssign + ops::DivAssign,
{
    type Output = T;

    fn loss(&self, pred: &Array<T, D>, target: &Array<T, D>) -> Self::Output {
        let mut res = T::zero();
        for (p, t) in pred.iter().cloned().zip(target.iter().cloned()) {
            res += (p - t).abs();
        }
        res /= T::from(pred.len()).unwrap();
        res
    }
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct MeanSquaredError;

impl MeanSquaredError {
    pub fn new() -> Self {
        Self
    }

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

impl<T, D> Loss<Array<T, D>> for MeanSquaredError
where
    D: Dimension,
    T: NdFloat,
{
    type Output = T;

    fn loss(&self, pred: &Array<T, D>, target: &Array<T, D>) -> Self::Output {
        let res = pred
            .iter()
            .cloned()
            .zip(target.iter().cloned())
            .fold(T::zero(), |i, (p, t)| i + (p - t).powi(2));
        res / T::from(pred.len()).unwrap()
    }
}
