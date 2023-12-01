/*
    Appellation: regress <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Loss;
use ndarray::prelude::{Array, Dimension, NdFloat};
use num::Float;
use std::ops;

pub enum RegressiveLoss {
    Huber(HuberLoss),
    MeanAbsoluteError,
    MeanSquaredError,
    Other(String),
}

pub struct HuberLoss<T: Float = f64> {
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

pub struct MeanSquaredError;

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
