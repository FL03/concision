/*
    Appellation: regress <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Loss;
use ndarray::{Dimension, ScalarOperand};
use ndarray::prelude::{Array, Array1};
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

impl<T> HuberLoss<T> where T: Float {
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

impl<T> Loss<T> for HuberLoss<T> where T: Float + ops::AddAssign {

    fn loss<D: Dimension>(&self, pred: &Array<T, D>, target: &Array1<T>) -> T {
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

impl<T> Loss<T> for MeanAbsoluteError where T: Float + ops::AddAssign + ops::DivAssign {
    fn loss<D: Dimension>(&self, pred: &Array<T, D>, target: &Array1<T>) -> T {
        let mut res = T::zero();
        for (p, t) in pred.iter().cloned().zip(target.iter().cloned()) {
            res += (p - t).abs();
        }
        res /= T::from(pred.len()).unwrap();
        res
    }
}

pub struct MeanSquaredError;

impl<T> Loss<T> for MeanSquaredError where T: Float + ops::AddAssign + ops::DivAssign {
    fn loss<D: Dimension>(&self, pred: &Array<T, D>, target: &Array1<T>) -> T {
        let mut res = T::zero();
        for (p, t) in pred.iter().cloned().zip(target.iter().cloned()) {
            res += (p - t).powi(2);
        }
        res /= T::from(pred.len()).unwrap();
        res
    }
}
