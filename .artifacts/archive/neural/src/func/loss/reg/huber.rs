/*
    Appellation: regress <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::loss::Loss;
use ndarray::prelude::{Array, Dimension, NdFloat};
use num::Float;
use serde::{Deserialize, Serialize};

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

    pub fn delta_mut(&mut self) -> &mut T {
        &mut self.delta
    }

    pub fn set_delta(&mut self, delta: T) {
        self.delta = delta;
    }

    pub fn with_delta(mut self, delta: T) -> Self {
        self.delta = delta;
        self
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
                loss += self.delta * (diff.abs() - half * self.delta());
            }
        }
        loss / T::from(pred.len()).unwrap()
    }
}

impl<T> From<T> for HuberLoss<T>
where
    T: Float,
{
    fn from(delta: T) -> Self {
        Self::new(delta)
    }
}
