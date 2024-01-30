/*
    Appellation: mae <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::loss::Loss;
use ndarray::prelude::{Array, Dimension, NdFloat};
use num::FromPrimitive;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct MeanAbsoluteError;

impl MeanAbsoluteError {
    pub fn new() -> Self {
        Self
    }
}

impl<T, D> Loss<Array<T, D>> for MeanAbsoluteError
where
    D: Dimension,
    T: FromPrimitive + NdFloat,
{
    type Output = T;

    fn loss(&self, pred: &Array<T, D>, target: &Array<T, D>) -> Self::Output {
        (pred - target).mapv(T::abs).mean().unwrap()
    }
}
