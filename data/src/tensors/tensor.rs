/*
   Appellation: tensor <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::id::AtomicId;
use ndarray::prelude::{Array, Dimension, Ix2};
use ndarray::IntoDimension;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct Tensor<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    id: AtomicId,
    data: Array<T, D>,
}

impl<T, D> Tensor<T, D>
where
    D: Dimension,
    T: Float,
{
    pub fn new(shape: impl IntoDimension<Dim = D>) -> Self {
        Self {
            id: AtomicId::new(),
            data: Array::zeros(shape),
        }
    }
}

impl<T, D> std::fmt::Display for Tensor<T, D>
where
    D: Dimension,
    T: Float + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}
