/*
   Appellation: tensor <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::id::AtomicId;
use crate::prelude::DType;
use ndarray::prelude::{Array, Dimension, Ix2};
use ndarray::IntoDimension;
use num::Num;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub struct Tensor<T = f64, D = Ix2>
where
    D: Dimension,
{
    id: AtomicId,
    data: Array<T, D>,
    dtype: DType,
}

impl<T, D> Tensor<T, D>
where
    D: Dimension,
    T: Clone + Num,
{
    pub fn zeros(shape: impl IntoDimension<Dim = D>) -> Self {
        Self {
            id: AtomicId::new(),
            data: Array::zeros(shape),
            dtype: DType::default(),
        }
    }
}

impl<T, D> std::fmt::Display for Tensor<T, D>
where
    D: Dimension,
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}
