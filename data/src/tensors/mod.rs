/*
   Appellation: tensors <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Tensors
pub use self::tensor::*;

pub(crate) mod tensor;

use ndarray::prelude::{Array, Dimension, Ix2};

pub trait NdTensor<T = f64> {
    type Dim: Dimension = Ix2;

    fn tensor(&self) -> &Array<T, Self::Dim>;
}

#[cfg(test)]
mod tests {}
