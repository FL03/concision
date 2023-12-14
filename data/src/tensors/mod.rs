/*
   Appellation: tensors <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Tensors
pub use self::{tensor::*, utils::*};

pub(crate) mod tensor;

use ndarray::prelude::{Array, Dimension, Ix2};

pub trait NdTensor<T, D = Ix2>
where
    D: Dimension,
{
    fn data(&self) -> &Array<T, D>;
}

pub(crate) mod utils {}
