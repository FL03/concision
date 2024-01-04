/*
   Appellation: tensors <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Tensors
//!
//! A tensor is a generalization of vectors and matrices to potentially higher dimensions.
pub use self::tensor::*;

pub(crate) mod tensor;

// use ndarray::prelude::{Array, Dimension, Ix2};
use crate::core::ops::Operation;

pub trait GradStore<T = f64> {
    type Tensor: NdTensor<T>;

    fn get(&self, id: &str) -> Option<&Self::Tensor>;
}

pub trait NdTensor<T = f64> {
    fn affine(&self, a: T, b: T) -> Self;

    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(T) -> T;

    fn apply_op(&self, op: impl Operation<T>) -> Self;

    fn backward(&self) -> Self;

    fn id(&self) -> &str;

    fn tensor(&self) -> &Self;
}

#[cfg(test)]
mod tests {}
