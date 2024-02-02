/*
   Appellation: tensors <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Tensors
//!
//! A tensor is a generalization of vectors and matrices to potentially higher dimensions.
pub use self::{mode::*, tensor::*};

pub(crate) mod mode;
pub(crate) mod tensor;

// use ndarray::prelude::{Array, Dimension, Ix2};
use crate::cmp::DType;
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

    fn apply_op<Op>(&self, op: Op) -> <Op as Operation<Self>>::Output
    where
        Op: Operation<Self>,
        Self: Sized,
    {
        op.eval(self)
    }

    fn backward(&self) -> Self;

    fn dtype(&self) -> DType;

    fn id(&self) -> &str;

    fn is_variable(&self) -> bool {
        self.mode().is_variable()
    }

    fn mode(&self) -> TensorKind;

    fn shape(&self) -> &[usize];
}

pub trait Genspace {
    type Tensor: NdTensor;

    fn arange(start: f64, stop: f64) -> Self::Tensor;

    fn range(start: f64, stop: f64, step: f64) -> Self::Tensor;
}

pub trait TensorOps<T = f64> {
    type Tensor: NdTensor<T>;

    fn add(&self, other: &Self::Tensor) -> Self::Tensor;

    fn add_scalar(&self, other: T) -> Self::Tensor;

    fn div(&self, other: &Self::Tensor) -> Self::Tensor;

    fn div_scalar(&self, other: T) -> Self::Tensor;

    fn mul(&self, other: &Self::Tensor) -> Self::Tensor;

    fn mul_scalar(&self, other: T) -> Self::Tensor;

    fn sub(&self, other: &Self::Tensor) -> Self::Tensor;

    fn sub_scalar(&self, other: T) -> Self::Tensor;
}

#[cfg(test)]
mod tests {}
