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
use crate::{core::ops::Operation, DType};
use num::traits::{Num, NumOps};

pub trait GradStore<T = f64> {
    type Tensor: NdTensor<T>;

    fn get(&self, id: &str) -> Option<&Self::Tensor>;
}

pub trait ComplexN: Num + NumOps<Self::Real> {
    type Real: NumOps<Self, Self>;

    fn im(&self) -> Self::Real;

    fn re(&self) -> Self::Real;
}

pub trait TensorScalar {
    type Complex: ComplexN<Real = Self::Real>;
    type Real: Num + NumOps + NumOps<Self::Complex, Self::Complex>;
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

    fn tensor(&self) -> &Self;
}

pub trait Genspace {
    type Tensor: NdTensor;

    fn arange(start: f64, stop: f64, step: f64) -> Self::Tensor;
}

#[cfg(test)]
mod tests {}
