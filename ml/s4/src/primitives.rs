/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::constants::*;
use crate::core::specs::{Arithmetic, AsComplex, Conjugate};
use ndarray::ScalarOperand;
use ndarray_linalg::{Lapack, Scalar};
use num::complex::{Complex, ComplexFloat};
use num::traits::FromPrimitive;
// use num::traits::{Float, FromPrimitive, FloatConst};
use std::ops;

pub trait S4Float: Arithmetic<Self>
    + AsComplex
    + Conjugate
    + ComplexFloat
    + FromPrimitive
    + Lapack
    + Scalar
    + ScalarOperand
    + ops::Mul<Complex<<Self as ComplexFloat>::Real>, Output = Complex<<Self as ComplexFloat>::Real>>
where
    <Self as ComplexFloat>::Real: FromPrimitive,
{
}

impl<T> S4Float for T
where
    T: Arithmetic<Self>
        + Arithmetic<
            Complex<<Self as ComplexFloat>::Real>,
            Output = Complex<<Self as ComplexFloat>::Real>,
        > + AsComplex
        + Conjugate
        + ComplexFloat
        + FromPrimitive
        + Lapack
        + ScalarOperand
        + ops::Mul<Complex<<T as ComplexFloat>::Real>, Output = Complex<<T as ComplexFloat>::Real>>,
    <Self as ComplexFloat>::Real: FromPrimitive,
{
}

mod constants {
    /// The default model size for S4 models
    pub const S4_MODEL_SIZE: usize = 2048;
}

mod statics {}

mod types {}
