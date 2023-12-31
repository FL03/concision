/*
    Appellation: primitives <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::constants::*;
use crate::core::specs::{Arithmetic, AsComplex, Conjugate};
use ndarray::ScalarOperand;
use ndarray_linalg::Lapack;
use num::complex::ComplexFloat;
use num::traits::FromPrimitive;
// use num::traits::{Float, FromPrimitive, FloatConst};
use std::ops;

pub trait S4Float:
    Arithmetic<Self> + AsComplex + Conjugate + ComplexFloat + FromPrimitive + Lapack + ScalarOperand
{
}

impl<T> S4Float for T where
    T: AsComplex + Conjugate + ComplexFloat + FromPrimitive + Lapack + ScalarOperand
{
}

mod constants {
    /// The default model size for S4 models
    pub const S4_MODEL_SIZE: usize = 2048;
}

mod statics {}

mod types {}
