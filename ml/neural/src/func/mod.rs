/*
   Appellation: func <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Functional
//!
//! This module implements several functional aspects of the neural network.
//!
//! ## Activation
//!
//! The activation functions are implemented as structs that implement the `Fn` trait.
//!
//! ## Loss
//!
//! The loss functions are implemented as structs that implement the `Fn` trait.
pub use self::{block::*, rms::*, utils::*};

pub mod activate;
pub mod loss;
pub mod prop;

pub(crate) mod block;
pub(crate) mod rms;

use ndarray::prelude::{Array, Dimension, Ix2};

pub trait Lin<T> {
    type Output;

    fn linear(&self, args: &T) -> Self::Output;
}

pub trait Objective<T = f64, D = Ix2>
where
    D: Dimension,
{
    fn objective(&self, args: &Array<T, D>) -> Array<T, D>;
}

pub(crate) mod utils {
    use ndarray::linalg::Dot;
    use ndarray::prelude::{Array, Dimension};
    use num::Float;
    use std::ops;

    pub fn lin<T, D, A, O>(
        args: &Array<T, A>,
        weights: &Array<T, D>,
        bias: &Array<T, D::Smaller>,
    ) -> Array<T, D>
    where
        A: Dimension,
        D: Dimension,
        O: Dimension,
        T: Float,
        Array<T, A>: Dot<Array<T, D>, Output = Array<T, O>>,
        Array<T, O>: ops::Add<Array<T, D::Smaller>, Output = Array<T, D>>,
    {
        args.dot(weights) + bias.clone()
    }

    pub fn slope_intercept<T>(args: T, slope: T, intercept: T) -> T
    where
        T: ops::Add<Output = T> + ops::Mul<Output = T>,
    {
        args * slope + intercept
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
}
