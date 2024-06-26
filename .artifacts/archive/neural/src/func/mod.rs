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
pub use self::{rms::*, utils::*};

pub mod activate;
pub mod loss;

pub(crate) mod rms;

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
