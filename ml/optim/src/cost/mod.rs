/*
    Appellation: cost <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # cost
//!
pub use self::{kinds::*, utils::*};

pub(crate) mod kinds;

use ndarray::prelude::Array;
use ndarray::Dimension;
use num::Float;

pub trait Cost<T = f64>
where
    T: Float,
{
    fn cost(&self, pred: &T, target: &T) -> T;
}

pub trait CostArr<T = f64>
where
    T: Float,
{
    type Dim: Dimension;

    fn cost(&self, pred: &Array<T, Self::Dim>, target: &Array<T, Self::Dim>)
        -> Array<T, Self::Dim>;
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
