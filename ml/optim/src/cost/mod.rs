/*
    Appellation: cost <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # cost
//!
pub use self::{kinds::*, utils::*};

pub(crate) mod kinds;

use ndarray::prelude::{Array, Array1};
use ndarray::Dimension;
use num::Float;

pub trait Cost {
    type Output;

    fn cost(&self, other: &Self) -> Self::Output;
}

pub trait CostArr<T = f64>
where
    T: Float,
{
    type Dim: Dimension;

    fn cost(&self, pred: &Array<T, Self::Dim>, target: &Array1<T>) -> Array<T, Self::Dim>;
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
