/*
    Appellation: ffn <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Feed Forward Neural Network
//!
pub use self::{mlp::*, model::*, utils::*};

pub(crate) mod mlp;
pub(crate) mod model;

use ndarray::prelude::{Array, Array2, Dimension, Ix2};
use num::Float;

pub trait Optimizer<T = f64>
where
    T: Float,
{
    fn step(&mut self, grad: &Array2<T>) -> Array2<T>;
}

pub trait FeedForward<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    type Opt;

    fn backward(&mut self, args: &Array2<T>, targets: &Array<T, D>, opt: &Self::Opt) -> Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Array<T, D>;
}

pub(crate) mod utils {}

#[cfg(tets)]
mod tests {}
