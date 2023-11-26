/*
    Appellation: ffn <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Feed Forward Neural Network
//!
pub use self::{model::*, utils::*};

pub(crate) mod model;

use ndarray::prelude::{Array, Array2, Dimension, Ix2};
use num::Float;



pub trait FeedForward<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    type Opt;

    fn backward(&mut self, args: &Array2<T>, targets: &Array2<T>, opt: &Self::Opt) -> Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Array<T, D>;

    
}

pub(crate) mod utils {}

#[cfg(tets)]
mod tests {}
