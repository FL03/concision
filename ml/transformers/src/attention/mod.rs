/*
   Appellation: attention <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Attention
//!
//! The attention mechanism is a key component of the transformer architecture.
//!
//!

pub use self::{head::*, reps::*, utils::*};

pub(crate) mod head;
pub(crate) mod reps;

pub mod multi;
pub mod ops;
pub mod params;

use ndarray::{Array, Array2};
use std::ops::MulAssign;

pub type BaseDim = ndarray::Dim<[usize; 4]>;

/// (batch, sample, seq, model)
pub type InputArray<T> = Array<T, BaseDim>;

pub type AttentionArray<T> = Array2<T>;

pub trait Attention {
    type Head: HeadSpace;

    fn head(&self) -> &Self::Head;
}

pub trait HeadSpace: MulAssign<Array2<f64>> {
    fn query(&self) -> &Array2<f64>;
    fn key(&self) -> &Array2<f64>;
    fn value(&self) -> &Array2<f64>;
}

pub trait Context<T: num::Float> {
    type Dim: ndarray::Dimension;

    fn query(&self) -> &Array<T, Self::Dim>;
    fn key(&self) -> &Array<T, Self::Dim>;
    fn value(&self) -> &Array<T, Self::Dim>;
}

pub(crate) mod utils {
    use super::ops::Split;
    use crate::neural::prelude::activate::{Activator, Softmax};
    use ndarray::ShapeError;
    use ndarray::prelude::{Array2, Array3};


    pub fn linear_layer<T: num::Float + 'static>(
        data: &Array2<T>,
        weights: &Array2<T>,
        heads: usize,
    ) -> Result<Array3<T>, ShapeError> {
        data.dot(weights).split(heads)
    }

    pub fn compute_attention(
        query: &Array2<f64>,
        key: &Array2<f64>,
        value: &Array2<f64>,
        mask: Option<Array2<f64>>,
    ) -> Array2<f64> {
        let (seq, dk) = query.dim();
        let mask = mask.unwrap_or_else(|| Array2::<f64>::zeros((seq, seq)));
        let scale = 1.0 / (dk as f64).sqrt();
        Softmax::rho((query.dot(&key.t()) + mask) * scale).dot(value)
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_attention() {}
}
