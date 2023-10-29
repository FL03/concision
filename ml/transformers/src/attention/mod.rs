/*
   Appellation: attention <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Attention
//!
//! The attention mechanism is a key component of the transformer architecture.
//!
//!

pub use self::{head::*, utils::*, weights::*};

pub(crate) mod head;
pub(crate) mod weights;

pub mod multi;
pub mod params;

use crate::prelude::BaseDim;
use crate::core::prelude::BoxResult;

use ndarray::Dimension;
use ndarray::prelude::{Array, Ix2};
use num::Float;

/// (batch, sample, seq, model)
pub type InputArray<T> = Array<T, BaseDim>;

pub type AttentionArray<T> = Array<T, Ix2>;

pub trait Attention<T: Float> {
    type Dim: Dimension;
    type Score;

    fn attention(&mut self, data: &Array<T, Ix2>) -> BoxResult<&Array<T, Ix2>>;
}

pub trait Head<T: Float> {

    fn query(&self) -> &Array<T, Ix2>;
    fn key(&self) -> &Array<T, Ix2>;
    fn value(&self) -> &Array<T, Ix2>;
}

pub trait Spaces<T: Float> {
    type Dim: Dimension;

    fn query(&self) -> &Array<T, Self::Dim>;
    fn key(&self) -> &Array<T, Self::Dim>;
    fn value(&self) -> &Array<T, Self::Dim>;
}

pub(crate) mod utils {
    use crate::ops::Split;
    use crate::neural::prelude::activate::{Activator, Softmax};
    use ndarray::prelude::{Array2, Array3};
    use ndarray::{ScalarOperand, ShapeError};
    use num::Float;

    pub fn linear_layer<T: num::Float + 'static>(
        data: &Array2<T>,
        weights: &Array2<T>,
        heads: usize,
    ) -> Result<Array3<T>, ShapeError> {
        data.dot(weights).split(heads)
    }

    pub fn compute_attention<T: Float + ScalarOperand>(
        query: &Array2<T>,
        key: &Array2<T>,
        value: &Array2<T>,
        mask: Option<Array2<T>>,
    ) -> Array2<T> {
        let (seq, dk) = query.dim();
        let mask = mask.unwrap_or_else(|| Array2::<T>::zeros((seq, seq)));
        let scale = T::one() / (T::from(dk).unwrap()).sqrt();
        Softmax::rho((query.dot(&key.t()) + mask) * scale).dot(value)
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_attention() {}
}
