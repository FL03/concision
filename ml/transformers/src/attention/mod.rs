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

use crate::core::prelude::BoxResult;
use crate::prelude::BaseDim;

use ndarray::prelude::{Array, Array2, Ix2};
use ndarray::{Dimension, ScalarOperand};
use num::Float;
use std::ops::Mul;

/// (batch, sample, seq, model)
pub type InputArray<T> = Array<T, BaseDim>;

pub type AttentionArray<T> = Array<T, Ix2>;

pub trait Attention<T: Float> {
    fn attention(&self, data: &Array2<T>) -> BoxResult<Array2<T>>
    where
        T: ScalarOperand,
    {
        // let (seq, model) = data.dim();

        let q = self.query().dot(data);
        let k = self.key().dot(data);
        let v = self.value().dot(data);

        let score = attention(&q, &k, &v, Some(self.mask().clone()));
        Ok(score)
    }

    fn key(&self) -> &Array2<T>;

    fn mask(&self) -> &Array2<T>;

    fn query(&self) -> &Array2<T>;

    fn value(&self) -> &Array2<T>;
}

pub trait Head<T: Float> {
    fn key(&self) -> &Array2<T>;

    fn mask(&self) -> &Array2<T>;

    fn query(&self) -> &Array2<T>;

    fn value(&self) -> &Array2<T>;
}

pub trait Spaces<T: Float> {
    type Dim: Dimension;

    fn query(&self) -> &Array<T, Self::Dim>;
    fn key(&self) -> &Array<T, Self::Dim>;
    fn value(&self) -> &Array<T, Self::Dim>;
}

pub trait Weights<T: Float>: Mul<Array2<T>, Output = Self> {
    fn key(&self) -> &Array2<T>;

    fn query(&self) -> &Array2<T>;

    fn value(&self) -> &Array2<T>;

    fn qkv(&self) -> (&Array2<T>, &Array2<T>, &Array2<T>) {
        (self.query(), self.key(), self.value())
    }
}

pub(crate) mod utils {
    use crate::neural::prelude::{Activate, Softmax};
    use ndarray::prelude::Array2;
    use ndarray::ScalarOperand;
    use num::Float;

    pub fn attention<T: Float + ScalarOperand>(
        query: &Array2<T>,
        key: &Array2<T>,
        value: &Array2<T>,
        mask: Option<Array2<T>>,
    ) -> Array2<T> {
        let (seq, dk) = query.dim();
        let mask = mask.unwrap_or_else(|| Array2::<T>::zeros((seq, seq)));
        let scale = T::one() / (T::from(dk).unwrap()).sqrt();
        let softmax = Softmax::new(Some(1));
        softmax
            .activate((query.dot(&key.t()) + mask) * scale)
            .dot(value)
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_attention() {}
}
