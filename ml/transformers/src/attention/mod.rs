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

use params::QKV;

use crate::core::prelude::BoxResult;
use crate::prelude::BaseDim;

use ndarray::prelude::{Array, Array2, Ix2, NdFloat};
use num::Float;
use std::ops;

/// (batch, sample, seq, model)
pub type InputArray<T> = Array<T, BaseDim>;

pub type AttentionArray<T> = Array<T, Ix2>;

pub trait Attention<T: NdFloat = f64> {
    fn attention(&self, data: &Array2<T>) -> BoxResult<Array2<T>> {
        // let (seq, model) = data.dim();

        let q = self.query().dot(data);
        let k = self.key().dot(data);
        let v = self.value().dot(data);

        let score = scaled_dot_product_attention(&q, &k, &v, Some(self.mask().clone()));
        Ok(score)
    }

    fn key(&self) -> &Array2<T>;

    fn mask(&self) -> &Array2<T>;

    fn query(&self) -> &Array2<T>;

    fn value(&self) -> &Array2<T>;
}

pub trait Head<T>
where
    T: Float,
{
    fn key(&self) -> &Array2<T>;

    fn query(&self) -> &Array2<T>;

    fn value(&self) -> &Array2<T>;

    fn query_size(&self) -> usize {
        self.query().dim().1
    }

    fn qkv(&self) -> (&Array2<T>, &Array2<T>, &Array2<T>) {
        (self.query(), self.key(), self.value())
    }

    fn scale(&self) -> T {
        T::one() / (T::from(self.key().dim().1).unwrap()).sqrt()
    }
}

impl<S, T> Head<T> for S
where
    S: ops::Index<QKV, Output = Array2<T>>,
    T: Float,
{
    fn key(&self) -> &Array2<T> {
        &self[QKV::Key]
    }

    fn query(&self) -> &Array2<T> {
        &self[QKV::Query]
    }

    fn value(&self) -> &Array2<T> {
        &self[QKV::Value]
    }
}

pub(crate) mod utils {
    use crate::neural::prelude::{Activate, Softmax};
    use ndarray::prelude::{Array2, NdFloat};

    pub fn scaled_dot_product_attention<T: NdFloat>(
        query: &Array2<T>,
        key: &Array2<T>,
        value: &Array2<T>,
        mask: Option<Array2<T>>,
    ) -> Array2<T> {
        let (seq, dk) = query.dim();
        let mask = mask.unwrap_or_else(|| Array2::<T>::zeros((seq, seq)));
        let scale = T::one() / (T::from(dk).unwrap()).sqrt();
        let score = (query.dot(&key.t()) + mask) * scale;
        Softmax::new(Some(1)).activate(&score).dot(value)
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_attention() {}
}
