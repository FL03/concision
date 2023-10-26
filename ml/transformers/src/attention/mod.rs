/*
   Appellation: attention <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Attention
//!
//! The attention mechanism is a key component of the transformer architecture.
//!
//!

pub use self::{context::*, head::*, utils::*};

pub(crate) mod context;
pub(crate) mod head;

pub mod multi;
pub mod params;

use ndarray::{Array, Array2};
use std::ops;

pub type BaseDim = ndarray::Dim<[usize; 4]>;

/// (batch, sample, seq, model)
pub type InputArray<T> = Array<T, BaseDim>;

pub type AttentionArray<T> = Array2<T>;

pub trait Representations {
    fn query(&self) -> &InputArray<f64>;
    fn key(&self) -> &InputArray<f64>;
    fn value(&self) -> &InputArray<f64>;
}

pub trait Attention {
    type Head: Head;

    fn head(&self) -> &Self::Head;
 
}

pub trait Head: ops::MulAssign<AttentionArray<f64>> {
    fn query(&self) -> &AttentionArray<f64>;
    fn key(&self) -> &AttentionArray<f64>;
    fn value(&self) -> &AttentionArray<f64>;


}



pub(crate) mod utils {
    use crate::neural::prelude::activate::{Activator, Softmax};
    use ndarray::Array2;

    pub fn linear_layer<T: num::Float + 'static>(
        data: &Array2<T>,
        weights: &Array2<T>,
    ) -> Array2<T> {
        data.dot(weights)
    }

    pub fn compute_attention(
        query: &Array2<f64>,
        key: &Array2<f64>,
        value: &Array2<f64>,
    ) -> Array2<f64> {
        let dk = query.shape()[1] as f64;
        Softmax::rho(query.dot(&key.t()) / dk.sqrt()) * value
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_attention() {}
}
