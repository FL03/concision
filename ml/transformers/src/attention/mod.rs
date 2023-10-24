/*
   Appellation: attention <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Attention
//! 
//! 
pub use self::{head::*, utils::*, weights::*,};

pub(crate) mod head;
pub(crate) mod weights;

pub type AttentionArray<T> = ndarray::Array2<T>;

pub trait Attention {}

pub(crate) mod utils {
    use crate::neural::prelude::activate::{Activator, Softmax};
    use ndarray::{s, Array3, Array2};

    pub fn compute_attention(qkv: &Array3<f64>) -> Array2<f64> {
        let query = qkv.slice(s![0, .., ..]).to_owned();
        let key = qkv.slice(s![1, .., ..]).to_owned();
        let value = qkv.slice(s![2, .., ..]).to_owned();
        let dk = qkv.shape()[1] as f64;

        let inner = (query * key.t()) / dk.sqrt();
        Softmax::rho(inner) * value
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_attention() {}
}
