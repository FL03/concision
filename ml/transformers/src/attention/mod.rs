/*
   Appellation: attention <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Attention
pub use self::{head::*, utils::*};

pub(crate) mod head;

pub type AttentionArray<T> = ndarray::Array2<T>;

pub trait Attention {}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_attention() {}
}
