/*
   Appellation: attention <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::attention::params::AttentionParameters;
use crate::attention::Weight;

pub struct MultiHeadAttention {
   heads: usize,
   model: usize,
   weights: Weight,
}

impl MultiHeadAttention {
   pub fn new(heads: usize, model: usize) -> Self {
      let weights = Weight::new((model, model));
      Self { heads, model, weights }
   }
}
