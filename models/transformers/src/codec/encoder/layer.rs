/*
    Appellation: layer <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::attention::multi::MultiHeadAttention;

#[derive(Default)]
pub struct EncoderLayer {
    pub(crate) attention: MultiHeadAttention,
}

impl EncoderLayer {
    pub fn new() -> Self {
        let attention = MultiHeadAttention::default();

        Self { attention }
    }
    /// Returns an immutable reference to the multi-head, self-attention layer.
    pub fn attention(&self) -> &MultiHeadAttention {
        &self.attention
    }
    /// Returns a mutable reference to the multi-head, self-attention layer.
    pub fn attention_mut(&mut self) -> &mut MultiHeadAttention {
        &mut self.attention
    }
}
