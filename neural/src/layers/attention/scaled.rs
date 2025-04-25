/*
    Appellation: scaled <module>
    Contrib: @FL03
*/

use ndarray::{Array2, Ix2, ScalarOperand, ShapeBuilder};
use num_traits::Float;

/// Scaled Dot-Product Attention mechanism is the core of the Transformer architecture.
/// It computes the attention scores using the dot product of the query and key vectors,
/// scales them by the square root of the dimension of the key vectors, and applies a softmax
/// function to obtain the attention weights.
///
/// This implementation follows the original paper
/// ["Attention is All You Need"](https://arxiv.org/pdf/1706.03762) by Vaswani et al.
pub struct ScaledDotProductAttention<T = f32> {
    pub(crate) dropout: T,
    /// The temperature parameter used to scale the attention scores.
    pub(crate) temperature: T,
    /// The attention mask used to prevent attending to certain positions in the input sequence.
    pub(crate) mask: Option<Array2<T>>,
    /// The attention scores computed from the query and key vectors.
    pub(crate) scores: Array2<T>,
}

impl<T> ScaledDotProductAttention<T> {
    /// Creates a new instance of the ScaledDotProductAttention struct.
    ///
    /// ### _Arguments_
    ///
    /// * `shape` - The shape of the mask and attention scores array.
    /// * `dropout` - The dropout rate to be applied to the attention scores.
    /// * `temperature` - The temperature parameter used to scale the attention scores.
    pub fn new<Sh>(shape: Sh, dropout: T, temperature: T) -> Self
    where
        Sh: ShapeBuilder<Dim = Ix2>,
        T: Default,
    {
        Self {
            dropout,
            temperature,
            mask: None,
            scores: Array2::default(shape), // Initialize with an empty array
        }
    }
    /// returns an immutable reference to the dropout probability
    pub const fn dropout(&self) -> &T {
        &self.dropout
    }
    /// returns a mutable reference to the dropout probability
    pub fn dropout_mut(&mut self) -> &mut T {
        &mut self.dropout
    }
    /// returns an immutable reference to the attention mask
    pub const fn mask(&self) -> Option<&Array2<T>> {
        self.mask.as_ref()
    }
    /// returns an immutable reference to the mechanisms mask
    pub fn mask_mut(&mut self) -> Option<&mut Array2<T>> {
        self.mask.as_mut()
    }
    /// returns an immutable reference to the attention scores
    pub const fn scores(&self) -> &Array2<T> {
        &self.scores
    }
    /// returns a mutable reference to the attention scores
    pub fn scores_mut(&mut self) -> &mut Array2<T> {
        &mut self.scores
    }
    /// returns an immutable reference to the temperature parameter
    pub const fn temperature(&self) -> &T {
        &self.temperature
    }
    /// returns a mutable reference to the temperature parameter
    pub fn temperature_mut(&mut self) -> &mut T {
        &mut self.temperature
    }
    /// set the attention mask to the given value
    pub fn set_mask(&mut self, mask: Array2<T>) -> &mut Self {
        self.mask = Some(mask);
        self
    }
    /// consumes the current instance to create another with the given mask
    pub fn with_mask(self, mask: Array2<T>) -> Self {
        Self {
            mask: Some(mask),
            ..self
        }
    }
    /// Computes the attention scores using the dot product of the query and key vectors,
    pub fn attention(&self, query: &Array2<T>, key: &Array2<T>, value: &Array2<T>) -> Array2<T>
    where
        T: Float + ScalarOperand,
    {
        // Compute the dot product of the query and key vectors
        let scores = query.dot(&key.t());

        // Scale the scores by the square root of the dimension of the key vectors
        let scaled_scores = scores / self.temperature;

        // Apply the softmax function to obtain the attention weights
        let attention_weights = scaled_scores.mapv(|x| x.exp()) / scaled_scores.sum();

        // Compute the final attention output by multiplying the attention weights with the value vectors
        let output = attention_weights.dot(value);

        output
    }
}
