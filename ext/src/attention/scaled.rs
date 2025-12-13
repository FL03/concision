/*
    Appellation: scaled <module>
    Contrib: @FL03
*/
use super::QkvParamsBase;
use ndarray::linalg::Dot;
use ndarray::{
    Array, ArrayBase, ArrayView, Data, DataOwned, Dimension, Ix2, OwnedRepr, RawData, ScalarOperand,
};
use num_traits::Float;

/// Scaled Dot-Product Attention mechanism is the core of the Transformer architecture.
/// It computes the attention scores using the dot product of the query and key vectors,
/// scales them by the square root of the dimension of the key vectors, and applies a softmax
/// function to obtain the attention weights.
///
/// This implementation follows the original paper
/// ["Attention is All You Need"](https://arxiv.org/pdf/1706.03762) by Vaswani et al.
pub struct SDPA<A, D = Ix2, S = OwnedRepr<A>>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub(crate) dropout: A,
    /// The temperature parameter used to scale the attention scores.
    pub(crate) temperature: A,
    /// The attention mask used to prevent attending to certain positions in the input sequence.
    pub(crate) mask: Option<ArrayBase<S, D>>,
}

impl<A, S, D> SDPA<A, D, S>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// Creates a new instance of the ScaledDotProductAttention struct.
    ///
    /// ### _Arguments_
    ///
    /// * `shape` - The shape of the mask and attention scores array.
    /// * `dropout` - The dropout rate to be applied to the attention scores.
    /// * `temperature` - The temperature parameter used to scale the attention scores.
    pub fn new(dropout: A, temperature: A) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
    {
        Self {
            dropout,
            temperature,
            mask: None,
        }
    }
    /// returns an immutable reference to the dropout probability
    pub const fn dropout(&self) -> &A {
        &self.dropout
    }
    /// returns a mutable reference to the dropout probability
    pub fn dropout_mut(&mut self) -> &mut A {
        &mut self.dropout
    }
    /// returns an immutable reference to the attention mask
    pub const fn mask(&self) -> Option<&ArrayBase<S, D>> {
        self.mask.as_ref()
    }
    /// returns an immutable reference to the mechanisms mask
    pub fn mask_mut(&mut self) -> Option<&mut ArrayBase<S, D>> {
        self.mask.as_mut()
    }
    /// returns an immutable reference to the temperature parameter
    pub const fn temperature(&self) -> &A {
        &self.temperature
    }
    /// returns a mutable reference to the temperature parameter
    pub fn temperature_mut(&mut self) -> &mut A {
        &mut self.temperature
    }

    pub fn set_dropout(&mut self, dropout: A) -> &mut Self {
        *self.dropout_mut() = dropout;
        self
    }

    pub fn set_temperature(&mut self, temperature: A) -> &mut Self {
        *self.temperature_mut() = temperature;
        self
    }
    /// set the attention mask to the given value
    pub fn set_mask(&mut self, mask: ArrayBase<S, D>) -> &mut Self {
        self.mask = Some(mask);
        self
    }
    /// consumes the current instance to create another with the given mask
    pub fn with_mask(self, mask: ArrayBase<S, D>) -> Self {
        Self {
            mask: Some(mask),
            ..self
        }
    }
    /// Computes the attention scores using the dot product of the query and key vectors,
    pub fn attention(
        &self,
        QkvParamsBase { query, key, value }: &QkvParamsBase<S, D>,
    ) -> Array<A, D>
    where
        A: Float + ScalarOperand,
        S: Data,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
        for<'a> ArrayBase<S, D>: Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
    {
        // Compute the dot product of the query and key vectors
        let scores = query.dot(&key.t());

        // Scale the scores by the square root of the dimension of the key vectors
        let scaled_scores = scores / self.temperature;

        // Apply the softmax function to obtain the attention weights
        let attention_weights = scaled_scores.mapv(|x| x.exp()) / scaled_scores.sum();

        // Compute the final attention output by multiplying the attention weights with the value vectors

        attention_weights.dot(value)
    }
}
