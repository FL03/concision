/*
    Appellation: attention <module>
    Contrib: @FL03
*/
//! ## Attention
//!
//! This module focuses on implementing various attention mechanisms that are commonly used in
//! the transformer architecture.
//!
//! ### _Features_
//!
//! - **Scaled Dot-Product Attention**: This is the basic attention mechanism that computes the
//!   attention scores using the dot product of the query and key vectors, scales them by the
//!   square root of the dimension of the key vectors, and applies a softmax function to obtain
//!   the attention scores.
//! - **Multi-Head Attention**: This mechanism extends the scaled dot-product attention by
//!   allowing the model to jointly attend to information from different representation
//!   subspaces at different positions. It does this by projecting the queries, keys and values
//!   multiple times with different learned linear projections, and then concatenating the
//!   results.
//! - **FFT Attention**: This is a more advanced attention mechanism that uses the Fast Fourier
//!   Transform (FFT) to compute the attention scores more efficiently. It is particularly
//!   useful for long sequences where the standard attention mechanism can be computationally
//!   expensive.
//!
#[doc(inline)]
pub use self::{multi_head::MultiHeadAttention, qkv::*, scaled::SDPA};

#[cfg(feature = "signal")]
pub mod fft;
pub mod multi_head;
pub mod qkv;
pub mod scaled;

pub trait Attention<T> {
    /// Computes the attention scores using the dot product of the query and key vectors,
    /// scales them by the square root of the dimension of the key vectors, and applies a softmax
    /// function to obtain the attention weights.
    fn attention(&self, query: T, key: T, value: T) -> T;
}

pub(crate) mod prelude {
    #[cfg(feature = "signal")]
    pub use super::fft::FftAttention;
    #[doc(inline)]
    pub use super::multi_head::MultiHeadAttention;
    #[doc(inline)]
    pub use super::qkv::QkvParamsBase;
    #[doc(inline)]
    pub use super::scaled::SDPA;
}

#[cfg(test)]
mod tests {
    use super::{Qkv, SDPA};

    #[test]
    fn test_scaled_dot_product_attention() {
        // define the shape of the params
        let [m, n] = [7, 10];
        // initialize some params
        let qkv = Qkv::<f64>::ones((m, n));
        // initialize the scaled dot-product attention layer
        let layer = SDPA::<f64>::new(0.1, 1.0);
        // compute the attention scores
        let z_score = layer.attention(&qkv);
        // verify the output dimensions
        assert_eq!(z_score.shape(), &[m, n]);
    }
}
