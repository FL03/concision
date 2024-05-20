/*
    Appellation: attention <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Attention
//!
//! Attention allows a model to focus on specific parts of the input sequence.
//! Today, these mechanisms are found in several state-of-the-art models, such as
//! the Transformer model, primarily due to its capabilities in natural language
//! processing (NLP) domains
pub use self::head::AttentionHead;
pub use self::utils::*;

pub(crate) mod head;

// #69: Multi-Head Attention implementation
pub mod multi;

pub(crate) mod prelude {
    pub use super::head::AttentionHead;
    pub use super::utils::*;
}

pub trait Attention {
    type Output;

    fn attention(&self) -> Self::Output;
}

pub(crate) mod utils {
    use concision::func::Softmax;
    use concision::nn::DropoutLayer;
    use concision::MaskFill;
    use nd::linalg::Dot;
    use nd::prelude::{Array, ArrayBase, ArrayView, Axis, Dimension};
    use nd::{Data, ScalarOperand};
    use num::complex::ComplexFloat;

    pub(crate) fn scale<A>(dk: usize) -> A
    where
        A: ComplexFloat,
    {
        A::from(dk).unwrap().sqrt().recip()
    }

    /// A functional implementation of the scaled dot-product attention mechanism;
    pub fn scaled_dot_product_attention<A, S, D>(
        q: &ArrayBase<S, D>,
        k: &ArrayBase<S, D>,
        v: &ArrayBase<S, D>,
        mask: Option<&Array<bool, D>>,
    ) -> Array<A, D>
    where
        A: ComplexFloat + ScalarOperand,
        S: Data<Elem = A>,
        D: Dimension,
        ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
    {
        _attention_no_dropout(q, k, v, mask)
    }

    pub(crate) fn _attention_no_dropout<A, S, D>(
        q: &ArrayBase<S, D>,
        k: &ArrayBase<S, D>,
        v: &ArrayBase<S, D>,
        mask: Option<&Array<bool, D>>,
    ) -> Array<A, D>
    where
        A: ComplexFloat + ScalarOperand,
        S: Data<Elem = A>,
        D: Dimension,
        ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
    {
        let dk = scale::<A>(k.len_of(Axis(1)));
        let mut z = q.dot(&k.t()) * dk;
        if let Some(mask) = mask {
            z = z.masked_fill(mask, A::zero());
        }
        z.softmax().dot(&v)
    }
    #[cfg(feature = "rand")]
    pub(crate) fn _attention<A, S, D>(
        q: &ArrayBase<S, D>,
        k: &ArrayBase<S, D>,
        v: &ArrayBase<S, D>,
        mask: Option<&Array<bool, D>>,
        dropout: Option<&DropoutLayer>,
    ) -> Array<A, D>
    where
        A: ComplexFloat + ScalarOperand,
        S: Data<Elem = A>,
        D: Dimension,
        ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
    {
        use concision::Forward;
        let dk = scale::<A>(k.len_of(Axis(1)));
        let mut z = q.dot(&k.t()) * dk;
        if let Some(mask) = mask {
            z = z.masked_fill(mask, A::zero());
        }
        z = z.softmax();
        if let Some(dropout) = dropout {
            z = dropout.forward(&z);
        }
        z.dot(&v)
    }
}
