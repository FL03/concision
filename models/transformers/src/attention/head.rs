/*
    Appellation: head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::QkvBase;
use concision::getters;
use concision::nn::DropoutLayer;

use core::borrow::{Borrow, BorrowMut};
use nd::linalg::Dot;
use nd::*;
use num::complex::ComplexFloat;

// #68
pub struct AttentionHead<A = f64, D = Ix2, S = OwnedRepr<A>>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub(crate) dropout: Option<DropoutLayer>,
    pub(crate) mask: Option<Array<bool, D>>,
    pub(crate) params: QkvBase<S, D>,
}

impl<A, S, D> AttentionHead<A, D, S>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn from_params(params: QkvBase<S, D>) -> Self {
        Self {
            dropout: None,
            mask: None,
            params,
        }
    }

    pub fn builder<Sh, F>(shape: Sh, builder: F) -> Self
    where
        F: Fn(D) -> ArrayBase<S, D>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_params(QkvBase::builder(shape, builder))
    }

    pub fn from_elem<Sh>(shape: Sh, value: A) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        A: Clone,
        S: DataOwned,
    {
        Self::from_params(QkvBase::from_elem(shape, value))
    }
    #[cfg(not(feature = "rand"))]
    pub fn attention(&self) -> Array<A, D>
    where
        A: ComplexFloat + ScalarOperand,
        S: Data,
        ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
    {
        let (q, k, v) = self.qkv();
        super::_attention_no_dropout(q, k, v, self.mask())
    }
    #[cfg(feature = "rand")]
    pub fn attention(&self) -> Array<A, D>
    where
        A: ComplexFloat + ScalarOperand,
        S: Data,
        ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
        Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
    {
        let (q, k, v) = self.qkv();
        super::_attention(q, k, v, self.mask(), self.dropout())
    }

    pub fn dropout(&self) -> Option<&DropoutLayer> {
        self.dropout.as_ref()
    }
    /// Returns an immutable reference to the, optional, [Dropout] layer
    pub fn mask(&self) -> Option<&Array<bool, D>> {
        self.mask.as_ref()
    }
    /// Returns an immuable reference to the underlying parameters.
    pub const fn params(&self) -> &QkvBase<S, D> {
        &self.params
    }
    /// Returns a mutable reference to the underlying parameters.
    pub fn params_mut(&mut self) -> &mut QkvBase<S, D> {
        &mut self.params
    }
    /// Returns a three-tuple consisting of immputable references to the query, key, and value matrices respectively.
    pub fn qkv(&self) -> (&ArrayBase<S, D>, &ArrayBase<S, D>, &ArrayBase<S, D>) {
        self.params().qkv()
    }
    /// Consumes the head, returning a three-tuple consisting of mutable references to the query, key, and value matrices respectively.
    pub fn into_qkv(self) -> (ArrayBase<S, D>, ArrayBase<S, D>, ArrayBase<S, D>) {
        self.params.into_qkv()
    }

    getters!(params::<[q, k, v]> => ArrayBase<S, D>);
    ndbuilder!(new::default() where A: Default, S: DataOwned);
    ndbuilder!(ones() where A: Clone + num::One, S: DataOwned);
    ndbuilder!(zeros() where A: Clone + num::Zero, S: DataOwned);
}

impl<A, S, D> super::Attention for AttentionHead<A, D, S>
where
    A: ComplexFloat + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
    ArrayBase<S, D>: for<'a> Dot<ArrayView<'a, A, D>, Output = Array<A, D>>,
    Array<A, D>: Dot<ArrayBase<S, D>, Output = Array<A, D>>,
{
    type Output = Array<A, D>;

    fn attention(&self) -> Self::Output {
        self.attention()
    }
}

impl<A, S, D> Borrow<QkvBase<S, D>> for AttentionHead<A, D, S>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn borrow(&self) -> &QkvBase<S, D> {
        self.params()
    }
}

impl<A, S, D> BorrowMut<QkvBase<S, D>> for AttentionHead<A, D, S>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn borrow_mut(&mut self) -> &mut QkvBase<S, D> {
        self.params_mut()
    }
}
