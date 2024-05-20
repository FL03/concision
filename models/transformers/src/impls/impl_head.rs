/*
    Appellation: impl_head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::attention::{Attention, AttentionHead};
use crate::params::QkvBase;
use core::borrow::{Borrow, BorrowMut};
use nd::linalg::Dot;
use nd::prelude::*;
use nd::{Data, DataOwned, RawData, RawDataClone, ScalarOperand};
use num::complex::ComplexFloat;

impl<A, S, D> Attention for AttentionHead<A, D, S>
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

impl<A, S, D> Clone for AttentionHead<A, D, S>
where
    A: Copy,
    D: Dimension,
    S: RawDataClone<Elem = A>,
{
    fn clone(&self) -> Self {
        Self {
            #[cfg(feature = "rand")]
            dropout: self.dropout.clone(),
            mask: self.mask.clone(),
            params: self.params.clone(),
        }
    }
}

impl<A, S, D> Copy for AttentionHead<A, D, S>
where
    A: Copy,
    D: Copy + Dimension,
    S: Copy + RawDataClone<Elem = A>,
    Array<bool, D>: Copy,
{
}

impl<A, S, D> Default for AttentionHead<A, D, S>
where
    A: Default,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    fn default() -> Self {
        Self::from_params(QkvBase::default())
    }
}
