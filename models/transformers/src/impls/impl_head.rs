/*
    Appellation: impl_head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::attention::AttentionHead;
use crate::params::QkvBase;
use nd::prelude::*;
use nd::{DataOwned, RawDataClone};

impl<A, S, D> Clone for AttentionHead<A, D, S>
where
    A: Copy,
    D: Dimension,
    S: RawDataClone<Elem = A>,
{
    fn clone(&self) -> Self {
        Self {
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
