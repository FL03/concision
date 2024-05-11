/*
    Appellation: impl_head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::attention::AttentionHead;
use crate::params::QKVBase;
use nd::DataOwned;
use nd::prelude::*;

impl<A, S, D> Default for AttentionHead<A, S, D>
where
    A: Default,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    fn default() -> Self {
        Self::from_params(QKVBase::default())
    }
}