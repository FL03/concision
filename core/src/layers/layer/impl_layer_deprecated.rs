/*
    appellation: impl_layer_deprecated <module>
    authors: @FL03
*/
#![allow(deprecated)]

use crate::layers::LayerBase;

use ndarray::{Dimension, RawData};

#[doc(hidden)]
impl<F, S, D, A> LayerBase<F, S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
}
