/*
    appellation: impl_layer_deprecated <module>
    authors: @FL03
*/
use crate::layers::LayerBase;

use ndarray::{Dimension, RawData};

#[doc(hidden)]
impl<F, S, D> LayerBase<F, S, D>
where
    D: Dimension,
    S: RawData,
{
}
