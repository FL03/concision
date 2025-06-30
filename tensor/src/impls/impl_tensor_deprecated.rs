/*
    appellation: impl_tensor_deprecated <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;

use ndarray::{Dimension, RawData};

#[doc(hidden)]
impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
}
