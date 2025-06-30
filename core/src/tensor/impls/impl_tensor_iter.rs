/*
    appellation: impl_tensor_iter <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;

use ndarray::{Dimension, RawData};

impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
}
