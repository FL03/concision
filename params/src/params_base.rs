/*
    Appellation: params <module>
    Contrib: @FL03
*/
#[cfg(feature = "alloc")]
use alloc::boxed::Box;
use ndarray::{ArrayBase, ArrayRef, Dimension, RawData};

#[cfg(feature = "alloc")]
pub struct ParamsRef<A, D: Dimension> {
    pub bias: Box<ArrayRef<A, D::Smaller>>,
    pub weights: ArrayRef<A, D>,
}

/// The [`ParamsBase`] implementation aims to provide a generic, n-dimensional weight and bias
/// pair for a model (or layer). The object requires the bias tensor to be a single dimension
/// smaller than the weights tensor.
///
/// Therefore, we allow the weight tensor to be the _shape_ of the parameters, using the shape
/// as the basis for the bias tensor by removing the first axis.
/// Consequently, this constrains the [`ParamsBase`] implementation to only support dimensions
/// that can be reduced by one axis, typically the "zero-th" axis: $\text{rank}(D)$.
pub struct ParamsBase<S, D = ndarray::Ix2, A = <S as RawData>::Elem>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub bias: ArrayBase<S, D::Smaller, A>,
    pub weights: ArrayBase<S, D, A>,
}
