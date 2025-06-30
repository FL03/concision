/*
    appellation: aliases <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;
use ndarray::{Ix2, OwnedArcRepr, OwnedRepr, RawViewRepr, ViewRepr};

/// a type alias for a [`TensorBase`] with an owned representation
pub type Tensor<A, D = Ix2> = TensorBase<OwnedRepr<A>, D>;
/// a type alias for a [`TensorBase`] with a view representation
pub type TensorView<'a, A, D = Ix2> = TensorBase<ViewRepr<&'a A>, D>;
/// a type alias for a [`TensorBase`] with a mutable view representation
pub type TensorViewMut<'a, A, D = Ix2> = TensorBase<ViewRepr<&'a mut A>, D>;

/// a type alias for a [`TensorBase`] with an owned representation
pub type ArcTensor<A, D = Ix2> = TensorBase<OwnedArcRepr<A>, D>;
/// A type alias for a [`TensorBase`] with a raw pointer representation.
pub type RawViewTensor<A, D = Ix2> = TensorBase<RawViewRepr<*const A>, D>;
/// a type alias for a [`TensorBase`] with an owned representation
pub type RawViewMutTensor<A, D = Ix2> = TensorBase<RawViewRepr<*mut A>, D>;
