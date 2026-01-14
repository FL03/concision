/*
    Appellation: params <module>
    Contrib: @FL03
*/
#[cfg(feature = "alloc")]
use alloc::boxed::Box;
use ndarray::{
    ArrayBase, CowRepr, Dimension, Ix2, OwnedArcRepr, OwnedRepr, RawData, RawRef, RawViewRepr,
    ViewRepr,
};

/// A type alias for a [`ParamsBase`] with an owned internal layout
pub type Params<A = f32, D = Ix2> = ParamsBase<OwnedRepr<A>, D, A>;
/// A type alias for shared parameters
pub type ArcParams<A = f32, D = Ix2> = ParamsBase<OwnedArcRepr<A>, D, A>;
/// A type alias for an immutable view of the parameters
pub type ParamsView<'a, A = f32, D = Ix2> = ParamsBase<ViewRepr<&'a A>, D, A>;
/// A type alias for a mutable view of the parameters
pub type ParamsViewMut<'a, A = f32, D = Ix2> = ParamsBase<ViewRepr<&'a mut A>, D, A>;
/// A type alias for a [`ParamsBase`] with a _borrowed_ internal layout
pub type CowParams<'a, A = f32, D = Ix2> = ParamsBase<CowRepr<'a, A>, D, A>;
/// A type alias for the [`ParamsBase`] whose elements are of type `*const A` using a
/// [`RawViewRepr`] layout
pub type RawViewParams<A = f32, D = Ix2> = ParamsBase<RawViewRepr<*const A>, D, A>;
/// A type alias for the [`ParamsBase`] whose elements are of type `*mut A` using a
/// [`RawViewRepr`] layout
pub type RawMutParams<A = f32, D = Ix2> = ParamsBase<RawViewRepr<*mut A>, D, A>;

#[cfg(feature = "alloc")]
pub struct ParamsRef<A, D: Dimension> {
    pub bias: Box<RawRef<A, D::Smaller>>,
    pub weights: RawRef<A, D>,
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
