/*
    appellation: aliases <module>
    authors: @FL03
*/
use crate::params::ParamsBase;

use ndarray::{CowRepr, Ix2, OwnedArcRepr, OwnedRepr, RawViewRepr, ViewRepr};

/// A type alias for a [`ParamsBase`] with an owned internal layout
pub type Params<A, D = Ix2> = ParamsBase<OwnedRepr<A>, D, A>;
/// A type alias for shared parameters
pub type ArcParams<A, D = Ix2> = ParamsBase<OwnedArcRepr<A>, D, A>;
/// A type alias for an immutable view of the parameters
pub type ParamsView<'a, A, D = Ix2> = ParamsBase<ViewRepr<&'a A>, D, A>;
/// A type alias for a mutable view of the parameters
pub type ParamsViewMut<'a, A, D = Ix2> = ParamsBase<ViewRepr<&'a mut A>, D, A>;
/// A type alias for a [`ParamsBase`] with a _borrowed_ internal layout
pub type CowParams<'a, A, D = Ix2> = ParamsBase<CowRepr<'a, A>, D, A>;
/// A type alias for the [`ParamsBase`] whose elements are of type `*const A` using a
/// [`RawViewRepr`] layout
pub type RawViewParams<A, D = Ix2> = ParamsBase<RawViewRepr<*const A>, D, A>;
/// A type alias for the [`ParamsBase`] whose elements are of type `*mut A` using a
/// [`RawViewRepr`] layout
pub type RawMutParams<A, D = Ix2> = ParamsBase<RawViewRepr<*mut A>, D, A>;
