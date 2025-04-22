/*
    Appellation: params <module>
    Contrib: @FL03
*/
#[doc(inline)]
pub use self::params::ParamsBase;

pub mod iter;
pub mod params;

mod impls {
    pub(crate) mod impl_ops;
    pub(crate) mod impl_repr;
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::params::ParamsBase;
    #[doc(inline)]
    pub use super::{Params, ParamsView, ParamsViewMut};
}

/// a type alias for owned parameters
pub type Params<A, D = ndarray::Ix2> = ParamsBase<ndarray::OwnedRepr<A>, D>;
/// a type alias for shared parameters
pub type ArcParams<A, D = ndarray::Ix2> = ParamsBase<ndarray::OwnedArcRepr<A>, D>;
/// a type alias for an immutable view of the parameters
pub type ParamsView<'a, A, D = ndarray::Ix2> = ParamsBase<ndarray::ViewRepr<&'a A>, D>;
/// a type alias for a mutable view of the parameters
pub type ParamsViewMut<'a, A, D = ndarray::Ix2> = ParamsBase<ndarray::ViewRepr<&'a mut A>, D>;
/// a type alias for borrowed parameters
pub type CowParams<'a, A, D = ndarray::Ix2> = ParamsBase<ndarray::CowRepr<'a, A>, D>;
/// a raw view of the parameters; internally uses a constant pointer
pub type RawViewParams<A, D = ndarray::Ix2> = ParamsBase<ndarray::RawViewRepr<*const A>, D>;
/// a mutable raw view of the parameters; internally uses a mutable pointer
pub type RawMutParams<A, D = ndarray::Ix2> = ParamsBase<ndarray::RawViewRepr<*mut A>, D>;
