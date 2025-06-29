/*
    Appellation: params <module>
    Contrib: @FL03
*/
//! Parameters for constructing neural network models. This module implements parameters using
//! the [ParamsBase] struct and its associated types. The [ParamsBase] struct provides:
//!
//! - An (n) dimensional weight tensor as [ArrayBase](ndarray::ArrayBase)
//! - An (n-1) dimensional bias tensor as [ArrayBase](ndarray::ArrayBase)
//!
//! The associated types follow suite with the [`ndarray`] crate, each of which defines a
//! different style of representation for the parameters.
#[doc(inline)]
pub use self::{error::ParamsError, params::ParamsBase};

pub mod error;
pub mod iter;
pub mod params;

mod impls {
    mod impl_params;
    #[allow(deprecated)]
    mod impl_params_deprecated;
    mod impl_params_init;
    mod impl_params_iter;
    mod impl_params_ops;
    #[cfg(feature = "rand")]
    mod impl_params_rand;
    #[cfg(feature = "serde")]
    mod impl_params_serde;
}

pub(crate) mod prelude {
    pub use super::error::ParamsError;
    pub use super::params::ParamsBase;
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
