/*
    Appellation: concision-traits <library>
    Contrib: @FL03
*/
//! Traits for the concicion machine learning framework
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::should_implement_trait,
    clippy::upper_case_acronyms,
    rustdoc::redundant_explicit_links
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![crate_type = "lib"]

#[cfg(not(any(feature = "std", feature = "alloc")))]
compiler_error! {
    "At least one of the \"std\" or \"alloc\" features must be enabled for the crate to compile."
}

#[cfg(feature = "alloc")]
extern crate alloc;
extern crate ndarray as nd;

#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
}

mod impls {
    mod impl_activate_linear;
    mod impl_activate_nonlinear;
    mod impl_activator;
    mod impl_backward;
    mod impl_forward;
}

mod activate;
mod clip;
mod codex;
mod complex;
mod entropy;
mod loss;
mod norm;
mod predict;
mod propagate;
mod rho;
mod rounding;
mod training;

pub mod math {
    //! Mathematically oriented operators and functions useful in machine learning contexts.
    #[doc(inline)]
    pub use self::{difference::*, gradient::*, linalg::*, roots::*, stats::*, unary::*};

    mod difference;
    mod gradient;
    mod linalg;
    mod roots;
    mod stats;
    mod unary;
}

pub mod ops {
    //! composable operators for tensor manipulations and transformations, neural networks, and
    //! more
    #[doc(inline)]
    pub use self::{apply::*, fill::*, like::*, map::*, reshape::*, unary::*};

    mod apply;
    mod fill;
    mod like;
    mod map;
    mod reshape;
    mod unary;
}

pub mod tensor {
    #[doc(inline)]
    pub use self::{dimensionality::*, ndtensor::*, tensor_data::*};

    mod dimensionality;
    mod ndtensor;
    mod tensor_data;
}

// re-exports
#[cfg(feature = "complex")]
#[doc(inline)]
pub use self::complex::*;
#[doc(inline)]
pub use self::{
    activate::*, clip::*, codex::*, entropy::*, loss::*, math::*, norm::*, ops::*, predict::*,
    propagate::*, rho::*, rounding::*, tensor::*, training::*,
};
// prelude
#[doc(hidden)]
pub mod prelude {
    pub use crate::activate::*;
    pub use crate::clip::*;
    pub use crate::codex::*;
    pub use crate::entropy::*;
    pub use crate::loss::*;
    pub use crate::math::*;
    pub use crate::norm::*;
    pub use crate::ops::*;
    pub use crate::predict::*;
    pub use crate::propagate::*;
    pub use crate::rho::*;
    pub use crate::rounding::*;
    pub use crate::tensor::*;
    pub use crate::training::*;

    #[cfg(feature = "complex")]
    pub use crate::complex::*;
}
