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

pub mod error;

mod apply;
mod clip;
mod codex;
mod complex;
mod container;
mod convert;
mod entropy;
mod loss;
mod norm;
mod predict;
mod propagation;
mod rounding;
mod store;
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
    #[allow(unused_imports)]
    #[doc(inline)]
    pub use self::{binary::*, fill::*, like::*, reshape::*, unary::*};

    mod binary;
    mod fill;
    mod like;
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
#[doc(inline)]
pub use self::error::*;
#[doc(inline)]
pub use self::prelude::*;

#[doc(hidden)]
pub mod prelude {
    pub use crate::apply::*;
    pub use crate::clip::*;
    pub use crate::codex::*;
    pub use crate::container::*;
    pub use crate::convert::*;
    pub use crate::entropy::*;
    pub use crate::loss::*;
    pub use crate::math::*;
    pub use crate::norm::*;
    pub use crate::ops::*;
    pub use crate::predict::*;
    pub use crate::propagation::*;
    pub use crate::rounding::*;
    pub use crate::store::*;
    pub use crate::tensor::*;
    pub use crate::training::*;

    #[cfg(feature = "complex")]
    pub use crate::complex::*;
}
