/*
    Appellation: concision-traits <library>
    Contrib: @FL03
*/
//! Traits for the concicion machine learning framework
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::upper_case_acronyms
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![crate_type = "lib"]

#[cfg(not(all(feature = "std", feature = "alloc")))]
compiler_error! {
    "At least one of the 'std' or 'alloc' features must be enabled."
}

#[cfg(feature = "alloc")]
extern crate alloc;

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
mod convert;
mod entropy;
mod loss;
mod norm;
mod predict;
mod propagation;
mod rounding;
mod store;

pub mod math {
    //! Mathematically oriented operators and functions useful in machine learning contexts.
    #[doc(inline)]
    pub use self::{difference::*, gradient::*, roots::*, stats::*, unary::*};

    mod difference;
    mod gradient;
    mod roots;
    mod stats;
    mod unary;
}

pub mod tensor {
    #[doc(inline)]
    pub use self::{fill::*, like::*, linalg::*, ndtensor::*, shape::*};

    mod fill;
    mod like;
    mod linalg;
    mod ndtensor;
    mod shape;
}

// re-exports
#[doc(inline)]
pub use self::error::*;
#[doc(inline)]
pub use self::prelude::*;

#[doc(hidden)]
pub mod prelude {
    pub use crate::math::*;
    pub use crate::tensor::*;

    pub use crate::apply::*;
    pub use crate::clip::*;
    pub use crate::codex::*;
    pub use crate::convert::*;
    pub use crate::entropy::*;
    pub use crate::loss::*;
    pub use crate::norm::*;
    pub use crate::predict::*;
    pub use crate::propagation::*;
    pub use crate::rounding::*;
    pub use crate::store::*;

    #[cfg(feature = "complex")]
    pub use crate::complex::*;
}
