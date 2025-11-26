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
mod difference;
mod entropy;
mod gradient;
mod loss;
mod norm;
mod propagation;
mod roots;
mod rounding;
mod store;
mod wnb;

pub mod ops {
    #[doc(inline)]
    pub use self::prelude::*;

    mod stats;
    mod unary;

    pub(crate) mod prelude {
        pub use super::stats::*;
        pub use super::unary::*;
    }
}

pub mod tensor {
    #[doc(inline)]
    pub use self::prelude::*;

    mod fill;
    mod like;
    mod reshape;
    mod shape;
    mod tensor_ops;

    pub(crate) mod prelude {
        pub use super::fill::*;
        pub use super::like::*;
        pub use super::reshape::*;
        pub use super::shape::*;
        pub use super::tensor_ops::*;
    }
}

// re-exports
#[doc(inline)]
pub use self::error::*;
#[doc(inline)]
pub use self::prelude::*;

#[doc(hidden)]
pub mod prelude {
    pub use crate::ops::*;
    pub use crate::tensor::*;

    pub use crate::apply::*;
    pub use crate::clip::*;
    pub use crate::codex::*;
    pub use crate::convert::*;
    pub use crate::difference::*;
    pub use crate::entropy::*;
    pub use crate::gradient::*;
    pub use crate::loss::*;
    pub use crate::norm::*;
    pub use crate::propagation::*;
    pub use crate::roots::*;
    pub use crate::rounding::*;
    pub use crate::store::*;
    pub use crate::wnb::*;

    #[cfg(feature = "complex")]
    pub use crate::complex::*;
}
