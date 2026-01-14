//! Core traits defining fundamental abstractions and operations useful for neural networks.
//!
#![crate_type = "lib"]
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::should_implement_trait,
    clippy::upper_case_acronyms,
    rustdoc::redundant_explicit_links
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(all(feature = "nightly", feature = "alloc"), feature(allocator_api))]
#![cfg_attr(all(feature = "nightly", feature = "autodiff"), feature(autodiff))]
// compile-time checks
#[cfg(not(any(feature = "std", feature = "alloc")))]
compiler_error! {
    "At least one of the \"std\" or \"alloc\" features must be enabled for the crate to compile."
}
// external crates
#[cfg(feature = "alloc")]
extern crate alloc;

#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
}

mod impls {
    mod impl_backward;
    mod impl_forward;
}

mod clip;
mod codex;
mod complex;
mod entropy;
mod gradient;
mod init;
mod loss;
mod norm;
mod predict;
mod propagate;
mod rounding;
mod shuffle;
mod store;
mod training;

pub mod math {
    //! Mathematically oriented operators and functions useful in machine learning contexts.
    #[doc(inline)]
    pub use self::{linalg::*, percentages::*, roots::*, stats::*, unary::*};

    mod linalg;
    mod percentages;
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
#[doc(inline)]
pub use self::{
    clip::*, codex::*, complex::*, entropy::*, gradient::*, init::*, loss::*, math::*, norm::*,
    ops::*, predict::*, propagate::*, rounding::*, shuffle::*, store::*, tensor::*, training::*,
};
// prelude
#[doc(hidden)]
pub mod prelude {
    pub use crate::clip::*;
    pub use crate::codex::*;
    pub use crate::complex::*;
    pub use crate::entropy::*;
    pub use crate::gradient::*;
    pub use crate::init::*;
    pub use crate::loss::*;
    pub use crate::math::*;
    pub use crate::norm::*;
    pub use crate::ops::*;
    pub use crate::predict::*;
    pub use crate::propagate::*;
    pub use crate::rounding::*;
    pub use crate::shuffle::*;
    pub use crate::store::*;
    pub use crate::tensor::*;
    pub use crate::training::*;
}
