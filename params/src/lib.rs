/*
    Appellation: concision-params <library>
    Created At: 2025.11.26:13:15:32
    Contrib: @FL03
*/
//! In machine learning, each layer is composed of some set of neurons that process input data
//! to produce some meaningful output. Each neuron typically has associated parameters, namely
//! weights and biases, which are adjusted during training to optimize the model's performance.
//!
//! ## Overview
//!
//! The [`params`](self) crate provides a generic and flexible structure for handling these
//! values. At its core, the [`ParamsBase`] object is defined as an object composed of two
//! independent tensors:
//!
//! - An $n$ dimensional weight tensor
//! - An $n-1$ dimensional bias tensor
//!
//! These tensors can be of any shape or size, allowing for a wide range of neural network
//! architectures to be represented. The crate also provides various utilities and traits for
//! manipulating and interacting with these parameters, making it easier to build and train
//! neural networks.
//!
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::should_implement_trait,
    clippy::upper_case_acronyms,
    rustdoc::redundant_explicit_links
)]
#![cfg_attr(all(feature = "nightly", feature = "alloc"), feature(allocator_api))]
#![cfg_attr(all(feature = "nightly", feature = "autodiff"), feature(autodiff))]
// compiler checks
#[cfg(not(any(feature = "alloc", feature = "std")))]
compiler_error! { "Either the \"alloc\" or \"std\" feature must be enabled for this crate." }
// external crates
#[cfg(feature = "alloc")]
extern crate alloc;
// macros
#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
}
// public modules
pub mod error;
pub mod iter;
// internal modules
mod params_base;

mod impls {
    mod impl_params;
    mod impl_params_ext;
    mod impl_params_iter;
    mod impl_params_ops;
    mod impl_params_rand;
    mod impl_params_ref;
    mod impl_params_repr;
    mod impl_params_serde;
}

mod traits {
    #[doc(inline)]
    pub use self::{iterators::*, raw_params::*, shape::*, wnb::*};

    mod iterators;
    mod raw_params;
    mod shape;
    mod wnb;
}

mod utils {
    #[doc(inline)]
    pub use self::shape::*;

    mod shape;
}
// re-exports
#[doc(inline)]
pub use self::{error::*, params_base::*, traits::*, utils::*};
// prelude
#[doc(hidden)]
pub mod prelude {
    pub use crate::params_base::*;
    pub use crate::traits::*;
    pub use crate::utils::*;
}
