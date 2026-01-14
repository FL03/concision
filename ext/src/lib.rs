/*
    Appellation: concision-ext <library>
    Contrib: @FL03
*/
//! An extension of the [`concision`](https://docs.rs/concision) library, focusing on
//! implementing additional layers, models, and utilities for deep learning applications.
//!
//! ## Features
//!
//! - `attention`: Enables attention mechanisms commonly used in transformer architectures.
//! - `snn`: Introduces spiking neural network components for neuromorphic computing.
//!
#![allow(
    clippy::missing_errors_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::upper_case_acronyms
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
extern crate concision as cnc;

#[cfg(feature = "attention")]
pub mod attention;
#[cfg(feature = "snn")]
pub mod snn;

/// re-exports
#[cfg(feature = "attention")]
pub use self::prelude::*;
// prelude
#[doc(hidden)]
pub mod prelude {
    #[cfg(feature = "attention")]
    pub use crate::attention::prelude::*;
    #[cfg(feature = "snn")]
    pub use crate::snn::prelude::*;
}
