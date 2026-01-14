/*
    Appellation: concision-data <library>
    Contrib: @FL03
*/
//! This crate works to augment the training process by providing datasets and loaders for
//! common data formats.
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::upper_case_acronyms
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(all(feature = "alloc", feature = "nightly"), feature(allocator_api))]
#![cfg_attr(all(feature = "autodiff", feature = "nightly"), feature(autodiff))]
#![crate_type = "lib"]
// compile-time checks
#[cfg(not(any(feature = "std", feature = "alloc")))]
compiler_error! { "Either the \"std\" feature or the \"alloc\" feature must be enabled." }
// external crates
#[cfg(feature = "alloc")]
extern crate alloc;
// macros
#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
}
// modules
pub mod dataset;
pub mod error;
#[cfg(feature = "loader")]
pub mod loader;
pub mod trainer;

pub mod traits {
    //! Additional traits and interfaces for working with datasets and data loaders.
    #[doc(inline)]
    pub use self::{convert::*, records::*};

    mod convert;
    mod records;
}
// re-exports
#[doc(inline)]
#[cfg(feature = "loader")]
pub use self::loader::*;
#[doc(inline)]
pub use self::{dataset::DatasetBase, error::*, trainer::*, traits::*};
// prelude
pub mod prelude {
    #[doc(no_inline)]
    pub use crate::dataset::*;
    #[cfg(feature = "loader")]
    #[doc(no_inline)]
    pub use crate::loader::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::*;
}
