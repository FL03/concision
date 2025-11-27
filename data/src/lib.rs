/*
    Appellation: concision-train <library>
    Contrib: @FL03
*/
//! Datasets and data loaders for the Concision framework.
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

pub mod dataset;
pub mod error;
#[cfg(feature = "loader")]
pub mod loader;
pub mod trainer;

#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
}

pub mod traits {
    #[doc(inline)]
    pub use self::{convert::*, records::*, train::*, trainers::*};

    mod convert;
    mod records;
    mod train;
    mod trainers;
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
