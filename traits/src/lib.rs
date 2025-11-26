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
mod convert;
mod fill;
mod gradient;
mod like;
mod norm;
mod propagation;
mod reshape;
mod shape;
mod store;
mod tensor_ops;
mod wnb;

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
    pub use crate::convert::*;
    pub use crate::fill::*;
    pub use crate::gradient::*;
    pub use crate::like::*;
    pub use crate::norm::*;
    pub use crate::propagation::*;
    pub use crate::reshape::*;
    pub use crate::shape::*;
    pub use crate::store::*;
    pub use crate::tensor_ops::*;
    pub use crate::wnb::*;
}
