/*
    Appellation: init <library>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-init
//!
//! This library provides various random distribution and initialization routines for the
//! `concision` framework. It includes implementations for different initialization strategies
//! optimized for neural networks, such as Glorot (Xavier) initialization, LeCun
//! initialization, etc.
//!
#![cfg(feature = "rand")]
#![allow(
    clippy::missing_saftey_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::should_implement_trait,
    clippy::upper_case_acronyms
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly", feature(allocator_api))]

#[doc(inline)]
pub use self::{distr::prelude::*, error::*, traits::*, utils::*};

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "rand")]
#[doc(no_inline)]
pub use rand;
#[cfg(feature = "rand")]
#[doc(no_inline)]
pub use rand_distr;

pub mod error;
#[cfg(feature = "rand")]
pub(crate) mod utils;

#[cfg(feature = "rand")]
mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    mod initialize;

    mod prelude {
        #[doc(inline)]
        pub use super::initialize::*;
    }
}

#[cfg(feature = "rand")]
pub mod distr {
    //! this module implements various random distributions optimized for neural network
    //! initialization.
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod lecun;
    pub mod trunc;
    pub mod xavier;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::lecun::*;
        #[doc(inline)]
        pub use super::trunc::*;
        #[doc(inline)]
        pub use super::xavier::*;
    }
}

#[doc(hidden)]
pub mod prelude {
    #[cfg(feature = "rand")]
    pub use super::distr::prelude::*;
    #[cfg(feature = "rand")]
    pub use super::traits::*;
    #[cfg(feature = "rand")]
    pub use super::utils::*;
}

