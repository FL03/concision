/*
    Appellation: concision-init <library>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-init
//!
//! This library provides various random distribution and initialization routines for the
//! `concision` framework. It includes implementations for different initialization strategies
//! optimized for neural networks, such as Glorot (Xavier) initialization, LeCun
//! initialization, etc.
//!
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::should_implement_trait,
    clippy::upper_case_acronyms
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly", feature(allocator_api))]

#[doc(inline)]
#[cfg(feature = "rand")]
pub use self::{distr::prelude::*, utils::*};
#[doc(inline)]
pub use self::{error::*, traits::*};

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "rand")]
#[doc(no_inline)]
pub use rand;
#[cfg(feature = "rand")]
#[doc(no_inline)]
pub use rand_distr;

pub mod error;

pub(crate) mod utils {
    //! this module provides various utility functions for random initialization.
    #[doc(inline)]
    #[cfg(feature = "rand")]
    pub use self::prelude::*;

    #[cfg(feature = "rand")]
    mod rand_utils;

    mod prelude {
        #[cfg(feature = "rand")]
        pub use super::rand_utils::*;
    }
}

mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    mod init;
    #[cfg(feature = "rand")]
    mod initialize;

    mod prelude {
        #[doc(inline)]
        pub use super::init::*;
        #[doc(inline)]
        #[cfg(feature = "rand")]
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
    pub use super::traits::*;
    #[cfg(feature = "rand")]
    pub use super::utils::*;
}
