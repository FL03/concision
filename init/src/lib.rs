/*
    Appellation: concision-init <library>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! Initialization related tools and utilities for neural networks and machine learning models.
//! This crate provides various initialization distributions and traits to facilitate
//! the effective initialization of model parameters.
//!
//! ## Features
//!
//! - `rand`: Enables random number generation functionalities using the `rand` crate.
//!
//! Implementors of the [`Initialize`] trait can leverage the various initialization
//! distributions provided within this crate to initialize their model parameters in a
//! manner conducive to effective training and convergence.
//!
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::should_implement_trait,
    clippy::upper_case_acronyms,
    rustdoc::redundant_explicit_links
)]
#![cfg_attr(not(feature = "std"), no_std)]

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
    pub use self::{lecun::*, trunc::*, xavier::*};

    pub mod lecun;
    pub mod trunc;
    pub mod xavier;

    pub(crate) mod prelude {
        pub use super::lecun::*;
        pub use super::trunc::*;
        pub use super::xavier::*;
    }
}

#[doc(hidden)]
pub mod prelude {
    pub use crate::error::InitError;
    pub use crate::traits::*;

    #[cfg(feature = "rand")]
    pub use crate::distr::prelude::*;
    #[cfg(feature = "rand")]
    pub use crate::utils::*;
}
