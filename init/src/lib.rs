/*
    Appellation: concision-init <library>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! One of the most important aspects of training neural networks and machine learning
//! lies within the _initialization_ of model parameters. Here, we work to provide additional
//! tools and utilities to facilitate effective initialization strategies including various
//! random distributions tailored directly to machine learning workloads such as:
//! Glorot (Xavier) initialization, LeCun initialization, etc.
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
    pub use crate::error::InitError;
    pub use crate::traits::*;

    #[cfg(feature = "rand")]
    pub use crate::distr::prelude::*;
    #[cfg(feature = "rand")]
    pub use crate::utils::*;
}
