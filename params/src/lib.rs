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
//! - An $`n`$ dimensional weight tensor
//! - An $`n-1`$ dimensional bias tensor
//!
//! These tensors can be of any shape or size, allowing for a wide range of neural network
//! architectures to be represented. The crate also provides various utilities and traits for
//! manipulating and interacting with these parameters, making it easier to build and train
//! neural networks.
//!
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::missing_saftey_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::upper_case_acronyms
)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(all(not(feature = "alloc"), not(feature = "std")))]
compiler_error! {
 "Either the \"alloc\" or \"std\" feature must be enabled for this crate."
}

#[doc(inline)]
pub use self::{error::*, params::ParamsBase, types::*};

#[cfg(feature = "init")]
extern crate concision_init as init;

/// Error handling for parameters
pub mod error;
/// The [`iter`] module implements various iterators for parameters
pub mod iter;

mod params;

mod impls {
    mod impl_params;
    #[allow(deprecated)]
    mod impl_params_deprecated;
    #[cfg(feature = "init")]
    mod impl_params_init;
    mod impl_params_iter;
    mod impl_params_ops;
    #[cfg(feature = "rand")]
    mod impl_params_rand;
    #[cfg(feature = "serde")]
    mod impl_params_serde;
}

mod types {
    //! Additional types supporting the params module
    #[doc(inline)]
    pub use self::prelude::*;

    mod aliases;

    mod prelude {
        #[doc(inline)]
        pub use super::aliases::*;
    }
}

#[doc(hidden)]
pub mod prelude {
    pub use crate::error::ParamsError;
    pub use crate::params::*;
    pub use crate::types::*;
}
