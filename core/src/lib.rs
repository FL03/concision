/*
    Appellation: concision-core <library>
    Contrib: @FL03
*/
//! This crate provides the core implementations for the `cnc` framework, defining various
//! traits, types, and utilities essential for building neural networks.
//!
//! - [`ParamsBase`]: A structure for defining the parameters within a neural network.
//! - [`Backward`]: This trait establishes a common interface for backward propagation.
//! - [`Forward`]: This trait denotes a single forward pass through a layer of a neural network
//!
//! ## Features
//!
//! The crate is heavily feature-gate, enabling users to customize their experience based on
//! their needs.
//!
//! - `init`: Enables (random) initialization routines for models, parameters, and tensors.
//! - `utils`: Provides various utilities for developing machine learning models.
//!
//! ### Dependency-specific Features
//!
//! Additionally, the crate provides various dependency-specific features that can be enabled:
//!
//! - `anyhow`: Enables the use of the `anyhow` crate for error handling.
//! - `approx`: Enables approximate equality checks for floating-point numbers.
//! - `complex`: Enables complex number support.
//! - `json`: Enables JSON serialization and deserialization capabilities.
//! - `rand`: Enables random number generation capabilities.
//! - `serde`: Enables serialization and deserialization capabilities.
//! - `tracing`: Enables tracing capabilities for debugging and logging.
//!
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

#[cfg(feature = "rand")]
#[doc(no_inline)]
pub use rand;
#[cfg(feature = "rand")]
#[doc(no_inline)]
pub use rand_distr;

/// this module establishes generic random initialization routines for models, params, and
/// tensors.
#[doc(inline)]
#[cfg(feature = "cnc_init")]
pub use concision_init as init;
/// this module implements various utilities useful for developing machine learning models
#[doc(inline)]
#[cfg(feature = "cnc_utils")]
pub use concision_utils as utils;
/// An n-dimensional tensor
pub use ndtensor as tensor;

pub use ndtensor::prelude::*;

#[cfg(feature = "cnc_init")]
pub use self::init::prelude::*;
#[cfg(feature = "cnc_utils")]
pub use self::utils::prelude::*;

#[doc(inline)]
pub use self::{
    activate::prelude::*, error::*, loss::prelude::*, ops::prelude::*, params::prelude::*,
    traits::*,
};

#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
}
/// this module is dedicated to activation function
pub mod activate;
/// this module provides the base [`Error`] type for the library
pub mod error;
/// this module focuses on the loss functions used in training neural networks.
pub mod loss;
/// this module provides the [`ParamsBase`] type for the library, which is used to define the
/// parameters of a neural network.
pub mod params;

pub mod ops {
    //! This module provides the core operations for tensors, including filling, padding,
    //! reshaping, and tensor manipulation.
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod fill;
    pub mod mask;
    pub mod norm;
    pub mod pad;
    pub mod reshape;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::fill::*;
        #[doc(inline)]
        pub use super::mask::*;
        #[doc(inline)]
        pub use super::norm::*;
        #[doc(inline)]
        pub use super::pad::*;
        #[doc(inline)]
        pub use super::reshape::*;
    }
}

pub mod traits {
    //! This module provides the core traits for the library, such as  [`Backward`] and
    //! [`Forward`]
    #[doc(inline)]
    pub use self::prelude::*;

    mod apply;
    mod clip;
    mod codex;
    mod convert;
    mod gradient;
    mod like;
    mod propagation;
    mod shape;
    mod store;
    mod wnb;

    mod prelude {
        #[doc(inline)]
        pub use super::apply::*;
        #[doc(inline)]
        pub use super::clip::*;
        #[doc(inline)]
        pub use super::codex::*;
        #[doc(inline)]
        pub use super::convert::*;
        #[doc(inline)]
        pub use super::gradient::*;
        #[doc(inline)]
        pub use super::like::*;
        #[doc(inline)]
        pub use super::propagation::*;
        #[doc(inline)]
        pub use super::shape::*;
        #[doc(inline)]
        pub use super::store::*;
        #[doc(inline)]
        pub use super::wnb::*;
    }
}

#[doc(hidden)]
pub mod prelude {
    #[cfg(feature = "cnc_init")]
    pub use concision_init::prelude::*;
    #[cfg(feature = "cnc_utils")]
    pub use concision_utils::prelude::*;
    pub use ndtensor::prelude::*;

    #[doc(no_inline)]
    pub use crate::activate::prelude::*;
    #[doc(no_inline)]
    pub use crate::loss::prelude::*;
    #[doc(no_inline)]
    pub use crate::ops::prelude::*;
    #[doc(no_inline)]
    pub use crate::params::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::*;
}
