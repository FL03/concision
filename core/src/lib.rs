/*
    Appellation: concision-core <library>
    Contrib: @FL03
*/
//! # concision-core
//!
//! This library provides the core abstractions and utilities for the concision (cnc) machine
//! learning framework.
//!
//! ## Features
//!
//! - [`ParamsBase`]: A structure for defining the parameters within a neural network.
//! - [`Backward`]: This trait establishes a common interface for backward propagation.
//! - [`Forward`]: This trait denotes a single forward pass through a layer of a neural network
//!
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::upper_case_acronyms
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![crate_type = "lib"]

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
#[cfg(feature = "concision_init")]
pub use concision_init as init;

#[cfg(feature = "utils")]
pub use concision_utils as utils;

#[cfg(feature = "concision_init")]
pub use self::init::prelude::*;
#[cfg(feature = "utils")]
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
    pub mod pad;
    pub mod reshape;
    pub mod tensor;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::fill::*;
        #[doc(inline)]
        pub use super::pad::*;
        #[doc(inline)]
        pub use super::reshape::*;
        #[doc(inline)]
        pub use super::tensor::*;
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
    mod gradient;
    mod like;
    mod mask;
    mod norm;
    mod propagation;
    mod scalar;
    mod tensor;
    mod wnb;

    mod prelude {
        #[doc(inline)]
        pub use super::apply::*;
        #[doc(inline)]
        pub use super::clip::*;
        #[doc(inline)]
        pub use super::codex::*;
        #[doc(inline)]
        pub use super::gradient::*;
        #[doc(inline)]
        pub use super::like::*;
        #[doc(inline)]
        pub use super::mask::*;
        #[doc(inline)]
        pub use super::norm::*;
        #[doc(inline)]
        pub use super::propagation::*;
        #[doc(inline)]
        pub use super::scalar::*;
        #[doc(inline)]
        pub use super::tensor::*;
        #[doc(inline)]
        pub use super::wnb::*;
    }
}

#[doc(hidden)]
pub mod prelude {
    #[cfg(feature = "concision_init")]
    pub use concision_init::prelude::*;
    #[cfg(feature = "utils")]
    pub use concision_utils::prelude::*;

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
