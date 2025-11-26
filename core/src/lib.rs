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

#[doc(inline)]
pub use concision_traits as traits;

/// this module establishes generic random initialization routines for models, params, and
/// tensors.
#[doc(inline)]
pub use concision_init as init;
#[doc(inline)]
pub use concision_params as params;

#[doc(inline)]
pub use self::{
    activate::prelude::*,
    error::*,
    init::{Init, InitInplace, InitRand},
    ops::prelude::*,
    params::prelude::*,
    traits::prelude::*,
    utils::*,
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
#[cfg(feature = "signal")]
pub mod signal;

pub mod ops;
pub mod utils;

#[doc(hidden)]
pub mod prelude {
    pub use concision_init::prelude::*;
    pub use concision_params::prelude::*;
    pub use concision_traits::prelude::*;

    #[cfg(feature = "signal")]
    pub use crate::signal::prelude::*;

    pub use crate::activate::prelude::*;
    pub use crate::ops::prelude::*;
    pub use crate::utils::*;
}
