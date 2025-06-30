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
#![crate_type = "lib"]

#[cfg(not(all(feature = "std", feature = "alloc")))]
compiler_error! {
    "\
        Either the `std` or `alloc` feature must be enabled. 
        Please enable one of them in your Cargo.toml file.
    "
}

#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
}

#[cfg(feature = "alloc")]
extern crate alloc;

#[doc(inline)]
pub use self::{error::*, tensor::*, traits::*, types::*};

/// this module defines the [`TensorError`] enum for handling tensor-related errors
pub mod error;
/// this module defines various iterators for the [`TensorBase`]
pub mod iter;

mod tensor;

mod impls {
    mod impl_tensor;
    mod impl_tensor_iter;
    mod impl_tensor_ops;
    mod impl_tensor_repr;

    #[allow(deprecated)]
    mod impl_tensor_deprecated;
    #[cfg(feature = "init")]
    mod impl_tensor_init;
    #[cfg(feature = "rand")]
    mod impl_tensor_rand;
    #[cfg(feature = "serde")]
    mod impl_tensor_serde;
}

mod traits {
    //! this module provides additional traits for the `tensor` module
    #[doc(inline)]
    pub use self::prelude::*;

    mod ops;
    mod raw_tensor;
    mod scalar;

    mod prelude {
        #[doc(inline)]
        pub use super::ops::*;
        #[doc(inline)]
        pub use super::raw_tensor::*;
        #[doc(inline)]
        pub use super::scalar::*;
    }
}

mod types {
    //! this module defines various type aliases and primitives used by the `tensor` module
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
    #[doc(inline)]
    pub use super::tensor::*;
    #[doc(inline)]
    pub use super::traits::*;
    #[doc(inline)]
    pub use super::types::*;
}
