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
//! - `rand`: Enables random number generation
//! - `serde`: Enables serialization and deserialization capabilities.
//! - `tracing`: Enables tracing capabilities for debugging and logging.
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
#![cfg_attr(all(feature = "nightly", feature = "alloc"), feature(allocator_api))]
#![cfg_attr(all(feature = "nightly", feature = "autodiff"), feature(autodiff))]
// compile-time checks
#[cfg(not(any(feature = "std", feature = "alloc")))]
compiler_error! { "Either the \"std\" feature or the \"alloc\" feature must be enabled." }
// external crates
#[cfg(feature = "alloc")]
extern crate alloc;
// re-definitions
#[doc(inline)]
#[cfg(feature = "rand")]
pub use concision_init as init;
#[doc(inline)]
pub use concision_params as params;
#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod config;
    #[macro_use]
    pub mod gsw;
    #[macro_use]
    pub mod seal;
    #[macro_use]
    pub mod units;
}

pub mod activate;
pub mod config;
pub mod error;
pub mod models;
pub mod nn;
pub mod utils;

#[doc(hidden)]
pub mod ex {
    pub mod sample;
}

mod types {
    #[doc(inline)]
    pub use self::parameters::*;

    mod parameters;
}
// re-exports
#[doc(inline)]
pub use self::{
    activate::{Activate, Activator, ActivatorGradient},
    config::StandardModelConfig,
    error::*,
    models::prelude::*,
    nn::prelude::*,
    types::*,
    utils::*,
};
#[doc(inline)]
#[cfg(feature = "rand")]
pub use concision_init::prelude::*;
#[doc(inline)]
pub use concision_params::prelude::*;
#[doc(inline)]
pub use concision_traits::prelude::*;

// prelude
#[doc(hidden)]
pub mod prelude {
    #[cfg(feature = "rand")]
    pub use concision_init::prelude::*;
    pub use concision_params::prelude::*;
    pub use concision_traits::prelude::*;

    pub use crate::config::prelude::*;
    pub use crate::models::prelude::*;
    pub use crate::nn::prelude::*;
    pub use crate::types::*;
    pub use crate::utils::*;
}
