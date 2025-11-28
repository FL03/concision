/*
   Appellation: concision <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision (cnc)
//!
//! [![crates.io](https://img.shields.io/crates/v/concision?style=for-the-badge&logo=rust)](https://crates.io/crates/concision)
//! [![docs.rs](https://img.shields.io/docsrs/concision?style=for-the-badge&logo=docs.rs)](https://docs.rs/concision)
//! [![GitHub License](https://img.shields.io/github/license/FL03/concision?style=for-the-badge&logo=github)](https://github.com/FL03/concision/blob/main/LICENSE)
//!
//! ***
//!
//! `concision` aims to be a complete machine-learning toolkit written in Rust. The framework
//! is designed to be performant, extensible, and easy to use while offering a wide range of
//! features for building and training machine learning models.
//!
//! The framework relies heavily on the [`ndarray`](https://docs.rs/ndarray) crate for its
//! n-dimensional arrays, which are essential for efficient data manipulation and mathematical
//! operations.
//!
//! ## Features
//!
//! - `data`: Provides utilities for data loading, preprocessing, and augmentation.
//! - `derive`: Custom derive macros for automatic implementation of traits
//! - `init`: Enables various initialization strategies for model parameters.
//! - `macros`: Procedural macros for simplifying common tasks in machine learning.
//! - `neural`: A neural network module that includes layers, optimizers, and training
//!   utilities.
//!
//! ### _Extensions_
//!
//! The crate is integrated with several optional externcal crates that are commonly used in
//! Rust development; listed below are some of the most relevant of these _extensions_ as they
//! add additional functionality to the framework.
//!
//! - [`approx`](https://docs.rs/approx): Enables approximate equality checks for
//!   floating-point arithmetic, useful for testing and validation of model outputs.
//! - `json`: Enables JSON serialization and deserialization for models and data.
//! - [`rayon`](https://docs.rs/rayon): Enables parallel processing for data loading and
//!   training.
//! - [`serde`](https://serde.rs): Enables the `serde` crate for the serialization and
//!   deserialization of models and data.
//! - [`tracing`](https://docs.rs/tracing): Enables the `tracing` crate for structured logging
//!   and diagnostics.
//!
//! ## Roadmap
//!
//! - **DSL**: Create a pseudo-DSL for defining machine learning models and training processes.
//! - **GPU**: Support for GPU acceleration to speed up training and inference.
//! - **Interoperability**: Integrate with other libraries and frameworks (TensorFlow, PyTorch)
//! - **Visualization**: Utilities for visualizing model architectures and training progress
//! - **WASM**: Native support for WebAssembly enabling models to be run in web browsers.
//!
#![crate_type = "lib"]
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::upper_case_acronyms
)]
#![cfg_attr(not(feature = "std"), no_std)]

#[doc(inline)]
pub use concision_core::*;
#[doc(inline)]
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[doc(inline)]
#[cfg(feature = "macros")]
pub use concision_macros::*;

/// this module contains various data loaders, preprocessors, and augmenters
#[doc(inline)]
#[cfg(feature = "data")]
pub use concision_data as data;

#[doc(hidden)]
pub mod prelude {
    #[doc(no_inline)]
    pub use concision_core::prelude::*;
    #[doc(no_inline)]
    #[cfg(feature = "data")]
    pub use concision_data::prelude::*;
    #[doc(no_inline)]
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[doc(no_inline)]
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
}
