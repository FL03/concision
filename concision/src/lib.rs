/*
   Appellation: concision <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! `concision` aims to be a complete machine-learning toolkit written in Rust. The framework
//! is designed to be performant, extensible, and easy to use while offering a wide range of
//! features for building and training machine learning models.
//!
//! ## Features
//!
//! - `ndarray`: extensive support for multi-dimensional arrays, enabling efficient data
//!   manipulation.
//!
//! ### Long term goals
//!
//! - **DSL**: Create a pseudo-DSL for defining machine learning models and training processes.
//! - **GPU**: Support for GPU acceleration to speed up training and inference.
//! - **Interoperability**: Integrate with other libraries and frameworks (TensorFlow, PyTorch)
//! - **Visualization**: Utilities for visualizing model architectures and training progress
//! - **WASM**: Native support for WebAssembly enabling models to be run in web browsers.
//!
#![crate_name = "concision"]

#[doc(inline)]
pub use concision_core::*;
#[doc(inline)]
#[cfg(feature = "data")]
pub use concision_data as data;
#[doc(inline)]
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[doc(inline)]
#[cfg(feature = "macros")]
pub use concision_macros::*;
#[doc(inline)]
#[cfg(feature = "neural")]
pub use concision_neural as nn;

pub mod prelude {
    #[doc(inline)]
    pub use concision_core::prelude::*;
    #[doc(inline)]
    #[cfg(feature = "data")]
    pub use concision_data::prelude::*;
    #[doc(inline)]
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[doc(inline)]
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
    #[doc(inline)]
    #[cfg(feature = "neural")]
    pub use concision_neural::prelude::*;
}
