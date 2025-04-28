/*
    Appellation: signal <module>
    Contrib: @FL03
*/
//! # Signal Processing
//!
//! This module contains functions for signal processing such as convolution, filtering, and Fourier transforms.

#[cfg(all(feature = "complex", feature = "std"))]
pub mod fourier;

#[allow(unused_imports)]
pub(crate) mod prelude {
    #[cfg(all(feature = "complex", feature = "std"))]
    pub use super::fourier::prelude::*;
}
