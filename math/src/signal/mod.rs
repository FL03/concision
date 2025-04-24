/*
    Appellation: signal <module>
    Contrib: @FL03
*/
//! # Signal Processing
//!
//! This module contains functions for signal processing such as convolution, filtering, and Fourier transforms.

#[cfg(feature = "std")]
pub mod fourier;

#[allow(unused_imports)]
pub(crate) mod prelude {
    #[cfg(feature = "std")]
    pub use super::fourier::prelude::*;
}
