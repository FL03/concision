/*
    Appellation: signal <module>
    Contrib: @FL03
*/
//! # Signal Processing
//!
//! This module contains functions for signal processing such as convolution, filtering, and Fourier transforms.

#![cfg(feature = "signal")]
#[doc(inline)]
pub use self::prelude::*;

pub mod fft;

pub(crate) mod prelude {
    pub use super::fft::prelude::*;
}
