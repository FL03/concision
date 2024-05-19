/*
   Appellation: fft <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Fast Fourier Transform
//!
//! The `fft` module provides an implementation of the Fast Fourier Transform (FFT) algorithm.
//! The Fast Fourier Transform is an efficient algorithm for computing the Discrete Fourier Transform (DFT).
pub use self::prelude::*;

pub(crate) mod fft;
pub(crate) mod utils;

pub mod cmp {
    pub use self::prelude::*;

    pub mod direction;
    pub mod mode;
    pub mod plan;

    pub(crate) mod prelude {
        pub use super::direction::FftDirection;
        pub use super::mode::FftMode;
        pub use super::plan::FftPlan;
    }
}

/// Trait for computing the Discrete Fourier Transform (DFT) of a sequence.
pub trait DFT<T> {
    type Output;

    fn dft(&self) -> Self::Output;
}

pub(crate) mod prelude {
    pub use super::cmp::prelude::*;
    pub use super::fft::*;
    pub use super::utils::*;
    pub use super::DFT;
}

#[cfg(test)]
mod tests {}
