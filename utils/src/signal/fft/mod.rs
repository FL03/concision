/*
    appellation: fft <module>
    authors: @FL03
*/
//! this module implements the custom fast-fourier transform (FFT) algorithm
#[doc(inline)]
pub use self::{types::prelude::*, utils::*};

#[doc(hidden)]
pub mod dft;
/// this module implements the methods for the fast-fourier transform (FFT) module
pub mod utils;

pub mod types {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod mode;
    pub mod plan;

    pub(crate) mod prelude {
        pub use super::mode::*;
        pub use super::plan::*;
    }
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::utils::*;
}

/// The [`DFT`] trait establishes a common interface for discrete Fourier transform
/// implementations.
pub trait DFT<T> {
    type Output;

    fn dft(&self) -> Self::Output;
}

#[cfg(test)]
mod tests {
    use super::FftPlan;
    use super::utils::fft_permutation;

    #[test]
    fn test_plan() {
        let samples = 16;

        let plan = FftPlan::new(samples).build();
        assert_eq!(plan.plan(), fft_permutation(16).as_slice());
    }
}
