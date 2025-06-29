/*
    Appellation: concision-utils <library>
    Contrib: @FL03
*/
//! A suite of utilities tailored toward neural networks.
//!
//! Our focus revolves around the following areas:
//!
//! - **Numerical Operations**: Efficient implementations of mathematical functions.
//! - **Statistical Functions**: Tools for statistical analysis and operations.
//! - **Signal Processing**: Functions and utilities for signal manipulation.
//! - **Utilities**: General-purpose utilities to aid in mathematical computations.
//!
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub(crate) mod seal;
    #[macro_use]
    pub(crate) mod unary;
}

#[doc(inline)]
pub use self::{error::*, ops::*, traits::*, utils::*};

pub mod error;
#[cfg(feature = "signal")]
pub mod signal;
pub mod stats;

mod ops {
    #[doc(inline)]
    pub use self::prelude::*;

    mod unary;

    mod prelude {
        #[doc(inline)]
        pub use super::unary::*;
    }
}

mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    #[cfg(feature = "complex")]
    mod complex;
    mod difference;
    mod precision;
    mod root;

    mod prelude {
        #[doc(inline)]
        #[cfg(feature = "complex")]
        pub use super::complex::*;
        #[doc(inline)]
        pub use super::difference::*;
        #[doc(inline)]
        pub use super::precision::*;
        #[doc(inline)]
        pub use super::root::*;
    }
}

mod utils {
    //! utilties supporting various mathematical routines for machine learning tasks.
    #[doc(inline)]
    pub use self::prelude::*;

    mod arith;
    mod gradient;
    mod norm;
    mod patterns;
    mod tensor;

    mod prelude {
        #[doc(inline)]
        pub use super::arith::*;
        #[doc(inline)]
        pub use super::gradient::*;
        #[doc(inline)]
        pub use super::norm::*;
        #[doc(inline)]
        pub use super::patterns::*;
        #[doc(inline)]
        pub use super::tensor::*;
    }
}

#[doc(hidden)]
pub mod prelude {
    pub use crate::stats::prelude::*;
    pub use crate::traits::*;
    pub use crate::utils::*;

    #[cfg(feature = "signal")]
    pub use crate::signal::prelude::*;
}
