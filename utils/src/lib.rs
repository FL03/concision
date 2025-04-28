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

#[doc(inline)]
pub use self::{error::*, ops::prelude::*, traits::prelude::*, utils::prelude::*};

#[cfg(feature = "alloc")]
extern crate alloc;

#[macro_use]
pub(crate) mod macros;

pub mod error;
#[cfg(feature = "signal")]
pub mod signal;
pub mod stats;

pub mod ops {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod unary;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::unary::*;
    }
}

pub mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    #[cfg(feature = "complex")]
    pub mod complex;
    pub mod difference;
    pub mod precision;
    pub mod root;

    pub(crate) mod prelude {
        #[cfg(feature = "complex")]
        #[doc(inline)]
        pub use super::complex::*;
        #[doc(inline)]
        pub use super::difference::*;
        #[doc(inline)]
        pub use super::precision::*;
        #[doc(inline)]
        pub use super::root::*;
    }
}

pub mod utils {
    //! utilties supporting various mathematical routines for machine learning tasks.
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod activate;
    pub mod arith;
    pub mod gradient;
    pub mod norm;
    pub mod patterns;
    pub mod tensor;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::activate::*;
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

#[allow(unused_imports)]
pub mod prelude {
    #[doc(no_inline)]
    pub use crate::error::*;
    #[cfg(feature = "signal")]
    #[doc(no_inline)]
    pub use crate::signal::prelude::*;
    #[doc(no_inline)]
    pub use crate::stats::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::prelude::*;
    #[doc(no_inline)]
    pub use crate::utils::prelude::*;
}
