/*
    Appellation: concision-math <library>
    Contrib: @FL03
*/
//! A suite of mathematical tool and utilities tailored toward neural networks.
//!
//! Our focus revolves around the following areas:
//!
//! - **Numerical Operations**: Efficient implementations of mathematical functions.
//! - **Statistical Functions**: Tools for statistical analysis and operations.
//! - **Signal Processing**: Functions and utilities for signal manipulation.
//! - **Utilities**: General-purpose utilities to aid in mathematical computations.
//!
#![cfg_attr(not(feature = "std"), no_std)]

pub use self::{error::*, traits::prelude::*, utils::prelude::*};

#[cfg(feature = "alloc")]
extern crate alloc;

#[macro_use]
pub(crate) mod macros;

pub mod error;
#[doc(hidden)]
pub mod signal;
pub mod stats;
pub mod utils;

pub mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    pub mod num;
    pub mod root;
    pub mod unary;

    pub(crate) mod prelude {
        #[doc(inline)]
        pub use super::num::*;
        #[doc(inline)]
        pub use super::root::*;
        #[doc(inline)]
        pub use super::unary::*;
    }
}
#[allow(unused_imports)]
pub mod prelude {
    #[doc(no_inline)]
    pub use crate::error::*;
    #[doc(hidden)]
    pub use crate::signal::prelude::*;
    pub use crate::stats::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::prelude::*;
    pub use crate::utils::prelude::*;
}
