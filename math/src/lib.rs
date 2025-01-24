/*
    Appellation: concision-math <library>
    Contrib: @FL03
*/
//! A collection of mathematical functions and utilities for signal processing, statistics, and more.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub use self::traits::prelude::*;

#[macro_use]
pub(crate) mod macros;

pub mod signal;
pub mod stats;
pub mod traits;

#[allow(unused_imports)]
pub mod prelude {
    pub use crate::signal::prelude::*;
    pub use crate::stats::prelude::*;
    pub use crate::traits::prelude::*;
}