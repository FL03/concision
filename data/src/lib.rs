/*
   Appellation: data <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Data
//!
//! This library works to provide a comprehensive set of utilities for working with datasets.
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate concision_core as concision;

pub use self::dataset::Dataset;
pub use self::traits::prelude::*;

pub mod dataset;
#[doc(hidden)]
pub mod preproc;
pub mod tensor;
pub mod traits;

pub mod prelude {
    pub use crate::dataset::*;
    pub use crate::traits::prelude::*;
}
