/*
    Appellation: concision-ext <library>
    Contrib: @FL03
*/
//! An extension of the [`concision`](https://crates.io/crates/concision) library, focusing on
//! providing additional layers and other non-essential features for building more complex
//! neural networks and machine learning models.
//!
//! ## Features
//!
//! - **attention**: Enables various attention mechanisms from scaled dot-product and
//!   multi-headed attention to FFT-based attention.
//!
#![allow(clippy::module_inception, clippy::needless_doctest_main)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate concision as cnc;

#[cfg(feature = "attention")]
pub use self::attention::prelude::*;

#[cfg(feature = "attention")]
pub mod attention;

#[cfg(feature = "kan")]
pub mod kan;
#[cfg(feature = "s4")]
pub mod s4;
#[cfg(feature = "snn")]
pub mod snn;
#[cfg(feature = "transformer")]
pub mod transformer;

pub mod prelude {
    #[cfg(feature = "attention")]
    pub use crate::attention::prelude::*;
    #[cfg(feature = "kan")]
    pub use crate::kan::prelude::*;
    #[cfg(feature = "s4")]
    pub use crate::s4::prelude::*;
    #[cfg(feature = "snn")]
    pub use crate::snn::prelude::*;
    #[cfg(feature = "transformer")]
    pub use crate::transformer::prelude::*;
}
