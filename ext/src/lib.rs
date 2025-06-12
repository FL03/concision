/*
    Appellation: concision-models <library>
    Contrib: @FL03
*/
//! # concision-ext
//!
//! This library uses the [`concision`](https://docs.rs/concision) framework to implement a
//! variety of additional machine learning models and layers.
//!
#![allow(clippy::module_inception, clippy::needless_doctest_main)]
#![cfg_attr(not(feature = "std"), no_std)]

// #[cfg(feature = "alloc")]
// extern crate alloc;

extern crate concision as cnc;

#[cfg(feature = "simple")]
pub mod simple;
#[cfg(feature = "kan")]
pub use concision_kan as kan;
#[cfg(feature = "s4")]
pub use concision_s4 as s4;
#[cfg(feature = "transformer")]
pub use concision_transformer as transformer;

pub mod prelude {
    #[cfg(feature = "simple")]
    pub use crate::simple::SimpleModel;
    #[cfg(feature = "kan")]
    pub use concision_kan::prelude::*;
    #[cfg(feature = "s4")]
    pub use concision_s4::prelude::*;
    #[cfg(feature = "transformer")]
    pub use concision_transformer::prelude::*;
}
