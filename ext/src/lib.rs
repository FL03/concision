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

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate concision as cnc;

#[cfg(feature = "attention")]
pub use self::attention::prelude::*;

#[cfg(feature = "attention")]
pub mod attention;

pub mod simple;

pub mod prelude {
    #[cfg(feature = "attention")]
    pub use crate::attention::prelude::*;
}
