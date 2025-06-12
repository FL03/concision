/*
    appellation: concision-kan <library>
    authors: @FL03
*/
//! # `concision-kan`
//!
//! This library provides an implementation of the Kolmogorov–Arnold Networks (kan) model using
//! the [`concision`](https://docs.rs/concision) framework.
//!
//! ## References
//!
//! - [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/html/2404.19756v1)
//!
#![crate_name = "concision_kan"]
#![crate_type = "lib"]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::module_inception)]

extern crate concision as cnc;

#[doc(inline)]
pub use self::model::*;

pub mod model;

pub mod prelude {
    #[doc(inline)]
    pub use super::model::*;
}
