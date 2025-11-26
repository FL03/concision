/*
    appellation: concision-snn <library>
    authors: @FL03
*/
//!
//!
//! ## References
//!
//! - [Deep Learning in Spiking Neural Networks](https://arxiv.org/abs/1804.08150)
//!
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
