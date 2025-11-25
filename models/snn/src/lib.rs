/*
    appellation: concision-snn <library>
    authors: @FL03
*/
//! 
//!
//! ## References
//!
//! - [Structured State Spaces for Sequence Modeling](https://arxiv.org/abs/2106.08084)
//! - [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
//!
#![crate_name = "concision_s4"]
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
