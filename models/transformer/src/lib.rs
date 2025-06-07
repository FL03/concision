/*
    appellation: concision-transformer <library>
    authors: @FL03
*/
//! # `concision-transformer`
//!
//! `concision-transformer` is a library for building and training transformer models
#![crate_name = "concision_transformer"]
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
