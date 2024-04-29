/*
   Appellation: data <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Data
//!
#![feature(associated_type_defaults)]
extern crate concision_core as concision;

pub use self::utils::*;

pub(crate) mod primitives;
pub(crate) mod utils;

pub mod cmp;
pub mod datasets;
pub mod flows;
pub mod specs;

pub mod prelude {
    pub use crate::utils::*;

    pub use crate::cmp::*;
    pub use crate::datasets::*;
    pub use crate::flows::*;
    pub use crate::specs::ops::*;
    pub use crate::specs::*;
}
