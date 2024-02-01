/*
   Appellation: data <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Data
//!
#![feature(associated_type_defaults)]
pub use self::utils::*;

pub(crate) mod primitives;
pub(crate) mod utils;

pub mod cmp;
pub mod datasets;
pub mod flows;
pub mod shape;
pub mod specs;
pub mod store;
pub mod tensors;

pub(crate) use concision_core as core;

pub mod prelude {
    pub use crate::utils::*;

    pub use crate::cmp::*;
    pub use crate::datasets::*;
    pub use crate::flows::*;
    pub use crate::shape::*;
    pub use crate::specs::ops::*;
    pub use crate::specs::*;
    pub use crate::store::*;
    pub use crate::tensors::*;
}
