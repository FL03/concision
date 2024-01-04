/*
   Appellation: data <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Data
//!
#![feature(associated_type_defaults)]

pub use self::misc::*;

pub(crate) mod misc;
pub(crate) mod primitives;
pub(crate) mod utils;

pub mod datasets;
pub mod df;
pub mod flows;
pub mod mat;
pub mod shape;
pub mod specs;
pub mod store;
pub mod tensors;

pub(crate) use concision_core as core;

pub mod prelude {
    // pub use linfa::dataset::{Dataset, DatasetBase, DatasetView};

    pub use crate::datasets::*;
    pub use crate::df::*;
    pub use crate::flows::*;
    pub use crate::shape::*;
    pub use crate::specs::*;
    pub use crate::store::*;
    pub use crate::tensors::*;
}
