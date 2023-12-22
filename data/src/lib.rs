/*
   Appellation: data <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Data
//!
#![feature(associated_type_defaults)]

pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod datasets;
pub mod df;
pub mod flows;
pub mod tensors;

pub mod prelude {
    pub use linfa::dataset::{Dataset, DatasetBase, DatasetView};

    pub use crate::datasets::*;
    pub use crate::df::*;
    pub use crate::flows::*;
    pub use crate::tensors::*;

    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
