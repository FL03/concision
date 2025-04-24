/*
    Appellation: concision-neural <library>
    Contrib: @FL03
*/
//! A collection of mathematical functions and utilities for signal processing, statistics, and more.

#![crate_name = "concision_neural"]
#![crate_type = "lib"]

extern crate concision_core as cnc;

#[allow(unused_imports)]
#[doc(inline)]
pub use self::{
    error::*,
    layer::Layer,
    model::{Model, ModelConfig, ModelParams},
    traits::*,
    types::*,
    utils::*,
};

#[allow(unused_macros)]
#[macro_use]
pub(crate) mod macros;

pub mod error;
pub mod layer;
pub mod model;
pub mod train;
pub mod utils;

pub mod traits {}

pub mod types {
    #[doc(inline)]
    pub use self::{dropout::*, features::*, hyperparameters::*};

    pub(crate) mod dropout;
    pub(crate) mod features;
    pub(crate) mod hyperparameters;
}

#[allow(unused_imports)]
pub mod prelude {
    pub use crate::layer::prelude::*;
    pub use crate::model::prelude::*;
    pub use crate::train::prelude::*;
    pub use crate::traits::*;
    pub use crate::types::*;
    pub use crate::utils::*;
}
