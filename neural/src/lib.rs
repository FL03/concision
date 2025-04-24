/*
    Appellation: concision-neural <library>
    Contrib: @FL03
*/
//! The neural network abstractions used to create and train models.
//! 
//! This library provides a 

#![crate_name = "concision_neural"]#![crate_type = "lib"]

extern crate concision_core as cnc;

#[doc(inline)]
pub use self::{
    error::*,
    model::{Model, ModelParams, StandardModelConfig},
    train::Trainer,
    traits::*,
    types::*,
};

#[macro_use]
pub(crate) mod macros;

pub mod error;
#[doc(hidden)]
pub mod layers;
pub mod model;
pub mod train;
pub mod utils;

pub mod traits {
    #[doc(inline)]
    pub use self::{activate::*, config::*};

    pub(crate) mod activate;
    pub(crate) mod config;
}

pub mod types {
    #[doc(inline)]
    pub use self::{dropout::*, features::*, hyperparameters::*};

    pub(crate) mod dropout;
    pub(crate) mod features;
    pub(crate) mod hyperparameters;
}

pub mod prelude {
    #[doc(hidden)]
    pub use crate::layers::prelude::*;
    #[doc(no_inline)]
    pub use crate::model::prelude::*;
    #[doc(no_inline)]
    pub use crate::train::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::*;
    #[doc(no_inline)]
    pub use crate::types::*;
}
