/*
    Appellation: concision-neural <library>
    Contrib: @FL03
*/
//! # concision-neural (cnc::neural)
//!
//! The neural network abstractions used to create and train models.
//!
//! ## Features
//!
//! - [`Model`]: A trait for defining a neural network model.
//! - [`ModelParams`]: A structure for storing the parameters of a neural network model.
//! - [`StandardModelConfig`]: A standard configuration for the models
//! - [`Predict`]: A trait extending the basic [`Forward`](cnc::Forward) pass
//! - [`Train`]: A trait for training a neural network model.
//!
//! ### _Work in Progress_
//!
//! - [LayerBase]: Functional wrappers for the [ParamsBase](cnc::ParamsBase) structure.

#![crate_name = "concision_neural"]
#![crate_type = "lib"]

extern crate concision_core as cnc;

#[doc(inline)]
pub use self::{
    error::*,
    layers::{Layer, LayerBase},
    model::prelude::*,
    traits::*,
    types::*,
};

#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
}

pub mod error;
pub mod layers;
pub mod model;
pub mod utils;

pub(crate) mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    pub(crate) mod config;
    pub(crate) mod hidden;
    pub(crate) mod predict;
    pub(crate) mod train;

    mod prelude {
        #[doc(inline)]
        pub use super::config::*;
        #[doc(inline)]
        pub use super::hidden::*;
        #[doc(inline)]
        pub use super::predict::*;
        #[doc(inline)]
        pub use super::train::*;
    }
}

pub(crate) mod types {
    #[doc(inline)]
    pub use self::prelude::*;

    mod dropout;
    mod hyperparameters;
    mod key_value;

    mod prelude {
        #[doc(inline)]
        pub use super::dropout::*;
        #[doc(inline)]
        pub use super::hyperparameters::*;
        #[doc(inline)]
        pub use super::key_value::*;
    }
}

pub mod prelude {
    #[doc(no_inline)]
    pub use crate::error::*;
    #[doc(no_inline)]
    pub use crate::layers::prelude::*;
    #[doc(no_inline)]
    pub use crate::model::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::*;
    #[doc(no_inline)]
    pub use crate::types::*;
}
