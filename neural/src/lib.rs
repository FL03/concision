/*
    Appellation: concision-neural <library>
    Contrib: @FL03
*/
//! # concision-neural
//!
//! This crate focuses on implementing various neural network components, including models,
//! layers, and training mechanisms.
//!
//! ## Overview
//!
//! ## Features
//!
//! - [`Model`]: A trait for defining a neural network model.
//! - [`ModelParamsBase`]: A dedicated object capable of storing the parameters for both
//!   shallow and deep neural networks.
//! - [`StandardModelConfig`]: A standard configuration for the models
//! - [`Predict`]: A trait extending the basic [`Forward`](cnc::Forward) pass
//! - [`Train`]: A trait for training a neural network model.
#![crate_type = "lib"]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::missing_saftey_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::upper_case_acronyms
)]

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate concision_core as cnc;

#[doc(inline)]
pub use self::{
    config::prelude::*,
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

pub mod config;
pub mod error;
pub mod layers;
pub mod model;
pub mod utils;

pub(crate) mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    pub(crate) mod hidden;
    pub(crate) mod predict;
    pub(crate) mod train;

    mod prelude {
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
    mod key_value;

    mod prelude {
        #[doc(inline)]
        pub use super::dropout::*;
        #[doc(inline)]
        pub use super::key_value::*;
    }
}

#[doc(hidden)]
pub mod prelude {
    #[doc(no_inline)]
    pub use super::config::prelude::*;
    // #[doc(no_inline)]
    // pub use crate::layers::prelude::*;
    #[doc(no_inline)]
    pub use crate::model::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::*;
    #[doc(no_inline)]
    pub use crate::types::*;
}
