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
//! Neural networks are a fundamental part of machine learning, and this crate provides a
//! comprehensive set of tools to build, configure, and train neural network models. Listed
//! below are several key components of the crate:
//!
//! - [`Model`]: A trait for defining a neural network model.
//! - [`ModelParamsBase`]: A dedicated object capable of storing the parameters for both
//!   shallow and deep neural networks.
//! - [`StandardModelConfig`]: A standard configuration for the models
//!
//! ### Traits
//!
//! This crate extends the [`Forward`](cnc::Forward) and [`Backward`](cnc::Backward) traits
//! from the [`core`](cnc) crate to provide additional functionality for neural networks.
//!
//! - [`Predict`]: A more robust implementation of the [`Forward`] trait
//! - [`Train`]: A trait for training a neural network model.
//!
#![cfg(feature = "alloc")]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::missing_saftey_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::upper_case_acronyms
)]
#[cfg(not(any(feature = "std", feature = "alloc")))]
compile_error! {
    "At least one of the 'std' or 'alloc' features must be enabled."
}

extern crate concision_core as cnc;

#[cfg(feature = "alloc")]
extern crate alloc;

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
    pub mod config;
    #[macro_use]
    pub mod seal;
}

pub mod config;
pub mod error;
pub mod layers;
pub mod model;

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

#[doc(inline)]
pub mod prelude {
    #[doc(no_inline)]
    pub use super::config::prelude::*;
    #[doc(no_inline)]
    pub use crate::layers::prelude::*;
    #[doc(no_inline)]
    pub use crate::model::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::*;
    #[doc(no_inline)]
    pub use crate::types::*;
}
