/*
    Appellation: concision-neural <library>
    Contrib: @FL03
*/
//! Various components, implementations, and traits for creating neural networks. The crate
//! builds off of the [`concision_core`] crate, making extensive use of the [`ParamsBase`](cnc::ParamsBase)
//! type to define the parameters of layers within a network.
//!
//! ## Overview
//!
//! Neural networks are a fundamental part of machine learning, and this crate provides a
//! comprehensive set of tools to build, configure, and train neural network models. Listed
//! below are several key components of the crate:
//!
//! - [`Model`]: A trait for defining a neural network model.
//! - [`StandardModelConfig`]: A standard configuration for the models
//! - [`ModelFeatures`]: A default implementation of the [`ModelLayout`] trait that
//!   sufficiently defines both shallow and deep neural networks.
//!
//! ### _Model Parameters_
//!
//! Additionally, the crate defines a sequential
//!
//! **Note**: You should stick with the type aliases for the [`ModelParamsBase`] type, as they
//! drastically simplify the type-face of the model parameters. Attempting to generalize over
//! the hidden layers of the model might lead to excessive complexity. That being said, there
//! are provided methods and routines to convert from a shallow to deep model, and vice versa.
//!
//! - [`DeepModelParams`]: An owned representation of the [`ModelParamsBase`] for deep
//!   neural networks.
//! - [`ShallowModelParams`]: An owned representation of the [`ModelParamsBase`] for shallow
//!   neural networks.
//!
//! ### Traits
//!
//! This crate extends the [`Forward`](cnc::Forward) and [`Backward`](cnc::Backward) traits
//! from the [`core`](cnc) crate to provide additional functionality for neural networks.
//!
//! - [`Predict`]: A more robust implementation of the [`Forward`] trait
//! - [`Train`]: A trait for training a neural network model.
//!
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
    clippy::missing_saftey_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::upper_case_acronyms
)]
// ensure that either `std` or `alloc` feature is enabled
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
    layout::prelude::*,
    params::prelude::*,
    train::prelude::*,
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
pub mod layout;
pub mod params;
pub mod train;

pub(crate) mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    mod hidden;
    mod models;
    mod predict;

    mod prelude {
        #[doc(inline)]
        pub use super::hidden::*;
        #[doc(inline)]
        pub use super::models::*;
        #[doc(inline)]
        pub use super::predict::*;
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
    #[doc(hidden)]
    pub use crate::layers::prelude::*;
    #[doc(no_inline)]
    pub use crate::layout::prelude::*;
    #[doc(no_inline)]
    pub use crate::params::prelude::*;
    #[doc(no_inline)]
    pub use crate::train::prelude::*;
    #[doc(no_inline)]
    pub use crate::traits::*;
    #[doc(no_inline)]
    pub use crate::types::*;
}
