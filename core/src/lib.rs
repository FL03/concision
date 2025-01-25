/*
   Appellation: core <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! The core modules of the `concision` sdk implement the building blocks for neural networks and other machine learning models.
//!
//! ## Overview
//!
//! `concision` and its constituent modules are designed to be a lightweight, flexible, and efficient machine learning library built around
//! well-documented and tested abstractions. The core modules provide the following:
//!
//! - **Neural Network Layers**: A collection of neural network layers and activation functions.
//! - **Optimization Algorithms**: A collection of optimization algorithms for training neural networks.
//! - **Loss Functions**: A collection of loss functions for training neural networks.
//! - **Initialization Strategies**: A collection of initialization strategies for initializing neural network weights.
#![cfg_attr(not(feature = "std"), no_std)]
#![crate_name = "concision_core"]

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate ndarray as nd;

#[cfg(feature = "rand")]
extern crate ndarray_rand as ndrand;

#[doc(inline)]
pub use concision_math as math;

pub use self::error::{Error, ModelError, Result};
pub use self::func::Activate;
pub use self::nn::Module;
pub use self::{primitives::*, traits::prelude::*, types::prelude::*, utils::prelude::*};

#[cfg(feature = "rand")]
pub use self::init::{Initialize, InitializeExt};

#[macro_use]
pub(crate) mod macros;
pub(crate) mod primitives;

pub mod error;
pub mod func;
pub mod init;
pub mod nn;
pub mod ops;

pub mod traits;
pub mod types;
pub mod utils;

pub mod prelude {
    #[allow(unused_imports)]
    pub(crate) use super::primitives::rust::*;
    pub use concision_math::prelude::*;

    pub use super::error::*;
    pub use super::func::prelude::*;
    #[cfg(feature = "rand")]
    pub use super::init::prelude::*;
    pub use super::nn::prelude::*;
    pub use super::ops::prelude::*;
    pub use super::primitives::*;
    pub use super::traits::prelude::*;
    pub use super::types::prelude::*;
    pub use super::utils::prelude::*;
}
