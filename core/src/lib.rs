/*
   Appellation: core <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-core
//!
//! This crate is the core library for the Concision project. It provides the basic building blocks for creating neural networks and other machine learning models.
#![cfg_attr(not(feature = "std"), no_std)]
#![crate_name = "concision_core"]

#[cfg(feature = "alloc")]
extern crate alloc;

extern crate concision_math as math;
extern crate ndarray as nd;
#[cfg(feature = "rand")]
extern crate ndarray_rand as ndrand;

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
