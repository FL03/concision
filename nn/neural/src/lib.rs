/*
   Appellation: neural <lib>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-neural
//!
//! This library implements the neural network primitives and specifications.
//!
#![feature(fn_traits, unboxed_closures)]
extern crate concision_core as concision;

pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod errors;
pub mod func;
pub mod layers;
pub mod models;
pub mod neurons;
pub mod nn;
pub mod ops;
pub mod params;

#[doc(hidden)]
pub mod exp;

pub(crate) use concision as core;

pub mod prelude {
    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;

    pub use crate::errors::*;
    pub use crate::func::{activate::*, loss::*, rms::*};
    pub use crate::layers::*;
    pub use crate::neurons::*;
    pub use crate::nn::*;
    pub use crate::ops::*;
    pub use crate::params::*;
}
