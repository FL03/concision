/*
   Appellation: concision-linear <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Linear Models
//!
//! This library implements the framework for building linear models.
//!
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(no_std)]
extern crate alloc;

extern crate concision_core as concision;
// extern crate concision_neural as neural;
extern crate ndarray as nd;
extern crate ndarray_rand as ndrand;
extern crate ndarray_stats as stats;

pub use self::model::{Config, Features, Linear, LinearParams};
pub use self::{neurons::*, traits::*};

pub mod conv;
pub mod dense;
pub mod model;
pub mod neurons;
pub mod traits;

pub mod prelude {
    pub use crate::model::prelude::*;
    pub use crate::neurons::Perceptron;
    pub use crate::traits::*;
}
