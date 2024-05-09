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
extern crate ndarray as nd;
#[cfg(feature = "rand")]
extern crate ndarray_rand as ndrand;
extern crate ndarray_stats as stats;

pub use self::model::{Config, Features, Layout, Linear};
pub use self::params::LinearParams;
#[allow(unused_imports)]
pub use self::{traits::*, utils::*};

pub(crate) mod utils;

#[doc(hidden)]
pub mod conv;
#[doc(hidden)]
pub mod dense;
#[doc(hidden)]
pub mod mlp;
pub mod model;
pub mod params;
pub mod traits;

pub mod prelude {
    pub use crate::model::prelude::*;
    pub use crate::params::prelude::*;
    pub use crate::traits::*;
}
