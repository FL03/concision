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
pub use self::{traits::prelude::*, utils::*};

#[macro_use]
pub(crate) mod seal;
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

mod impls {
    pub mod model {
        pub mod impl_init;
        pub mod impl_linear;
        pub mod impl_model;
    }

    pub mod params {
        pub mod impl_params;
        pub mod impl_rand;
        pub mod impl_serde;
    }
}

pub mod prelude {
    pub use crate::model::prelude::*;
    pub use crate::params::prelude::*;
    pub use crate::traits::prelude::*;
}
