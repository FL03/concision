/*
   Appellation: concision-linear <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Linear Models
//!
//! This library implements the framework for building linear models.
//!
extern crate concision_core as concision;
extern crate concision_neural as neural;

pub use self::{cmp::*, traits::*};

pub mod conv;
pub mod dense;
pub mod model;
pub mod traits;

pub(crate) mod cmp {
    pub(crate) use self::prelude::*;

    pub mod features;
    pub mod neurons;
    pub mod params;

    pub(crate) mod prelude {
        pub use super::features::*;
        pub use super::neurons::*;
        pub use super::params::*;
    }
}

use ndarray::Array2;
use std::collections::HashMap;

pub(crate) type ModuleParams<K, V> = HashMap<K, Array2<V>>;

pub mod prelude {
    pub use crate::cmp::prelude::*;
    pub use crate::model::*;
    pub use crate::traits::*;
}
