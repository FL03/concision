/*
   Appellation: data <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Data
//!
//!
//!
// #![feature(associated_type_defaults)]
extern crate concision_core as concision;

pub use self::{dataset::Dataset, traits::prelude::*, types::prelude::*, utils::*};

pub(crate) mod utils;

pub mod dataset;
pub mod traits;
pub mod types;

pub mod prelude {
    pub use crate::utils::*;

    pub use crate::dataset::*;
    pub use crate::traits::prelude::*;
    pub use crate::types::prelude::*;
}
