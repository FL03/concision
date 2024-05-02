/*
   Appellation: optim <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Optim
//!

extern crate concision_core as concision;
extern crate concision_neural as neural;
pub use self::{optimizer::*, primitives::*, specs::*, utils::*};

pub(crate) mod optimizer;
pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod grad;
pub mod norm;
pub mod params;

pub mod prelude {
    pub use crate::grad::*;
    pub use crate::norm::*;
    pub use crate::params::*;

    pub use crate::optimizer::*;
    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
