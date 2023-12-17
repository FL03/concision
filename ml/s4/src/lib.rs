/*
    Appellation: concision-s4 <library>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Structured State Space Sequential Models (S4)
//!
//!  
pub use self::{model::*, primitives::*, specs::*, utils::*};

pub(crate) mod model;
pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod ops;
pub mod params;
pub mod ssm;

pub(crate) use concision_core as core;
pub(crate) use concision_neural as neural;

pub mod prelude {
    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;

    pub use crate::model::*;
    pub use crate::ops::*;
    pub use crate::params::*;
    pub use crate::ssm::*;
}
