/*
   Appellation: neural <lib>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-neural
pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod layers;
pub mod neurons;
pub mod nn;
pub mod prop;

// pub(crate) use concision_core as core;

pub mod prelude {
    pub use crate::layers::*;
    pub use crate::neurons::*;
    pub use crate::nn::*;
    pub use crate::prop::*;

    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
