/*
   Appellation: optim <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Optim
pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub(crate) use concision_core as core;
pub(crate) use concision_neural as neural;

pub mod cost;
pub mod grad;
pub mod optimizer;

pub mod prelude {
    pub use crate::grad::*;

    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
