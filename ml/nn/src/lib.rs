/*
   Appellation: nn <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-nn
pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod neurons;

// pub(crate) use concision_core as core;

pub mod prelude {
    pub use crate::neurons::*;
    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
