/*
   Appellation: data <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Data
pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod df;
pub mod flows;
pub mod tensors;

pub mod prelude {
    pub use crate::df::*;
    pub use crate::flows::*;
    pub use crate::tensors::*;

    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
