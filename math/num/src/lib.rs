/*
   Appellation: concision-math <lib>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-math

pub use self::{factorials::*, primitives::*, specs::*, utils::*};

pub mod num;

pub(crate) mod factorials;

// pub(crate) use concision_core as core;

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod prelude {
    pub use crate::factorials::*;
    pub use crate::num::*;
    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
