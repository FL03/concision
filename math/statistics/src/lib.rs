/*
   Appellation: concision-statistics <lib>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-statistics

pub use self::{primitives::*, specs::*, utils::*};

pub mod calc;
pub mod regression;

// pub(crate) use concision_num as num;

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod prelude {

    pub use crate::calc::*;
    pub use crate::regression::*;

    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
