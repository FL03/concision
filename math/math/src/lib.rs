/*
   Appellation: concision-math <lib>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # concision-math

pub use self::{primitives::*, specs::*, utils::*};

pub mod calculus;

pub use concision_linalg as linalg;

pub use concision_num as num;

pub use concision_statistics as stats;

// pub(crate) use concision_core as core;

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod prelude {

    pub use concision_linalg::prelude::*;
    pub use concision_num::prelude::*;
    pub use concision_statistics::prelude::*;

    pub use crate::calculus::*;
    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
