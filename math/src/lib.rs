/*
   Appellation: math <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Math
pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod regress;

pub mod prelude {
    pub use crate::regress::*;

    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
