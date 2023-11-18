/*
   Appellation: core <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Core
pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod epochs;
pub mod errors;
pub mod states;
pub mod step;



pub mod prelude {
    pub use crate::epochs::*;
    pub use crate::errors::*;
    pub use crate::states::*;
    pub use crate::step::*;

    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
