/*
   Appellation: core <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![crate_name = "concision_core"]

pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;

pub(crate) mod utils;

pub mod errors;
pub mod ops;
pub mod params;
pub mod specs;
pub mod time;

pub mod prelude {

    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;

    pub use crate::errors::*;
    pub use crate::time::*;
}
