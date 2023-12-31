/*
   Appellation: core <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Core
pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;

pub(crate) mod utils;

pub mod errors;
pub mod id;
pub mod masks;
pub mod params;
pub mod specs;
pub mod states;
pub mod time;

pub trait Transform<T> {
    type Output;

    fn transform(&self, args: &T) -> Self::Output;
}

pub mod prelude {
    pub use super::Transform;

    pub use crate::primitives::*;
    pub use crate::utils::*;

    pub use crate::errors::*;
    pub use crate::id::*;
    pub use crate::masks::*;
    pub use crate::specs::*;
    pub use crate::states::*;
    pub use crate::time::*;
}
