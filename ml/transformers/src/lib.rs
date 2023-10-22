/*
   Appellation: transformers <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Transformers
pub use self::{primitives::*, specs::*, utils::*};

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod attention;
pub mod codec;
pub mod ffn;
pub mod transform;

pub(crate) use concision_neural as neural;

pub mod prelude {
    pub use crate::attention::*;
    pub use crate::codec::*;
    pub use crate::transform::*;

    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;
}
