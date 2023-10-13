/*
   Appellation: concision <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision
//!
//! Concision aims to be a complete machine learning library written in pure Rust.
//!
#[doc(inline)]
pub use crate::{primitives::*, specs::*, utils::*};
#[cfg(feature = "core")]
pub use concision_core as core;
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[cfg(feature = "macros")]
pub use concision_macros::*;

pub mod math;
pub mod nn;
pub mod num;

pub(crate) mod primitives;
pub(crate) mod specs;
pub(crate) mod utils;

pub mod prelude {
    pub use crate::math::*;
    pub use crate::nn::*;
    pub use crate::num::*;
    pub use crate::primitives::*;
    pub use crate::specs::*;
    pub use crate::utils::*;

    #[cfg(feature = "core")]
    pub use concision_core::prelude::*;
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
}
