/*
   Appellation: concision <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision
//!
//! Concision aims to be a complete machine learning library written in pure Rust.
//!

#[cfg(feature = "core")]
pub use concision_core as core;
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[cfg(feature = "macros")]
pub use concision_macros::*;
#[cfg(feature = "math")]
pub use concision_math as math;
#[cfg(feature = "nn")]
pub use concision_nn as nn;

pub mod prelude {
    #[cfg(feature = "core")]
    pub use concision_core::prelude::*;
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
    #[cfg(feature = "math")]
    pub use concision_math::prelude::*;
    #[cfg(feature = "nn")]
    pub use concision_nn::prelude::*;
}
