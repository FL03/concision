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
#[cfg(feature = "data")]
pub use concision_data as data;
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[cfg(feature = "macros")]
pub use concision_macros::*;
#[cfg(feature = "nn")]
pub use concision_nn as nn;
#[cfg(feature = "transformers")]
pub use concision_transformers as transformers;

pub mod prelude {
    #[cfg(feature = "core")]
    pub use concision_core::prelude::*;
    #[cfg(feature = "data")]
    pub use concision_data::prelude::*;
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
    #[cfg(feature = "nn")]
    pub use concision_nn::prelude::*;
    #[cfg(feature = "transformers")]
    pub use concision_transformers::prelude::*;
}
