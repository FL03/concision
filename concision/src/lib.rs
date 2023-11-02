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
#[cfg(feature = "neural")]
pub use concision_neural as neural;
#[cfg(feature = "nlp")]
pub use concision_nlp as nlp;
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
    #[cfg(feature = "neural")]
    pub use concision_neural::prelude::*;
    #[cfg(feature = "nlp")]
    pub use concision_nlp::prelude::*;
    #[cfg(feature = "transformers")]
    pub use concision_transformers::prelude::*;
}
