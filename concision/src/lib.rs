/*
   Appellation: concision <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision
//!
//! Concision aims to be a complete machine learning library written in pure Rust.
//!
#![crate_name = "concision"]

pub use concision_core::*;
#[cfg(feature = "data")]
pub use concision_data as data;
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[cfg(feature = "gnn")]
pub use concision_gnn as gnn;
#[cfg(feature = "linear")]
pub use concision_linear as linear;
#[cfg(feature = "macros")]
pub use concision_macros::*;

pub mod prelude {
    pub use concision_core::prelude::*;
    #[cfg(feature = "data")]
    pub use concision_data::prelude::*;
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[cfg(feature = "gnn")]
    pub use concision_gnn::prelude::*;
    #[cfg(feature = "linear")]
    pub use concision_linear::prelude::*;
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
}
