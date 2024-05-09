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
#[doc(inline)]
pub use concision_data as data;
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[cfg(feature = "gnn")]
#[doc(inline)]
pub use concision_gnn as gnn;
#[cfg(feature = "kan")]
#[doc(inline)]
pub use concision_kan as kan;
#[cfg(feature = "linear")]
#[doc(inline)]
pub use concision_linear as linear;
#[cfg(feature = "macros")]
pub use concision_macros::*;

#[allow(unused_imports)]
pub mod prelude {
    pub use concision_core::prelude::*;
    #[cfg(feature = "data")]
    pub use concision_data::prelude::*;
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[cfg(feature = "gnn")]
    pub use concision_gnn::prelude::*;
    #[cfg(feature = "kan")]
    pub use concision_kan::prelude::*;
    #[cfg(feature = "linear")]
    pub use concision_linear::prelude::*;
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
}
