/*
   Appellation: concision <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision
//!
//! Concision aims to be a complete machine learning library written in pure Rust.
//!
#![allow(unused_imports)]

#![crate_name = "concision"]

#[doc(inline)]
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
#[cfg(feature = "s4")]
#[doc(inline)]
pub use concision_s4 as s4;
#[cfg(feature = "transformer")]
#[doc(inline)]
pub use concision_transformer as transformer;

pub mod prelude {
    #[cfg(feature = "gnn")]
    pub use crate::gnn::prelude::*;
    #[cfg(feature = "kan")]
    pub use crate::kan::prelude::*;
    #[cfg(feature = "linear")]
    pub use crate::linear::prelude::*;
    #[cfg(feature = "s4")]
    pub use crate::s4::prelude::*;
    #[cfg(feature = "transformer")]
    pub use crate::transformer::prelude::*;
    pub use concision_core::prelude::*;
    #[cfg(feature = "data")]
    pub use concision_data::prelude::*;
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
}
