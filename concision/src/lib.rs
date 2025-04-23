/*
   Appellation: concision <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision
//!
//! Concision focuses on providing useful abstractions for building advanced neural network
//! models in pure rust.
#![allow(unused_imports)]
#![crate_name = "concision"]

#[doc(inline)]
pub use concision_core::*;
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[cfg(feature = "macros")]
pub use concision_macros::*;
#[doc(inline)]
#[cfg(feature = "neural")]
pub use concision_neural as nn;

pub mod prelude {
    pub use concision_core::prelude::*;
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
    #[cfg(feature = "neural")]
    pub use concision_neural::prelude::*;
}
