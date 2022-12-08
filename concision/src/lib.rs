/*
   Appellation: concision <library>
   Contrib: FL03 <jo3mccain@icloud.com>
   Description:
       Concision is a robust framework for creating powerful data-centric applications in Rust.
*/
#[doc(inline)]
pub use crate::{primitives::*, utils::*};
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[cfg(feature = "macros")]
pub use concision_macros::*;

pub mod math;
pub mod num;

pub(crate) mod primitives;
pub(crate) mod utils;
