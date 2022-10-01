/*
   Appellation: concision <library>
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       Concision is a robust framework for creating powerful data-centric applications in Rust.
*/
#[doc(inline)]
#[cfg(feature = "core")]
pub use crate::{actors::*, contexts::*, core::*, data::*};
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[cfg(feature = "macros")]
pub use concision_macros::*;

mod actors;
mod contexts;
mod core;
mod data;

pub mod prelude {
    #[cfg(feature = "core")]
    use super::*;
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
}
