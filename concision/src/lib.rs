/*
   Appellation: concision <library>
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       Concision is a robust framework for creating powerful data-centric applications in Rust.
*/
#[doc(inline)]
pub use crate::{actors::*, components::*, core::*, data::*};
#[cfg(feature = "derive")]
pub use concision_derive::*;
#[cfg(feature = "macros")]
pub use concision_macros::*;

mod actors;
mod components;
mod core;
mod data;

pub mod prelude {
    use crate::{
        actors::{aggregators::*, automata::*, converters::*, transformers::*},
        components::{forms::*, points::*, tables::*},
        core::*,
        data::{handlers::*, models::*, schemas::*},
    };
    #[cfg(feature = "derive")]
    pub use concision_derive::*;
    #[cfg(feature = "macros")]
    pub use concision_macros::*;
}