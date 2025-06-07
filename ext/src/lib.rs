/*
    Appellation: concision-models <library>
    Contrib: @FL03
*/
//! This crate extends the `concision` framework to provide a set of models and layers
#![cfg_attr(not(feature = "std"), no_std)]

// #[cfg(feature = "alloc")]
// extern crate alloc;

extern crate concision as cnc;

#[cfg(feature = "simple")]
pub mod simple;
#[cfg(feature = "transformer")]
pub use concision_transformer as transformer;

pub mod prelude {
    #[cfg(feature = "simple")]
    pub use crate::simple::SimpleModel;
    #[cfg(feature = "transformer")]
    pub use concision_transformer::TransformerModel;
}
