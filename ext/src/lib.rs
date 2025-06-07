/*
    Appellation: concision-models <library>
    Contrib: @FL03
*/
//! This crate extends the `concision` framework to provide a set of models and layers
extern crate concision_core as cnc;

#[cfg(feature = "simple")]
pub mod simple;
#[cfg(feature = "transformer")]
pub use concicion_transformer as transformer;

pub mod prelude {
    #[cfg(feature = "simple")]
    pub use crate::simple::SimpleModel;
    #[cfg(feature = "transformer")]
    pub use concision_transformer::TransformerModel;
}
