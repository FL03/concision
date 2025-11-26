/*
    appellation: snn <module>
    authors: @FL03
*/
//! Spiking neural networks (SNNs) for the [`concision`](https://crates.io/crates/concision) machine learning framework.
//!
//! ## References
//!
//! - [Deep Learning in Spiking Neural Networks](https://arxiv.org/abs/1804.08150)
//!
#[doc(inline)]
pub use self::{model::*, neuron::*, types::*};

mod model;
mod neuron;

pub mod types {
    //! Types for spiking neural networks
    #[doc(inline)]
    pub use self::{event::*, result::*};

    mod event;
    mod result;
}

pub(crate) mod prelude {
    pub use super::model::*;
    pub use super::neuron::*;
    pub use super::types::*;
}
