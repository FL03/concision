/*
    appellation: concision-snn <library>
    authors: @FL03
*/
//! Spiking neural networks (SNNs) for the [`concision`](https://crates.io/crates/concision) machine learning framework.
//!
//! ## References
//!
//! - [Deep Learning in Spiking Neural Networks](https://arxiv.org/abs/1804.08150)
//!
#![crate_type = "lib"]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::module_inception)]

extern crate concision as cnc;
#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(not(any(feature = "std", feature = "alloc")))]
compiler_error! {
    "Either feature \"std\" or feature \"alloc\" must be enabled."
}

#[doc(inline)]
pub use self::{model::*, neuron::*, types::*};

pub mod model;
pub mod neuron;

pub mod types {
    //! Types for spiking neural networks
    #[doc(inline)]
    pub use self::prelude::*;

    mod event;
    mod result;

    pub(crate) mod prelude {
        pub use super::event::*;
        pub use super::result::*;
    }
}

pub mod prelude {
    #[doc(inline)]
    pub use crate::model::*;
    #[doc(inline)]
    pub use crate::neuron::*;
    #[doc(inline)]
    pub use crate::types::*;
}
