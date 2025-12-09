//! Spiking neural networks (SNNs) for the [`concision`](https://crates.io/crates/concision) machine learning framework.
//!
//! ## References
//!
//! - [Deep Learning in Spiking Neural Networks](https://arxiv.org/abs/1804.08150)
//!
//! ## Background
//!
//! Spiking Neural Networks (SNNs) are a class of artificial neural networks that more closely
//! mimic the behavior of biological neurons compared to traditional artificial neural
//! networks. In SNNs, neurons communicate by sending discrete spikes (or action potentials)
//! rather than continuous values. This allows SNNs to capture temporal dynamics and
//! event-driven processing, making them suitable for tasks that involve time-series data
//! or require low-power computation.
//!
#[doc(inline)]
pub use self::{model::*, neurons::*, types::*, utils::*};

mod model;
mod neurons;
mod utils;

pub mod types {
    //! Types for spiking neural networks
    #[doc(inline)]
    pub use self::{event::*, result::*};

    mod event;
    mod result;
}

pub(crate) mod prelude {
    pub use super::model::*;
    pub use super::neurons::*;
    pub use super::types::*;
}
