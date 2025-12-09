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
//! ### Model (forward-Euler integration; units are arbitrary but consistent):
//!
//! ```math
//! \tau_m * \frac{dv}{dt} = -(v - v_{rest}) + R*(I_{ext} + I_{syn}) - \omega
//! ```
//!
//! ```math
//! \tau_w * \frac{d\omega}{dt} = -\omega
//! ```
//!
//! ```math
//! \tau_s * \frac{ds}{dt} = -s
//! ```
//!
//! where:
//!     - $`v`$: membrane potential
//!     - $`\omega`$: adaptation variable
//!     - $`s`$: synaptic variable representing total synaptic current
//!
//! If we allow the spike to be represented as $`\delta`$, then:
//!
//! ```math
//! v\geq{v_{thresh}}\rightarrow{\delta},v\leftarrow{v_{reset}},\omega\mathrel{+}=b
//! ```
//!
//! where $`b` is the adaptation increment added on spike. The synaptic current is given by:
//!
//! ```math
//! I_{syn} = s
//! ```
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
