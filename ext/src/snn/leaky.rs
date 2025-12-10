/*
    Appellation: leaky <module>
    Created At: 2025.11.25:09:33:30
    Contrib: @FL03
*/
//! A leaky integrate-and-fire (LIF) neuron implementation with adaptation and exponential
//! synaptic current.
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
//! - $`\tau_{m}`$: membrane time constant
//! - $`R`$: membrane resistance
//! - $`v_{rest}`$: resting potential
//! - $`I_{ext}`$: externally applied current
//! - $`I_{syn}`$: synaptic current
//! - $`\tau_{w}`$: adaptation time constant
//! - $`\tau_{s}`$: synaptic time constant
//!
//! - $`v`$: membrane potential
//! - $`\omega`$: adaptation variable
//! - $`s`$: synaptic variable representing total synaptic current
//!
//! If we allow the spike to be represented as $`\delta`$, then:
//!
//! ```math
//! v\geq{v_{thresh}}\rightarrow{\delta},v\leftarrow{v_{reset}},\omega\mathrel{+}=b
//! ```
//!
//! where $`b`$ is the adaptation increment added on spike. The synaptic current is given by:
//!
//! ```math
//! I_{syn} = s
//! ```

mod impl_leaky;
mod impl_leaky_params;
mod impl_leaky_state;

/// The params of a leaky integrate-and-fire (LIF) neuron
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
#[repr(C)]
pub struct LeakyParams<T = f32> {
    /// Adaptation increment added on spike `b` (same units as w/current)
    pub b: T,
    /// Membrane time constant $`\tau_{m}`$ (ms)
    pub tau_m: T,
    /// Synaptic time constant $`\tau_{s}`$ (ms)
    pub tau_s: T,
    /// Adaptation time constant $`\tau_{w}`$ (ms)
    pub tau_w: T,
    /// Membrane resistance `R` (MΩ or arbitrary)
    pub resistance: T,
    /// Resting potential $``v_{rest}`$ (mV)
    pub v_rest: T,
    /// Reset potential after spike $`v_{reset}`$ (mV)
    pub v_reset: T,
    /// Threshold potential $`v_{thresh}`$ (mV)
    pub v_thresh: T,
}

/// The state of a leaky integrate-and-fire (LIF) neuron
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
#[repr(C)]
pub struct LeakyState<T = f32> {
    /// The membrane potential `v`
    pub v: T,
    /// Adaptation `w`
    pub w: T,
    /// Total synaptic current `s`
    pub s: T,
}
/// A leaky integrate-and-fire (LIF) neuron with an adaptation term and exponential synaptic
/// current. The neuron's dynamics are governed by the following equations:
///
/// ```math
/// \frac{dv}{dt} = \frac{-(v - v_{rest}) + R \cdot{(i_{ext} + s)} - w}{\tau_{m}}
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
#[repr(C)]
pub struct Leaky<T = f32> {
    pub params: LeakyParams<T>,
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub state: LeakyState<T>,
    /// Minimum allowed dt for integration (ms)
    pub min_dt: T,
}
