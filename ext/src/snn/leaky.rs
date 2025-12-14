/*
    Appellation: leaky <module>
    Created At: 2025.11.25:09:33:30
    Contrib: @FL03
*/

mod impl_leaky;
mod impl_leaky_params;
mod impl_leaky_state;

/// An implementation of a leaky integrate-and-fire (LIF) neuron with an adaptation term and
/// exponential synaptic current. Generally speaking, an intergate-and-fire neuron integrates
/// the input current over time until the membrane potential reaches a certain threshold,
/// at which point it emits a spike and resets its membrane potential. The _leaky_ term speaks
/// to the decay of the membrane potential over time, simulating the effect of a leaky
/// capacitor.
///
/// ## Model
///
/// Here, we describe the dynamics of a leaky integrate-and-fire (LIF) neuron with an
/// adaptation term and exponential synaptic current. The neuron's behavior is governed by the
/// following set of equations:
///
/// ```math
/// \begin{aligned}
/// \tau_{m}\cdot{\frac{dv}{dt}} &= -(v - v_{rest}) + R\cdot{(I_{ext} + I_{syn})} - \omega \\
/// \tau_{w}\cdot{\frac{d\omega}{dt}} &= -\omega \\
/// \tau_{s}\cdot{\frac{ds}{dt}} &= -s
/// \end{aligned}
/// ```
///
/// ## Design
///
/// The implementation consists of three main components: a parameter struct (`LeakyParams`),
/// a state struct (`LeakyState`), and the main neuron struct (`Leaky`) that combines both
/// parameters and state. The `Leaky` struct provides methods to update the neuron's state based
/// on the input current and time step, check for spikes, and reset the neuron after a spike.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
#[repr(C)]
pub struct Leaky<T = f32> {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub params: LeakyParams<T>,
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub state: LeakyState<T>,
    /// Minimum allowed dt for integration [ms]
    pub min_dt: T,
}

/// The params of a leaky integrate-and-fire (LIF) neuron
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(rename_all = "snake_case")
)]
#[repr(C)]
pub struct LeakyParams<T = f32> {
    /// the adaptation increment `b` added to `w` on spike [same units as the current]
    #[cfg_attr(feature = "serde", serde(alias = "adaptation_increment"))]
    pub b: T,
    /// the membrane time constant [ms]
    pub tau_m: T,
    /// the synaptic time constant [ms]
    pub tau_s: T,
    /// the time constant for adaptation [ms]
    pub tau_w: T,
    /// the resistance, `R` of the membrane [MΩ or equivalent units]  
    #[cfg_attr(feature = "serde", serde(alias = "r"))]
    pub resistance: T,
    /// the resting potential of the neuron [mV]
    #[cfg_attr(feature = "serde", serde(alias = "resting_potential"))]
    pub v_rest: T,
    /// the spike reset potential [mV]
    #[cfg_attr(feature = "serde", serde(alias = "reset_potential"))]
    pub v_reset: T,
    /// threshold potential for a spike [mV]
    #[cfg_attr(feature = "serde", serde(alias = "threshold_potential"))]
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
    /// the neuron's membrane potential
    #[cfg_attr(feature = "serde", serde(alias = "membrane_potential"))]
    pub v: T,
    /// the adaptation variable of the neuron
    #[cfg_attr(feature = "serde", serde(alias = "adaptation"))]
    pub w: T,
    /// total synaptic current
    #[cfg_attr(
        feature = "serde",
        serde(alias = "synaptic_current", alias = "synaptic_state")
    )]
    pub s: T,
}
