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
use crate::snn::StepResult;
use num_traits::{Float, FromPrimitive, NumAssign, Zero};

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
pub struct Leaky<T = f32> {
    // ---- Parameters ----
    /// Membrane time constant $`\tau_{m}`$ (ms)
    pub tau_m: T,
    /// Membrane resistance `R` (MΩ or arbitrary)
    pub resistance: T,
    /// Resting potential $``v_{rest}`$ (mV)
    pub v_rest: T,
    /// Threshold potential $`v_{thresh}`$ (mV)
    pub v_thresh: T,
    /// Reset potential after spike $`v_{reset}`$ (mV)
    pub v_reset: T,

    /// Adaptation time constant $`\tau_{w}`$ (ms)
    pub tau_w: T,
    /// Adaptation increment added on spike `b` (same units as w/current)
    pub b: T,

    /// Synaptic time constant $`\tau_{s}`$ (ms)
    pub tau_s: T,

    // ---- State variables ----
    /// Membrane potential `v`
    pub v: T,
    /// Adaptation variable `w`
    pub w: T,
    /// Synaptic variable `s` representing total synaptic current
    pub s: T,

    /// Minimum allowed dt for integration (ms)
    pub min_dt: T,
}

impl<T> Leaky<T> {
    /// Create a neuron with explicit parameters and initial state.
    pub fn new(
        tau_m: T,
        resistance: T,
        v_rest: T,
        v_thresh: T,
        v_reset: T,
        tau_w: T,
        b: T,
        tau_s: T,
        initial_v: Option<T>,
    ) -> Self
    where
        T: Float + FromPrimitive,
    {
        let v0 = if let Some(v_init) = initial_v {
            v_init
        } else {
            v_rest
        };
        Self {
            tau_m,
            resistance,
            v_rest,
            v_thresh,
            v_reset,
            tau_w,
            b,
            tau_s,
            v: v0,
            w: T::zero(),
            s: T::zero(),
            min_dt: T::from_f32(1e-6).unwrap(),
        }
    }
    /// returns a reference to the neuron's adaptation variable (`w`)
    pub const fn adaptation(&self) -> &T {
        &self.w
    }
    /// returns a reference to the membrane potential, `v`, of the neuron
    pub const fn membrane_potential(&self) -> &T {
        &self.v
    }
    /// returns a reference to the current value, or synaptic state, of the neuron (`s`)
    pub const fn synaptic_state(&self) -> &T {
        &self.s
    }
    /// returns a reference to the membrane time constant, `tau_m`, of the neuron
    pub const fn tau_m(&self) -> &T {
        &self.tau_m
    }
    /// returns a reference to the membrane resistance, `R`, of the neuron
    pub const fn resistance(&self) -> &T {
        &self.resistance
    }
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all, level = "trace"))]
    /// Apply a presynaptic spike event to the neuron; this increments the synaptic variable `s`
    /// by `weight` instantaneously (models delta spike arrival).
    pub fn apply_spike(&mut self, weight: T)
    where
        T: NumAssign + Zero,
    {
        self.s += weight;
    }
    /// reset state variables (keeps parameters).
    pub fn reset_state(&mut self)
    where
        T: Clone + Default,
    {
        self.v = self.v_rest.clone();
        self.w = T::default();
        self.s = T::default();
    }
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all, level = "trace"))]
    /// Integrate the neuron state forward by `dt` [ms] using forward Euler; the externally
    /// applied current, `i_ext`, is added to the synaptic current `s` for the integration
    /// step. Therefore it is important to maintain unitary consistency between `i_ext` and `s`
    /// and to ensure that the provided `dt` is sufficiently small to avoid missing spikes, yet
    /// still greater than 0
    ///
    /// **Note**: This method checks for threshold crossing explicitly to avoid missing spikes
    /// due to large `dt`. Additionally, if `dt` is less than `min_dt`, it is clamped to
    /// `min_dt`.
    pub fn step(&mut self, dt: T, i_ext: T) -> StepResult<T>
    where
        T: Float + FromPrimitive + NumAssign,
    {
        let dt = if dt.is_sign_negative() {
            panic!("dt must be > 0")
        } else {
            dt.max(self.min_dt)
        };

        // remember previous membrane potential for crossing detection
        let v_prev = self.v;

        // synaptic current is represented by `s`
        // ds/dt = -s / tau_s
        let ds = -self.s / self.tau_s;
        let s_next = self.s + dt * ds;

        // total synaptic current for this step (use current s, or average between s and s_next)
        // we use s for explicit Euler consistency.
        let i_syn = self.s;

        // membrane dv/dt = (-(v - v_rest) + R*(i_ext + i_syn) - w) / tau_m
        let dv =
            (-(self.v - self.v_rest) + self.resistance * (i_ext + i_syn) - self.w) / self.tau_m;
        let v_next = self.v + dt * dv;

        // adaptation dw/dt = -w / tau_w
        let dw = -self.w / self.tau_w;
        let w_next = self.w + dt * dw;

        // Commit state tentatively
        self.v = v_next;
        self.w = w_next;
        self.s = s_next;

        // Check for threshold crossing (explicit crossing test to avoid misses)
        if v_prev < self.v_thresh && self.v >= self.v_thresh {
            // spike: capture pre-reset potential if that is expected by StepResult consumers
            let pre_spike_v = self.v;
            // apply reset and adaptation increment
            self.v = self.v_reset;
            self.w += self.b;
            StepResult {
                spiked: true,
                v: pre_spike_v,
            }
        } else {
            StepResult {
                spiked: false,
                v: self.v,
            }
        }
    }
}

impl<T> Default for Leaky<T>
where
    T: Float + FromPrimitive,
{
    fn default() -> Self {
        let tau_m = T::from_usize(20).unwrap(); // ms
        let resistance = T::one(); // arbitrary
        let v_rest = T::from_usize(65).unwrap().neg(); // mV
        let v_thresh = T::from_usize(50).unwrap().neg(); // mV
        let v_reset = T::from_usize(65).unwrap().neg(); // mV
        let tau_w = T::from_usize(200).unwrap(); // ms (slow adaptation)
        let b = T::from_f32(0.5).unwrap(); // adaptation increment
        let tau_s = T::from_usize(5).unwrap(); // ms (fast synapse)
        Self::new(
            tau_m, resistance, v_rest, v_thresh, v_reset, tau_w, b, tau_s, None,
        )
    }
}
