/*
    Appellation: impl_leaky <module>
    Created At: 2025.12.10:13:05:53
    Contrib: @FL03
*/
use super::Leaky;

use crate::snn::leaky::{LeakyState, LeakyParams};
use crate::snn::StepResult;
use num_traits::{Float, FromPrimitive};

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
        let params = LeakyParams {
            b,
            tau_m,
            tau_s,
            tau_w,
            resistance,
            v_rest,
            v_reset,
            v_thresh,
        };
        let state = LeakyState::from_v(v0);
        let min_dt = T::from_f32(1e-6).unwrap();

        Self {
            params,
            state,
            min_dt,
        }
    }
    /// returns a reference to the neuron's parameters
    pub const fn params(&self) -> &LeakyParams<T> {
        &self.params
    }
    /// returns a mutable reference to the neuron's parameters
    pub const fn params_mut(&mut self) -> &mut LeakyParams<T> {
        &mut self.params
    }
    /// returns a reference to the neuron's state
    pub const fn state(&self) -> &LeakyState<T> {
        &self.state
    }
    pub const fn state_mut(&mut self) -> &mut LeakyState<T> {
        &mut self.state
    }
    /// returns a reference to the neuron's adaptation variable (`w`)
    pub const fn adaptation(&self) -> &T {
        self.state().w()
    }
    /// returns a reference to the membrane potential, `v`, of the neuron
    pub const fn membrane_potential(&self) -> &T {
        self.state().v()
    }
    /// returns a reference to the current value, or synaptic state, of the neuron (`s`)
    pub const fn synaptic_state(&self) -> &T {
        self.state().s()
    }
    /// returns a reference to the adaptation increment, `b`, of the neuron
    pub const fn b(&self) -> &T {
        self.params().b()
    }
    /// returns a reference to the membrane resistance, `R`, of the neuron
    pub const fn resistance(&self) -> &T {
        self.params().resistance()
    }
    /// returns a reference to the membrane time constant, `tau_m`, of the neuron
    pub const fn tau_m(&self) -> &T {
        self.params().tau_m()
    }
    /// returns a reference to the synaptic time constant, `tau_s`, of the neuron
    pub const fn tau_s(&self) -> &T {
        self.params().tau_s()
    }
    /// returns a reference to the adaptation time constant, `tau_w`, of the neuron
    pub const fn tau_w(&self) -> &T {
        self.params().tau_w()
    }
    /// returns a reference to the spike threshold, `v_thresh`, of the neuron
    pub const fn v_thresh(&self) -> &T {
        self.params().v_thresh()
    }
    /// returns a reference to the reset potential, `v_reset`, of the neuron
    pub const fn v_reset(&self) -> &T {
        self.params().v_reset()
    }
    /// returns a reference to the resting membrane potential, `v_rest`, of the neuron
    pub const fn v_rest(&self) -> &T {
        self.params().v_rest()
    }
    /// Apply a presynaptic spike event to the neuron; this increments the synaptic variable `s`
    /// by `weight` instantaneously (models delta spike arrival).
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all, level = "trace"))]
    pub fn apply_spike(&mut self, weight: T)
    where
        T: core::ops::AddAssign,
    {
        self.state_mut().apply_spike(weight);
    }
    /// reset state variables (keeps parameters).
    pub fn reset_state(&mut self)
    where
        T: Default,
    {
        self.state_mut().reset();
    }
    /// Integrate the neuron state forward by `dt` [ms] using forward Euler; the externally
    /// applied current, `i_ext`, is added to the synaptic current `s` for the integration
    /// step. Therefore it is important to maintain unitary consistency between `i_ext` and `s`
    /// and to ensure that the provided `dt` is sufficiently small to avoid missing spikes, yet
    /// still greater than 0
    ///
    /// **Note**: This method checks for threshold crossing explicitly to avoid missing spikes
    /// due to large `dt`. Additionally, if `dt` is less than `min_dt`, it is clamped to
    /// `min_dt`.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip_all, level = "trace"))]
    pub fn step(&mut self, dt: T, i_ext: T) -> StepResult<T>
    where
        T: Float + FromPrimitive + core::ops::AddAssign,
    {
        let dt = if dt.is_sign_negative() {
            panic!("dt must be > 0")
        } else {
            dt.max(self.min_dt)
        };
        let LeakyParams {
            b,
            tau_m,
            tau_s,
            tau_w,
            resistance,
            v_rest,
            v_reset,
            v_thresh,
        } = self.params;
        let LeakyState { v, w, s } = self.state;
        // synaptic current is represented by `s`
        // ds/dt = -s / tau_s
        let ds = -s / tau_s;
        let s_next = s + dt * ds;
        // The total synaptic current for this step is given by `s` (for explicit Euler consistency).
        let i_syn = s;
        // evaluate dv/dt
        let dv = (-(v - v_rest) + resistance * (i_ext + i_syn) - w) / tau_m;
        let v_next = v + dt * dv;
        // adaptation dw/dt = -w / tau_w
        let dw = -w / tau_w;
        let w_next = w + dt * dw;
        // commit a new state
        self.state_mut().update(v_next, w_next, s_next);
        // check for crossing
        if v < v_thresh && v_next >= v_thresh {
            // apply reset and adaptation increment
            self.state_mut().set_v(v_reset).apply_adaptation(b);
            StepResult::spiked(v_next)
        } else {
            StepResult::not_spiked(v)
        }
    }
}

impl<T> Default for Leaky<T>
where
    T: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            params: LeakyParams::default(),
            state: LeakyState::default(),
            min_dt: T::from_f32(1e-6).unwrap(),
        }
    }
}