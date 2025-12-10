/*
    Appellation: impl_leaky_params <module>
    Created At: 2025.12.09:15:38:54
    Contrib: @FL03
*/
use super::LeakyParams;
use num_traits::{Float, FromPrimitive};

impl<T> LeakyParams<T> {
    /// Create a new `LeakyParams` with the given parameters.
    pub const fn new(
        b: T,
        tau_m: T,
        tau_s: T,
        tau_w: T,
        resistance: T,
        v_rest: T,
        v_reset: T,
        v_thresh: T,
    ) -> Self {
        Self {
            b,
            tau_m,
            tau_s,
            tau_w,
            resistance,
            v_rest,
            v_reset,
            v_thresh,
        }
    }
    /// returns a reference to the adaptation increment, `b`, of the neuron
    pub const fn b(&self) -> &T {
        &self.b
    }
    /// returns a reference to the membrane resistance, `R`, of the neuron
    pub const fn resistance(&self) -> &T {
        &self.resistance
    }
    /// returns a reference to the membrane time constant, `tau_m`, of the neuron
    pub const fn tau_m(&self) -> &T {
        &self.tau_m
    }
    /// returns a reference to the synaptic time constant, `tau_s`, of the neuron
    pub const fn tau_s(&self) -> &T {
        &self.tau_s
    }
    /// returns a reference to the adaptation time constant, `tau_w`, of the neuron
    pub const fn tau_w(&self) -> &T {
        &self.tau_w
    }
    /// returns a reference to the reset potential, `v_reset`, of the neuron
    pub const fn v_reset(&self) -> &T {
        &self.v_reset
    }
    /// returns a reference to the resting membrane potential, `v_rest`, of the neuron
    pub const fn v_rest(&self) -> &T {
        &self.v_rest
    }
    /// returns a reference to the spike threshold, `v_thresh`, of the neuron
    pub const fn v_thresh(&self) -> &T {
        &self.v_thresh
    }
}

impl<T> Default for LeakyParams<T>
where
    T: Float + FromPrimitive,
{
    fn default() -> Self {
        Self {
            b: T::from_f32(0.5).unwrap(),               // adaptation increment
            resistance: T::one(),                       // arbitrary
            tau_m: T::from_usize(20).unwrap(),          // ms
            tau_s: T::from_usize(5).unwrap(),           // ms (fast synapse)
            tau_w: T::from_usize(200).unwrap(),         // ms (slow adaptation)
            v_rest: T::from_usize(65).unwrap().neg(),   // mV
            v_thresh: T::from_usize(50).unwrap().neg(), // mV
            v_reset: T::from_usize(65).unwrap().neg(),  // mV
        }
    }
}
