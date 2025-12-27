/*
    Appellation: impl_leaky_params <module>
    Created At: 2025.12.09:15:38:54
    Contrib: @FL03
*/
use super::LeakyParams;
use num_traits::{FromPrimitive, One};

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
    /// returns a mutable reference to the adaptation increment, `b`, of the neuron
    pub const fn b_mut(&mut self) -> &mut T {
        &mut self.b
    }
    /// returns a reference to the membrane resistance, `R`, of the neuron
    pub const fn resistance(&self) -> &T {
        &self.resistance
    }
    /// returns a mutable reference to the membrane resistance, `R`, of the neuron
    pub const fn resistance_mut(&mut self) -> &mut T {
        &mut self.resistance
    }
    /// returns a reference to the membrane time constant, `tau_m`, of the neuron
    pub const fn tau_m(&self) -> &T {
        &self.tau_m
    }
    /// returns a reference to the mutable membrane time constant, `tau_m`, of the neuron
    pub const fn tau_m_mut(&mut self) -> &mut T {
        &mut self.tau_m
    }
    /// returns a reference to the synaptic time constant, `tau_s`, of the neuron
    pub const fn tau_s(&self) -> &T {
        &self.tau_s
    }
    /// returns a mutable reference to the synaptic time constant, `tau_s`, of the neuron
    pub const fn tau_s_mut(&mut self) -> &mut T {
        &mut self.tau_s
    }
    /// returns a reference to the adaptation time constant, `tau_w`, of the neuron
    pub const fn tau_w(&self) -> &T {
        &self.tau_w
    }
    /// returns a mutable reference to the adaptation time constant, `tau_w`, of the neuron
    pub const fn tau_w_mut(&mut self) -> &mut T {
        &mut self.tau_w
    }
    /// returns a reference to the reset potential, `v_reset`, of the neuron
    pub const fn v_reset(&self) -> &T {
        &self.v_reset
    }
    /// returns a mutable reference to the reset potential, `v_reset`, of the neuron
    pub const fn v_reset_mut(&mut self) -> &mut T {
        &mut self.v_reset
    }
    /// returns a reference to the resting membrane potential, `v_rest`, of the neuron
    pub const fn v_rest(&self) -> &T {
        &self.v_rest
    }
    /// returns a mutable reference to the resting membrane potential, `v_rest`, of the neuron
    pub const fn v_rest_mut(&mut self) -> &mut T {
        &mut self.v_rest
    }
    /// returns a reference to the spike threshold, `v_thresh`, of the neuron
    pub const fn v_thresh(&self) -> &T {
        &self.v_thresh
    }
    /// returns a mutable reference to the spike threshold, `v_thresh`, of the neuron
    pub const fn v_thresh_mut(&mut self) -> &mut T {
        &mut self.v_thresh
    }
    /// consumes the current instance to create another with the given adaptation
    pub fn with_b(self, b: T) -> Self {
        Self { b, ..self }
    }
    /// consumes the current instance to create another with the given resistance
    pub fn with_resistance(self, resistance: T) -> Self {
        Self { resistance, ..self }
    }
    /// consumes the current instance to create another with the given membrane time constant
    pub fn with_tau_m(self, tau_m: T) -> Self {
        Self { tau_m, ..self }
    }

    pub fn with_tau_s(self, tau_s: T) -> Self {
        Self { tau_s, ..self }
    }
}

impl<T> Default for LeakyParams<T>
where
    T: FromPrimitive + One + core::ops::Neg<Output = T>,
{
    fn default() -> Self {
        Self {
            b: T::from_f32(0.5).unwrap(),               // adaptation increment
            resistance: T::one(),                       // arbitrary
            tau_m: T::from_usize(20).unwrap(),          // ms
            tau_s: T::from_usize(5).unwrap(),           // ms (fast synapse)
            tau_w: T::from_usize(200).unwrap(),         // ms (slow adaptation)
            v_reset: T::from_usize(65).unwrap().neg(),  // mV
            v_rest: T::from_usize(65).unwrap().neg(),   // mV
            v_thresh: T::from_usize(50).unwrap().neg(), // mV
        }
    }
}

impl<T> core::fmt::Display for LeakyParams<T>
where
    T: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{{ b: {}, resistance: {}, tau_m: {}, tau_s: {}, tau_w: {}, v_rest: {}, v_reset: {}, v_thresh: {} }}",
            self.b,
            self.resistance,
            self.tau_m,
            self.tau_s,
            self.tau_w,
            self.v_rest,
            self.v_reset,
            self.v_thresh
        )
    }
}

impl<T> core::ops::Index<&str> for LeakyParams<T> {
    type Output = T;

    fn index(&self, index: &str) -> &Self::Output {
        match index {
            "b" | "adaptation_increment" => &self.b,
            "resistance" | "r" => &self.resistance,
            "tau_m" | "membrane_time_constant" => &self.tau_m,
            "tau_s" | "synaptic_time_constant" => &self.tau_s,
            "tau_w" | "adaptation_time_constant" => &self.tau_w,
            "v_reset" | "reset_potential" => &self.v_reset,
            "v_rest" | "resting_potential" => &self.v_rest,
            "v_thresh" | "threshold_potential" => &self.v_thresh,
            _ => panic!("invalid index for LeakyParams: {}", index),
        }
    }
}
