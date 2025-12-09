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
}

impl<T> Default for LeakyParams<T>
where
    T: FromPrimitive + One + core::ops::Neg<Output = T>,
{
    fn default() -> Self {
        Self {
         tau_m: T::from_usize(20).unwrap(), // ms
        resistance: T::one(), // arbitrary
        v_rest: T::from_usize(65).unwrap().neg(), // mV
        v_thresh: T::from_usize(50).unwrap().neg(), // mV
        v_reset: T::from_usize(65).unwrap().neg(), // mV
        tau_w: T::from_usize(200).unwrap(), // ms (slow adaptation)
        b: T::from_f32(0.5).unwrap(), // adaptation increment
        tau_s: T::from_usize(5).unwrap(), // ms (fast synapse)
        }
    }
}
