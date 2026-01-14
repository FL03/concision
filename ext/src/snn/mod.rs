//! An implementation of Spiking Neural Networks (SNNs) in Rust.
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
//! ## References
//!
//! - [Deep Learning in Spiking Neural Networks](https://arxiv.org/abs/1804.08150)
//!
#[doc(inline)]
pub use self::{leaky::*, types::*, utils::*};

#[cfg(feature = "complex")]
pub mod complex_neuron;
pub mod leaky;

mod utils;

mod types {
    #[doc(inline)]
    pub use self::{event::*, result::*};

    mod event;
    mod result;
}

pub(crate) mod prelude {
    pub use super::leaky::Leaky;
}

#[cfg(test)]
mod tests {
    use super::Leaky;

    #[test]
    fn test_leaky_neuron_resting() {
        let mut n = Leaky::<f64>::default();
        let dt: f64 = 1.0;
        // simulate 100 ms with no input -> should not spike and v near v_rest
        for _ in 0..100 {
            let res = n.step(dt, 0.0);
            assert!(!res.is_spiked());
        }
        let v = n.membrane_potential();
        assert!(
            (v - n.v_rest()).abs() < 1e-5,
            "v = {v}, v_rest = {}",
            n.v_rest()
        );
    }

    #[test]
    fn test_leaky_neuron_spiking() {
        // params
        let dt = 1f64;
        let i_ext = 50f64; // large i_ext to force spiking
        // neuron
        let mut n = Leaky::default();
        let mut spiked = false;
        let mut steps = 0usize;
        // run until spiked or max steps reached
        while !spiked && steps < 1000 {
            spiked = n.step(dt, i_ext).is_spiked();
            steps += 1;
        }
        assert!(
            spiked,
            "Neuron did not spike under a strong current (i_ext = {i_ext})"
        );
    }

    #[test]
    fn test_leaky_neuron_state_change() {
        let mut n = Leaky::default();
        let before = *n.synaptic_state();
        n.apply_spike(2.5);
        assert!(*n.synaptic_state() > before);
    }
}
