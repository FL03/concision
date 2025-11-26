/*
    appellation: snn <module>
    authors: @FL03
*/
//! Spiking neural networks (SNNs) for the [`concision`](https://crates.io/crates/concision) machine learning framework.
//!
//! ## References
//!
//! - [Deep Learning in Spiking Neural Networks](https://arxiv.org/abs/1804.08150)
//!
#[doc(inline)]
pub use self::{model::*, neuron::*, types::*};

mod model;
mod neuron;

pub mod types {
    //! Types for spiking neural networks
    #[doc(inline)]
    pub use self::{event::*, result::*};

    mod event;
    mod result;
}

pub(crate) mod prelude {
    pub use super::model::*;
    pub use super::neuron::*;
    pub use super::types::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snn_neuron_resting_no_input() {
        let mut n = SpikingNeuron::default();
        let dt = 1.0;
        // simulate 100 ms with no input -> should not spike and v near v_rest
        for _ in 0..100 {
            let res = n.step(dt, 0.0);
            assert!(!res.is_spiked());
        }
        let v = n.membrane_potential();
        assert!((v - n.v_rest).abs() < 1e-6 || (v - n.v_rest).abs() < 1e-2);
    }

    #[test]
    // #[ignore = "Need to fix"]
    fn test_snn_neuron_spikes() {
        // params
        let dt = 1f64;
        let i_ext = 50f64; // large i_ext to force spiking
        // neuron
        let mut n = SpikingNeuron::default();
        let mut spiked = false;
        let mut steps = 0usize;
        // apply strong constant external current for a while
        while !spiked && steps < 1000 {
            spiked = n.step(dt, i_ext).is_spiked();
            steps += 1;
        }
        assert!(spiked, "Neuron did not spike under strong current");
    }
}
