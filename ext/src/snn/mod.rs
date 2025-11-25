/*
    Appellation: snn <module>
    Created At: 2025.11.25:07:03:05
    Contrib: @FL03
*/
//! # Spiked Neural Networks (SNN)
//! 
//! 

pub use self::neuron::SpikingNeuron;

pub mod neuron;

#[cfg(test)]
mod tests {
    use super::SpikingNeuron;

    #[test]
    fn test_resting_no_input() {
        let mut n = SpikingNeuron::new_default();
        let dt = 1.0;
        // simulate 100 ms with no input -> should not spike and v near v_rest
        for _ in 0..100 {
            let res = n.step(dt, 0.0);
            assert!(!res.spiked);
        }
        let v = n.membrane_potential();
        assert!((v - n.v_rest).abs() < 1e-6 || (v - n.v_rest).abs() < 1e-2);
    }

    #[test]
    fn test_receive_spike_increases_synaptic_state() {
        let mut n = SpikingNeuron::new_default();
        let before = n.synaptic_state();
        n.receive_spike(2.5);
        assert!(n.synaptic_state() > before);
    }

    #[test]
    fn test_spiking_with_sufficient_input() {
        let mut n = SpikingNeuron::new_default();
        let dt = 0.1;
        // apply strong constant external current for a while
        let mut spiked = false;
        for _ in 0..10000 {
            let res = n.step(dt, 5.0); // large i_ext to force spiking
            if res.spiked {
                spiked = true;
                break;
            }
        }
        assert!(spiked, "Neuron did not spike under strong current");
    }
}