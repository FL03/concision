/*
    Appellation: snn <test>
    Created At: 2025.11.26:15:42:45
    Contrib: @FL03
*/
use approx::assert_abs_diff_eq;
use concision_ext::snn::SpikingNeuron;

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
    assert_abs_diff_eq!(v, n.v_rest);
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

#[test]
fn test_snn_neuron_synaptic_state_change() {
    let mut n = SpikingNeuron::default();
    let before = n.synaptic_state();
    n.apply_spike(2.5);
    assert!(n.synaptic_state() > before);
}
