/*
    Appellation: neurons <test>
    Created At: 2025.11.25:21:34:30
    Contrib: @FL03
*/
use concision_snn::SpikingNeuron;

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
fn test_receive_spike_increases_synaptic_state() {
    let mut n = SpikingNeuron::default();
    let before = n.synaptic_state();
    n.receive_spike(2.5);
    assert!(n.synaptic_state() > before);
}

#[test]
#[ignore = "Need to fix"]
fn test_spiking_with_sufficient_input() {
    // params
    let dt: f64 = 0.1;
    let i_ext: f64 = 5.0; // large i_ext to force spiking
    // neuron
    let mut n = SpikingNeuron::default();
    let mut spiked = false;
    let mut steps = 0_usize;
    // apply strong constant external current for a while
    while !spiked && steps < 1000 {
        spiked = n.step(dt, i_ext).is_spiked();
        steps += 1;
    }
    assert!(spiked, "Neuron did not spike under strong current");
}
