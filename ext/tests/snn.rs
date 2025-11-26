/*
    Appellation: snn <test>
    Created At: 2025.11.26:15:42:45
    Contrib: @FL03
*/
use concision_ext::snn::SpikingNeuron;

#[test]
fn test_snn_neuron_synaptic_state_change() {
    let mut n = SpikingNeuron::default();
    let before = n.synaptic_state();
    n.receive_spike(2.5);
    assert!(n.synaptic_state() > before);
}
