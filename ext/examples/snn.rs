/*
    Appellation: snn <module>
    Created At: 2025.12.08:15:27:07
    Contrib: @FL03
*/
//! Minimal demonstration of neuron usage. Simulates a neuron for `t_sim` ms with dt,
//! injects a constant external current `i_ext`, and injects discrete synaptic events at specified times.
use concision_ext::snn::{SpikingNeuron, SynapticEvent};

fn main() -> anyhow::Result<()> {
    // Simulation parameters
    let dt = 0.1; // ms
    let t_sim = 5000.0; // ms
    let steps = (t_sim / dt) as usize;

    // Create neuron with defaults
    let mut neuron = SpikingNeuron::default();

    // Example external current (constant)
    let i_ext = 2.8; // tune to see spiking (units consistent with resistance & s)

    // Example presynaptic spike times (ms) and weights
    let presyn_spikes: Vec<(f64, f64)> =
        vec![(50.0, 2.0), (100.0, 1.5), (150.0, 2.2), (300.0, 3.0)];

    // Convert into an index-able event list
    let mut events: Vec<Vec<SynapticEvent>> = vec![Vec::new(); steps + 1];
    for (t_spike, weight) in presyn_spikes {
        let idx = (t_spike / dt).round() as isize;
        if idx >= 0 && (idx as usize) < events.len() {
            events[idx as usize].push(SynapticEvent { weight });
        }
    }

    // Simulation loop
    let mut spike_times: Vec<f64> = Vec::new();
    for step in 0..steps {
        let t = step as f64 * dt;

        // deliver presynaptic events scheduled for this time step
        for ev in &events[step] {
            neuron.apply_spike(ev.weight);
        }

        // step the neuron
        let res = neuron.step(dt, i_ext);

        if res.is_spiked() {
            spike_times.push(t);
            // For debugging: print spike time
            println!("Spike at {:.3} ms (v reset = {:.3})", t, neuron.v);
        }

        // optionally, record v, w, s for analysis (omitted here for brevity)
        let _v = neuron.membrane_potential();
        let _w = neuron.adaptation();
        let _s = neuron.synaptic_state();

        // small example of printing membrane potential every 50 ms
        if step % ((50.0 / dt) as usize) == 0 {
            println!("t={:.1} ms, v={:.3} mV, w={:.3}, s={:.3}", t, _v, _w, _s);
        }
    }

    println!("Total spikes: {}", spike_times.len());
    println!("Spike times: {:?}", spike_times);

    Ok(())
}
